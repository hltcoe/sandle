import json
import logging
from itertools import chain
from time import time
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple
from uuid import uuid4

import torch
from flask import Flask, jsonify, Response, request
from transformers import AutoTokenizer, AutoModelForCausalLM
from waitress import serve


STREAM_TOKEN_BATCH_SIZE = 4
DEFAULT_MAX_TOKENS = 16
DEFAULT_TEMPERATURE = 1.
DEFAULT_TOP_P = 1.
EOS = '</s>'
DEFAULT_PROMPT = EOS
END_OF_STREAM = '[DONE]'
FINISH_REASON_EOS = 'stop'
FINISH_REASON_LENGTH = 'length'


class Completion(NamedTuple):
    text: str
    finish_reason: Optional[str]


class RawCompletion(NamedTuple):
    text: str
    pretruncation_num_new_tokens: int
    new_text: str
    truncated: bool


def clean_output_text(text: str) -> str:
    dirty_prefix = '</s>'
    return text[len(dirty_prefix):] if text.startswith(dirty_prefix) else text


def generate_response_id() -> str:
    return str(uuid4())


def get_timestamp() -> int:
    return int(time())


def truncate_at_stops(text: str, stop_strings: List[str]) -> Tuple[str, bool]:
    truncated = False
    for s in stop_strings:
        index = text.find(s)
        if index >= 0:
            text = text[:index]
            truncated = True
    return (text, truncated)


class LM:
    models: Dict[str, Tuple[AutoTokenizer, AutoModelForCausalLM]]
    device: str

    def __init__(self, preload_model: Optional[str] = None):
        self.models = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if preload_model is not None:
            self.get_tokenizer_and_model(preload_model)

    def get_tokenizer_and_model(self, model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        if model_id not in self.models:
            logging.info(f'Loading model: {model_id}')
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model.eval()
            model.to(self.device)
            self.models[model_id] = (tokenizer, model)
        return self.models[model_id]

    def complete(self, text: str, model_id: str,
                 stop_strings: List[str],
                 max_new_tokens: int = STREAM_TOKEN_BATCH_SIZE,
                 top_p: float = DEFAULT_TOP_P,
                 temperature: float = DEFAULT_TEMPERATURE) -> Completion:
        (tokenizer, model) = self.get_tokenizer_and_model(model_id)

        raw_completion = self._complete(
            text, tokenizer, model, stop_strings=stop_strings, max_new_tokens=max_new_tokens, top_p=top_p,
            temperature=temperature)

        finish_reason = FINISH_REASON_EOS if raw_completion.truncated else FINISH_REASON_LENGTH

        return Completion(raw_completion.new_text, finish_reason)

    def stream_complete(self, text: str, model_id: str,
                        stop_strings: List[str],
                        max_new_tokens: int = DEFAULT_MAX_TOKENS,
                        top_p: float = DEFAULT_TOP_P,
                        temperature: float = DEFAULT_TEMPERATURE,
                        token_batch_size: int = STREAM_TOKEN_BATCH_SIZE) -> Iterable[Completion]:
        (tokenizer, model) = self.get_tokenizer_and_model(model_id)

        prompt = text
        num_new_tokens = 0
        finish_reason = None
        while finish_reason is None:
            raw_completion = self._complete(
                text, tokenizer, model, stop_strings=stop_strings,
                max_new_tokens=min(token_batch_size, max_new_tokens - num_new_tokens),
                top_p=top_p, temperature=temperature,
            )

            if raw_completion.truncated:
                finish_reason = FINISH_REASON_EOS
            else:
                num_new_tokens += raw_completion.pretruncation_num_new_tokens
                if num_new_tokens >= max_new_tokens:
                    if num_new_tokens > max_new_tokens:
                        logging.warning('Generated more tokens than the max number specified')
                    finish_reason = FINISH_REASON_LENGTH

            # Check if a stop sequence spans the previous completion chunk and this one
            (truncated_text_after_prompt, truncated) = truncate_at_stops(
                raw_completion.text[len(prompt):],
                stop_strings)
            if truncated:
                yield Completion(
                    raw_completion.text[-len(raw_completion.new_text):len(prompt) + len(truncated_text_after_prompt)],
                    FINISH_REASON_EOS)
            else:
                yield Completion(raw_completion.new_text, finish_reason)

            text = raw_completion.text

    def _complete(self, text: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM,
                  stop_strings: List[str], max_new_tokens: int, top_p: float, temperature: float) -> RawCompletion:
        input_token_ids = tokenizer(text, return_tensors='pt')['input_ids']
        output_token_ids = model.generate(
            input_token_ids.to(self.device), max_new_tokens=max_new_tokens, do_sample=True, top_p=top_p,
            temperature=temperature)
        output_text = clean_output_text(tokenizer.decode(output_token_ids[0].tolist()))
        if output_text.startswith(text):
            new_text = output_text[len(text):]
            (new_text, truncated) = truncate_at_stops(new_text, stop_strings)
            output_text = text + new_text
            return RawCompletion(
                text=output_text,
                pretruncation_num_new_tokens=output_token_ids.size(dim=1) - input_token_ids.size(dim=1),
                new_text=new_text,
                truncated=truncated,
            )
        else:
            raise Exception(f'Generated text "{output_text}" does not begin with input text "{text}"')


def make_api_completion(response_id: str, created: int, model_id: str, completion: Completion) -> Dict[str, Any]:
    return {
        'id': response_id,
        'object': 'text_completion',
        'created': created,
        'model': model_id,
        'choices': [
            {
                'text': completion.text,
                'index': 0,
                'logprobs': None,
                'finish_reason': completion.finish_reason,
            }
        ],
        'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
    }


def create_app(preload_model: Optional[str]) -> Flask:
    logging.info('Loading model')
    lm = LM(preload_model)

    logging.info('Creating app')
    app = Flask(__name__)

    @app.route('/v1/completions', methods=['POST'])
    def post_completions():
        max_tokens = int(request.json.get('max_tokens', DEFAULT_MAX_TOKENS))

        model_id = request.json['model']

        prompt = request.json.get('prompt', DEFAULT_PROMPT)

        _stop = request.json.get('stop')
        if isinstance(_stop, list):
            stops = _stop
        elif isinstance(_stop, str):
            stops = [_stop]
        else:
            stops = []

        stream = request.json.get('stream', False)

        temperature = float(request.json.get('temperature', DEFAULT_TEMPERATURE))

        top_p = float(request.json.get('top_p', DEFAULT_TOP_P))

        user = request.json.get('user')

        completion_log_text = 'streaming completion' if stream else 'completion'
        tokens_log_text = 'token' if max_tokens == 1 else 'tokens'
        logging.debug(f'Computing {completion_log_text} of up to {max_tokens} {tokens_log_text} for user {user}')

        response_id = generate_response_id()
        created = get_timestamp()
        if stream:
            return Response(
                (f'data: {event_data}\n\n' for event_data in chain(
                    (
                        json.dumps(make_api_completion(response_id, created, model_id, completion))
                        for completion in lm.stream_complete(
                            prompt, model_id, stops, max_new_tokens=max_tokens, top_p=top_p, temperature=temperature)
                    ),
                    [END_OF_STREAM],
                )),
                mimetype='text/event-stream',
                headers={'X-Accel-Buffering': 'no'},  # tell nginx not to buffer
            )
        else:
            completion = lm.complete(
                prompt, model_id, stops, max_new_tokens=max_tokens, top_p=top_p, temperature=temperature)
            return jsonify(make_api_completion(response_id, created, model_id, completion))

    return app


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Run a simplified, single-threaded clone of OpenAI\'s /v1/completions endpoint',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--host', default='0.0.0.0',
                        help='Hostname or IP to serve on')
    parser.add_argument('-p', '--port', type=int, default=8000,
                        help='Port to serve on')
    parser.add_argument('--preload-model',
                        help='Repository of model to preload (example: facebook/opt-2.7b)')
    parser.add_argument('-l', '--log-level',
                        choices=('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'), default='INFO',
                        help='Logging verbosity level threshold (to stderr)')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                        level=args.log_level)

    app = create_app(args.preload_model)
    serve(app, host=args.host, port=args.port, threads=1)


if __name__ == '__main__':
    main()

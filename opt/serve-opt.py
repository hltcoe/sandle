import json
import logging
from itertools import chain
from time import time
from typing import Any, Dict, Iterable, NamedTuple, Optional, Tuple
from uuid import uuid4

from flask import Flask, jsonify, Response, request
from transformers import AutoTokenizer, AutoModelForCausalLM
from waitress import serve


STREAM_TOKEN_BATCH_SIZE = 4
DEFAULT_MAX_TOKENS = 16
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
    num_new_tokens: int


def clean_output_text(text: str) -> str:
    dirty_prefix = '</s>'
    return text[len(dirty_prefix):] if text.startswith(dirty_prefix) else text


def generate_response_id() -> str:
    return str(uuid4())


def get_timestamp() -> int:
    return int(time())


class LM:
    models: Dict[str, Tuple[AutoTokenizer, AutoModelForCausalLM]]

    def __init__(self, preload_model: Optional[str] = None):
        self.models = {}
        if preload_model is not None:
            self.get_tokenizer_and_model(preload_model)

    def get_tokenizer_and_model(self, model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        if model_id not in self.models:
            logging.info(f'Loading model: {model_id}')
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            model.eval()
            self.models[model_id] = (tokenizer, model)
        return self.models[model_id]

    def complete(self, text: str, model_id: str,
                 stop_strings: Iterable[str],
                 max_new_tokens: int = STREAM_TOKEN_BATCH_SIZE) -> Completion:
        (tokenizer, model) = self.get_tokenizer_and_model(model_id)

        logging.info(f'Generating completion of up to {max_new_tokens} tokens for {len(text)}-character prompt')
        raw_completion = self._complete(text, tokenizer, model, max_new_tokens)
        output_text = raw_completion.text

        finish_reason = FINISH_REASON_LENGTH
        for s in stop_strings:
            index = output_text.find(s)
            if index >= 0:
                output_text = output_text[:index]
                finish_reason = FINISH_REASON_EOS

        return Completion(output_text, finish_reason)

    def stream_complete(self, text: str, model_id: str,
                        stop_strings: Iterable[str],
                        max_new_tokens: int = DEFAULT_MAX_TOKENS,
                        token_batch_size: int = STREAM_TOKEN_BATCH_SIZE) -> Generator[Completion]:
        (tokenizer, model) = self.get_tokenizer_and_model(model_id)

        logging.info(f'Generating completion of up to {max_new_tokens} tokens for {len(text)}-character prompt')
        num_new_tokens = 0
        finish_reason = None
        while finish_reason is None:
            raw_completion = self._complete(
                text, tokenizer, model, min(token_batch_size, max_new_tokens - num_new_tokens)
            )
            output_text = raw_completion.text
            num_new_tokens += raw_completion.num_new_tokens

            if not output_text.startswith(text):
                raise Exception(f'Generated text "{output_text}" does begin with input text "{text}"')

            for s in stop_strings:
                index = output_text.find(s)
                if index >= 0:
                    output_text = output_text[:index]
                    finish_reason = FINISH_REASON_EOS
            if finish_reason is None and num_new_tokens == max_new_tokens:
                finish_reason = FINISH_REASON_LENGTH

            yield Completion(output_text[len(text):], finish_reason)

            text = output_text

    def _complete(self, text: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM,
                  max_new_tokens: int) -> RawCompletion:
        input_token_ids = tokenizer(text, return_tensors='pt')['input_ids']
        output_token_ids = model.generate(input_token_ids, max_new_tokens=max_new_tokens)
        output_text = clean_output_text(tokenizer.decode(output_token_ids[0].tolist()))
        return RawCompletion(output_text, output_token_ids.size(dim=1) - input_token_ids.size(dim=1))


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

        model_data = _get_model_data(request.json['model'])
        model_id = model_data['id']

        prompt = request.json.get('prompt', DEFAULT_PROMPT)

        _stop = request.json.get('stop', [])
        stops = _stop if isinstance(_stop, list) else [_stop]

        stream = request.json.get('stream', False)

        user = request.json.get('user')

        completion_log_text = 'streaming completion' if completion else 'completion'
        tokens_log_text = 'token' if max_tokens == 1 else 'tokens'
        logging.debug(f'Computing {completion_log_text} of up to {max_tokens} {tokens_log_text} for user {user}')

        response_id = generate_response_id()
        created = get_timestamp()
        if stream:
            return Response(
                (f'data: {event_data}\n\n' for event_data in chain(
                    (
                        json.dumps(make_api_completion(response_id, created, model_id, completion))
                        for completion in lm.stream_complete(prompt, model_id, stops, max_new_tokens=max_tokens)
                    ),
                    [END_OF_STREAM],
                )),
                mimetype='text/event-stream',
            )
        else:
            completion = lm.complete(prompt, model_id, stops, max_new_tokens=max_tokens)
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

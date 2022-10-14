import json
import gc
import logging
import os
from itertools import chain, zip_longest
from time import time
from typing import Any, cast, Dict, Iterable, List, NamedTuple, Optional, Tuple, Union
from uuid import uuid4

import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
import torch
from flask import Flask, jsonify, make_response, Response, request
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from waitress import serve


DEFAULT_STREAM_BATCH_SIZE = 0
DEFAULT_MAX_TOKENS = 16
DEFAULT_TEMPERATURE = 1.
DEFAULT_TOP_P = 1.
DEFAULT_NUM_RETURN_SEQUENCES = 1
EOS = '</s>'
DEFAULT_PROMPT = EOS
END_OF_STREAM = '[DONE]'
FINISH_REASON_EOS = 'stop'
FINISH_REASON_LENGTH = 'length'


MaxMemoryDict = Dict[Union[int, str], Union[int, str]]


class Completion(NamedTuple):
    text: str
    finish_reason: Optional[str]
    idx: int


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
    offload_dir: Optional[str]
    load_in_8bit: bool
    models: Dict[str, Tuple[PreTrainedTokenizer, PreTrainedModel]]
    main_device: str

    def __init__(self,
                 offload_dir: Optional[str] = None,
                 preload_model: Optional[str] = None,
                 load_in_8bit: bool = False):
        self.offload_dir = offload_dir
        self.load_in_8bit = load_in_8bit

        self.models = {}
        self.main_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if preload_model is not None:
            self.get_tokenizer_and_model(preload_model)

    def get_tokenizer_and_model(self, model_id: str) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        if model_id not in self.models:
            # Deallocate any existing models
            self.models.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logging.info(f'Loading model: {model_id}')
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            if self.offload_dir is not None:
                offload_state_dict = True
                if not os.path.isdir(self.offload_dir):
                    logging.info(f'offload dir {self.offload_dir} does not exist; creating')
                    try:
                        os.makedirs(self.offload_dir)
                    except Exception as ex:
                        logging.warning(f'Could not create offload dir {self.offload_dir}', exc_info=ex)
            else:
                offload_state_dict = False
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map='balanced_low_0',
                load_in_8bit=self.load_in_8bit,
                torch_dtype=torch.float16,
                offload_folder=self.offload_dir,
                offload_state_dict=offload_state_dict,
            )
            model.eval()
            self.models[model_id] = (tokenizer, model)
        return self.models[model_id]

    def complete(self, text: str, model_id: str,
                 stop_strings: List[str],
                 max_new_tokens: int = DEFAULT_MAX_TOKENS,
                 do_sample: bool = True,
                 top_p: float = DEFAULT_TOP_P,
                 temperature: float = DEFAULT_TEMPERATURE,
                 num_return_sequences: int = DEFAULT_NUM_RETURN_SEQUENCES) -> List[Completion]:
        (tokenizer, model) = self.get_tokenizer_and_model(model_id)

        return [
            Completion(
                text=raw_completion.new_text,
                finish_reason=FINISH_REASON_EOS if raw_completion.truncated else FINISH_REASON_LENGTH,
                idx=i,
            )
            for (i, raw_completion) in enumerate(self._complete(
                text, tokenizer, model, stop_strings=stop_strings, max_new_tokens=max_new_tokens,
                do_sample=do_sample, top_p=top_p, temperature=temperature,
                num_return_sequences=num_return_sequences,
            ))
        ]

    def stream_complete(
            self, text: str, model_id: str,
            stop_strings: List[str],
            max_new_tokens: int = DEFAULT_MAX_TOKENS,
            do_sample: bool = True,
            top_p: float = DEFAULT_TOP_P,
            temperature: float = DEFAULT_TEMPERATURE,
            num_return_sequences: int = DEFAULT_NUM_RETURN_SEQUENCES,
            stream_batch_size: int = DEFAULT_STREAM_BATCH_SIZE) -> Iterable[Completion]:
        streams = [
            self._stream_complete_single(
                text, model_id, stop_strings,
                max_new_tokens=max_new_tokens, do_sample=do_sample, top_p=top_p, temperature=temperature,
                index=i, stream_batch_size=stream_batch_size,
            )
            for i in range(num_return_sequences)
        ]
        return (
            c
            for c in (
                completion
                for completions in zip_longest(*streams, fillvalue=None)
                for completion in completions
            )
            if c is not None
        )

    def _stream_complete_single(
            self, text: str, model_id: str,
            stop_strings: List[str],
            max_new_tokens: int = DEFAULT_MAX_TOKENS,
            do_sample: bool = True,
            top_p: float = DEFAULT_TOP_P,
            temperature: float = DEFAULT_TEMPERATURE,
            index: int = 0,
            stream_batch_size: int = DEFAULT_STREAM_BATCH_SIZE) -> Iterable[Completion]:
        (tokenizer, model) = self.get_tokenizer_and_model(model_id)

        prompt = text
        num_new_tokens = 0
        finish_reason = None
        while finish_reason is None:
            [raw_completion] = self._complete(
                text, tokenizer, model, stop_strings=stop_strings,
                max_new_tokens=min(
                    stream_batch_size if stream_batch_size > 0 else max_new_tokens,
                    max_new_tokens - num_new_tokens
                ),
                do_sample=do_sample, top_p=top_p, temperature=temperature,
                num_return_sequences=1,
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
                truncation_index = len(prompt) + len(truncated_text_after_prompt)
                yield Completion(
                    text=raw_completion.text[-len(raw_completion.new_text):truncation_index],
                    finish_reason=FINISH_REASON_EOS,
                    idx=index,
                )
            else:
                yield Completion(text=raw_completion.new_text, finish_reason=finish_reason, idx=index)

            text = raw_completion.text

    def _complete(self, text: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel,
                  stop_strings: List[str], max_new_tokens: int,
                  do_sample: bool, top_p: float, temperature: float,
                  num_return_sequences: int) -> List[RawCompletion]:
        input_token_ids = tokenizer(text, return_tensors='pt')['input_ids']
        output_token_ids = cast(torch.Tensor, model.generate(
            input_token_ids.to(self.main_device), max_new_tokens=max_new_tokens, do_sample=do_sample, top_p=top_p,
            temperature=temperature, num_return_sequences=num_return_sequences,
        ))
        completions = []
        for completion_num in range(output_token_ids.shape[0]):
            output_text = clean_output_text(tokenizer.decode(output_token_ids[completion_num].tolist()))
            if output_text.startswith(text):
                new_text = output_text[len(text):]
                (new_text, truncated) = truncate_at_stops(new_text, stop_strings)
                output_text = text + new_text
                completions.append(RawCompletion(
                    text=output_text,
                    pretruncation_num_new_tokens=output_token_ids.size(dim=1) - input_token_ids.size(dim=1),
                    new_text=new_text,
                    truncated=truncated,
                ))
            else:
                raise Exception(f'Generated text "{output_text}" does not begin with input text "{text}"')

        return completions


def make_api_completions(response_id: str, created: int, model_id: str,
                         completions: List[Completion]) -> Dict[str, Any]:
    return {
        'id': response_id,
        'object': 'text_completion',
        'created': created,
        'model': model_id,
        'choices': [
            {
                'text': completion.text,
                'index': completion.idx,
                'logprobs': None,
                'finish_reason': completion.finish_reason,
            }
            for completion in completions
        ],
        'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
    }


def make_error_response(status: int, message: str, error_type: str,
                        param: Optional[Any] = None, code: Optional[str] = None) -> Response:
    return make_response((
        {
            'error': {
                'message': message,
                'type': error_type,
                'param': param,
                'code': code,
            },
        },
        status,
    ))


def create_app(offload_dir: Optional[str] = None,
               preload_model: Optional[str] = None,
               load_in_8bit: bool = False) -> Flask:
    lm = LM(
        offload_dir=offload_dir,
        preload_model=preload_model,
        load_in_8bit=load_in_8bit,
    )

    app = Flask(__name__)

    @app.errorhandler(404)
    def invalid_url(error):
        return make_error_response(
            404,
            f'Invalid URL ({request.method} {request.path})',
            'invalid_request_error',
        )

    @app.errorhandler(405)
    def invalid_method(error):
        return make_error_response(
            405,
            (
                f'Not allowed to {request.method} on {request.path} '
                '(HINT: Perhaps you meant to use a different HTTP method?)'
            ),
            'invalid_request_error',
        )

    @app.errorhandler(500)
    def internal_server_error(error):
        return make_error_response(
            500,
            'The server encountered an internal error',
            'internal_server_error',
        )

    @app.route('/v1/completions', methods=['POST'])
    def post_completions():
        if not request.is_json:
            return make_error_response(
                400,
                (
                    'Your request does not have a JSON Content-Type header. '
                    'The API expects "Content-Type: application/json".'
                ),
                'invalid_request_error',
            )

        try:
            request_json = request.get_json()
            if not isinstance(request_json, dict):
                raise Exception('Request body is not a JSON dictionary')
        except Exception:
            return make_error_response(
                400,
                (
                    'We could not parse the JSON body of your request. '
                    '(HINT: This likely means you aren\'t using your HTTP library correctly. '
                    'The API expects a JSON payload, but what was sent was not valid JSON.'
                ),
                'invalid_request_error',
            )

        try:
            max_tokens = int(request_json.get('max_tokens', DEFAULT_MAX_TOKENS))

            model_id = request_json['model']
            if not isinstance(model_id, str) or not model_id:
                raise Exception('model must be a non-empty string')

            prompt = request_json.get('prompt', DEFAULT_PROMPT)
            if not isinstance(prompt, str):
                raise Exception('prompt must be a string')

            _stop = request_json.get('stop')
            if isinstance(_stop, list):
                stops = _stop
            elif isinstance(_stop, str):
                stops = [_stop]
            else:
                stops = []

            num_return_sequences = int(request_json.get('n', DEFAULT_NUM_RETURN_SEQUENCES))

            stream = request_json.get('stream', False)

            greedy_decoding = request_json.get('greedy_decoding', False)

            temperature = float(request_json.get('temperature', DEFAULT_TEMPERATURE))

            top_p = float(request_json.get('top_p', DEFAULT_TOP_P))

            user = request_json.get('user')
            sentry_sdk.set_user({'id': user} if user else None)

            completion_log_text = 'completion' if num_return_sequences == 1 else 'completions'
            if stream:
                completion_log_text = 'streaming ' + completion_log_text
            completion_log_text = 'streaming completion' if stream else 'completion'
            tokens_log_text = 'token' if max_tokens == 1 else 'tokens'
            if num_return_sequences != 1:
                tokens_log_text = tokens_log_text + ' each'
            logging.debug(f'Computing {completion_log_text} of up to {max_tokens} {tokens_log_text} for user {user}')

            stream_batch_size = int(request_json.get('stream_batch_size', DEFAULT_STREAM_BATCH_SIZE))

        except Exception as ex:
            return make_error_response(
                400,
                str(ex),
                'invalid_request_error',
            )

        response_id = generate_response_id()
        created = get_timestamp()
        if stream:
            return Response(
                (f'data: {event_data}\n\n' for event_data in chain(
                    (
                        json.dumps(make_api_completions(response_id, created, model_id, [completion]))
                        for completion in lm.stream_complete(
                            prompt, model_id, stops, max_new_tokens=max_tokens,
                            do_sample=not greedy_decoding, top_p=top_p, temperature=temperature,
                            num_return_sequences=num_return_sequences,
                            stream_batch_size=stream_batch_size,
                        )
                    ),
                    [END_OF_STREAM],
                )),
                mimetype='text/event-stream',
                headers={'X-Accel-Buffering': 'no'},  # tell nginx not to buffer
            )
        else:
            return jsonify(make_api_completions(response_id, created, model_id, lm.complete(
                prompt, model_id, stops, max_new_tokens=max_tokens,
                do_sample=not greedy_decoding, top_p=top_p, temperature=temperature,
                num_return_sequences=num_return_sequences,
            )))

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
    parser.add_argument('-d', '--offload-dir',
                        help='Directory where model will be offloaded if available memory is exceeded')
    parser.add_argument('-m', '--preload-model',
                        help='Huggingface repository of model to preload (example: facebook/opt-2.7b)')
    parser.add_argument('-8', '--load-in-8bit', action='store_true',
                        help='Load model in 8 bit using the bits-and-bytes algorithm')
    parser.add_argument('-l', '--log-level',
                        choices=('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'), default='INFO',
                        help='Logging verbosity level threshold (to stderr)')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                        level=args.log_level)

    sentry_sdk.init(
        integrations=[
            FlaskIntegration(),
        ],
        traces_sample_rate=1.0,  # a rate < 1.0 is recommended for production, yolo
    )
    sentry_sdk.set_tag('component', 'backend-hf')

    if args.preload_model:
        preload_model = args.preload_model
    elif os.environ.get('SANDLE_SINGLE_MODEL'):
        preload_model = os.environ['SANDLE_SINGLE_MODEL']
    else:
        preload_model = None

    app = create_app(
        offload_dir=args.offload_dir,
        preload_model=preload_model,
        load_in_8bit=args.load_in_8bit,
    )

    serve(app, host=args.host, port=args.port, threads=1)


if __name__ == '__main__':
    main()

import json
import gc
import logging
from pathlib import Path
from time import time
from typing import Any, cast, Dict, List, NamedTuple, Optional, Tuple, Union
from uuid import uuid4

import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
import torch
from flask import Flask, jsonify, make_response, Response, request
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation_stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from waitress import serve
import click


DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 8000
DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_MAX_TOKENS = 16
DEFAULT_TEMPERATURE = 1.
DEFAULT_TOP_P = 1.
DEFAULT_NUM_RETURN_SEQUENCES = 1
EOS = '</s>'
DEFAULT_PROMPT = EOS
FINISH_REASON_EOS = 'stop'
FINISH_REASON_LENGTH = 'length'

with open('models.json') as f:
    MODELS = json.load(f)


MaxMemoryDict = Dict[Union[int, str], Union[int, str]]


def tokenizer_decode(token_ids: List[int], tokenizer: PreTrainedTokenizer,
                     prefix_to_skip: str = EOS) -> str:
    text = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
    if prefix_to_skip and text.startswith(prefix_to_skip):
        return text[len(prefix_to_skip):]
    else:
        return text


class SubstringMatchStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_string: str, prompt: str, tokenizer: PreTrainedTokenizer):
        self.stop_string = stop_string
        self.prompt = prompt
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for input_num in range(input_ids.shape[0]):
            text = tokenizer_decode(input_ids[input_num].tolist(), self.tokenizer)
            if not text.startswith(self.prompt):
                raise Exception(f'Generated text "{text}" does not begin with prompt "{self.prompt}"')
            if self.stop_string not in text[len(self.prompt):]:
                return False

        return True


class Completion(NamedTuple):
    text: str
    finish_reason: Optional[str]
    idx: int


class RawCompletion(NamedTuple):
    text: str
    pretruncation_num_new_tokens: int
    new_text: str
    truncated: bool


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
    offload_dir: Optional[Path]
    load_in_8bit: bool
    models: Dict[str, Tuple[PreTrainedTokenizer, PreTrainedModel]]
    main_device: str

    def __init__(self,
                 offload_dir: Optional[Path] = None,
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
                if not self.offload_dir.is_dir():
                    logging.info(f'offload dir {self.offload_dir} does not exist; creating')
                    try:
                        self.offload_dir.mkdir(parents=True, exist_ok=True)
                    except Exception as ex:
                        logging.warning(f'Could not create offload dir {self.offload_dir}', exc_info=ex)
            else:
                offload_state_dict = False
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map='balanced_low_0' if torch.cuda.device_count() > 1 else 'auto',
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

    def _complete(self, text: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel,
                  stop_strings: List[str], max_new_tokens: int,
                  do_sample: bool, top_p: float, temperature: float,
                  num_return_sequences: int) -> List[RawCompletion]:
        input_token_ids = tokenizer(text, return_tensors='pt')['input_ids']
        output_token_ids = cast(torch.Tensor, model.generate(
            input_token_ids.to(self.main_device), max_new_tokens=max_new_tokens, do_sample=do_sample, top_p=top_p,
            temperature=temperature, num_return_sequences=num_return_sequences,
            stopping_criteria=StoppingCriteriaList(
                SubstringMatchStoppingCriteria(stop_string, text, tokenizer)
                for stop_string in stop_strings
            ),
        ))
        completions = []
        for completion_num in range(output_token_ids.shape[0]):
            output_text = tokenizer_decode(output_token_ids[completion_num].tolist(), tokenizer)
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
                raise Exception(f'Generated text "{output_text}" does not begin with prompt "{text}"')

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


def create_app(offload_dir: Optional[Path] = None,
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

    @app.route('/v1/models')
    def get_models():
        return jsonify({
            'data': MODELS,
            'object': 'list'
        })

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
            if stream:
                raise NotImplementedError('Streaming is not implemented')

            greedy_decoding = request_json.get('greedy_decoding', False)

            temperature = float(request_json.get('temperature', DEFAULT_TEMPERATURE))

            top_p = float(request_json.get('top_p', DEFAULT_TOP_P))

            user = request_json.get('user')
            sentry_sdk.set_user({'id': user} if user else None)

            completion_log_text = 'completion' if num_return_sequences == 1 else 'completions'
            tokens_log_text = 'token' if max_tokens == 1 else 'tokens'
            if num_return_sequences != 1:
                tokens_log_text = tokens_log_text + ' each'
            logging.debug(f'Computing {completion_log_text} of up to {max_tokens} {tokens_log_text} for user {user}')

        except Exception as ex:
            return make_error_response(
                400,
                str(ex),
                'invalid_request_error',
            )

        response_id = generate_response_id()
        created = get_timestamp()
        return jsonify(make_api_completions(response_id, created, model_id, lm.complete(
            prompt, model_id, stops, max_new_tokens=max_tokens,
            do_sample=not greedy_decoding, top_p=top_p, temperature=temperature,
            num_return_sequences=num_return_sequences,
        )))

    return app


@click.command()
@click.option('--host', type=str, default=DEFAULT_HOST, help='Hostname or IP to serve on')
@click.option('-p', '--port', type=int, default=DEFAULT_PORT, help='Port to serve on')
@click.option('-d', '--offload-dir', type=click.Path(file_okay=False, path_type=Path),
              help='Directory where model will be offloaded if available memory is exceeded')
@click.option('-8', '--load-in-8bit', is_flag=True,
              help='Load model in 8 bit using the bits-and-bytes algorithm')
@click.option('-m', '--single-model', '--preload-model', type=str,
              help='Huggingface repository of model to preload (example: facebook/opt-2.7b)')
@click.option('-l', '--log-level', type=click.Choice(('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG')),
              default=DEFAULT_LOG_LEVEL, help='Logging verbosity level threshold (to stderr)')
def main(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    log_level: str = DEFAULT_LOG_LEVEL,
    offload_dir: Optional[Path] = None,
    load_in_8bit: bool = False,
    single_model: Optional[str] = None,
):
    """Run a simplified, single-threaded clone of OpenAI's /v1/completions endpoint"""

    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                        level=log_level)

    sentry_sdk.init(
        integrations=[
            FlaskIntegration(),
        ],
        traces_sample_rate=1.0,  # a rate < 1.0 is recommended for production, yolo
    )
    sentry_sdk.set_tag('component', 'backend-hf')

    app = create_app(
        offload_dir=offload_dir,
        preload_model=single_model,
        load_in_8bit=load_in_8bit,
    )

    serve(app, host=host, port=port, threads=1)


if __name__ == '__main__':
    main(auto_envvar_prefix='SANDLE')

import json
import logging
import os
from pathlib import Path
from time import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from uuid import uuid4

import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from flask import Flask, jsonify, make_response, Response, request
from waitress import serve
import mii
import click


DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 8000
DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_MAX_TOKENS = 16
DEFAULT_TEMPERATURE = 1.
DEFAULT_TOP_P = 1.
DEFAULT_NUM_RETURN_SEQUENCES = 1
DEFAULT_PROMPT = 'Hello world!'
END_OF_STREAM = '[DONE]'
FINISH_REASON_EOS = 'stop'
FINISH_REASON_LENGTH = 'length'

with open('models.json') as f:
    MODELS = json.load(f)


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


class Completion(NamedTuple):
    text: str
    finish_reason: Optional[str]
    idx: int

    @classmethod
    def from_truncation(cls, truncation: Tuple[str, bool], idx: int = 0):
        (text, truncated) = truncation
        return cls(text=text, finish_reason=FINISH_REASON_EOS if truncated else FINISH_REASON_LENGTH, idx=idx)


def make_api_completions(
        response_id: str, created: int, model_id: str, completions: List[Completion]
) -> Dict[str, Any]:
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
                'finish_reason': completion.finish_reason
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


def create_app(model_id: str, deployment_name: str) -> Flask:
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
            'data': [model_data for model_data in MODELS if model_data['id'] == model_id],
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

            requested_model_id = request_json['model']
            if requested_model_id != model_id:
                raise Exception(f'model must be {model_id}')

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

            if stops:
                logging.warning('Stop sequences are implemented naively')

            num_return_sequences = int(request_json.get('n', DEFAULT_NUM_RETURN_SEQUENCES))

            stream = request_json.get('stream', False)
            if stream and num_return_sequences != 1:
                raise NotImplementedError('Streaming with more than one return sequence is not implemented')

            greedy_decoding = request_json.get('greedy_decoding', False)

            temperature = float(request_json.get('temperature', DEFAULT_TEMPERATURE))

            top_p = float(request_json.get('top_p', DEFAULT_TOP_P))

            user = request_json.get('user')
            sentry_sdk.set_user({'id': user} if user else None)

            completion_log_text = 'completion' if num_return_sequences == 1 else 'completions'
            if stream:
                completion_log_text = 'streaming ' + completion_log_text
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
        prompts = [prompt] * num_return_sequences
        generator = mii.mii_query_handle(deployment_name)
        result = generator.query(
            {'query': prompts},
            max_new_tokens=max_tokens, do_sample=not greedy_decoding, top_p=top_p,
            temperature=temperature, num_return_sequences=num_return_sequences,
        )
        api_completions = make_api_completions(
            response_id,
            created,
            model_id,
            [
                Completion.from_truncation(truncate_at_stops(raw_completion_text[len(prompt):], stop_strings=stops), i)
                for (i, raw_completion_text) in enumerate(result.response)
            ],
        )
        if stream:
            return Response(
                (f'data: {event_data}\n\n' for event_data in (json.dumps(api_completions), END_OF_STREAM)),
                mimetype='text/event-stream',
                headers={'X-Accel-Buffering': 'no'},  # tell nginx not to buffer
            )
        else:
            return jsonify(api_completions)

    return app


def serve_app(model_id: str, deployment_name: str, **waitress_kwargs):
    app = create_app(model_id, deployment_name)
    serve(app, **waitress_kwargs)


@click.command()
@click.argument('model_id', type=click.Choice(tuple(model_data['id'] for model_data in MODELS)))
@click.option('--host', type=str, default=DEFAULT_HOST, help='Hostname or IP to serve on')
@click.option('-p', '--port', type=int, default=DEFAULT_PORT, help='Port to serve on')
@click.option('--model-path', type=click.Path(file_okay=False, path_type=Path),
              help='Path where downloaded checkpoints will be stored')
@click.option('-l', '--log-level', type=click.Choice(('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG')),
              default=DEFAULT_LOG_LEVEL, help='Logging verbosity level threshold (to stderr)')
def main(
    model_id: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    model_path: Optional[os.PathLike] = None,
    log_level: str = DEFAULT_LOG_LEVEL,
):
    """
    Run a simplified, single-threaded clone of OpenAI's /v1/completions endpoint on the specified model
    in DeepSpeed.
    """
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(process)d] [%(name)s] %(message)s',
                        level=log_level)

    sentry_sdk.init(
        integrations=[
            FlaskIntegration(),
        ],
        traces_sample_rate=0.1,
    )
    sentry_sdk.set_tag('component', 'backend-deepspeed')

    deployment_name = f'{model_id}_deployment'
    mii_config = {"tensor_parallel": 1, "dtype": "fp16"}

    try:
        mii.deploy(
            task='text-generation',
            model=model_id,
            model_path=model_path,
            deployment_name=deployment_name,
            mii_config=mii_config,
        )
        serve_app(model_id, deployment_name, host=host, port=port, threads=1)

    finally:
        mii.terminate(deployment_name)


if __name__ == '__main__':
    main(auto_envvar_prefix='SANDLE')

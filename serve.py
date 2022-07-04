import json
import logging
from base64 import b64encode
from functools import wraps
from random import randbytes
from time import time
from typing import Any, Dict
from uuid import uuid4

from flask import Flask, jsonify, make_response, Response, request
from waitress import serve


ModelData = Dict[str, Any]


DEFAULT_MAX_TOKENS = 16
DEFAULT_NUM_COMPLETIONS = 1
DEFAULT_PROMPT = '<|endoftext|>'
END_OF_STREAM = '[DONE]'
FINISH_REASON_EOS = 'stop'
FINISH_REASON_LENGTH = 'length'

MODELS = [
    {
        'id': 'text-davinci-002',
        'object': 'model',
        'owned_by': 'organization-owner',
        'permission': [],
    },
]


def _get_model_data(model_id: str) -> ModelData:
    [model_data] = [md for md in MODELS if md['id'] == model_id]
    return model_data


def generate_auth_token(num_bytes: int = 8) -> str:
    return b64encode(randbytes(num_bytes)).decode('ascii')


def generate_response_id() -> str:
    return str(uuid4())


def get_timestamp() -> int:
    return int(time())


def create_app(auth_token: str) -> Flask:
    app = Flask(__name__)

    def authorization_required(f):
        @wraps(f)
        def decorator(*args, **kwargs):
            try:
                (request_scheme, request_token) = request.headers['Authorization'].split(' ')
                if request_scheme != 'Bearer':
                    raise Exception(f'Expected Bearer authorization scheme but got {request_scheme}')
            except Exception:
                return make_response(jsonify({'message': 'Valid authorization bearer token is missing'}), 401)

            if request_token != auth_token:
                return make_response(jsonify({'message': 'Invalid authorization bearer token'}), 401)

            return f(*args, **kwargs)

        return decorator

    @app.route('/v1/models')
    @authorization_required
    def get_models():
        return jsonify({
            'data': MODELS,
            'object': 'list'
        })

    @app.route('/v1/model/<model>')
    @authorization_required
    def get_model(model):
        return jsonify(_get_model_data(model))

    @app.route('/v1/completions', methods=['POST'])
    @authorization_required
    def post_completions():
        max_tokens = int(request.json.get('max_tokens', DEFAULT_MAX_TOKENS))

        model_data = _get_model_data(request.json['model'])
        model_id = model_data['id']

        num_completions = int(request.json.get('n', DEFAULT_NUM_COMPLETIONS))

        _prompt = request.json.get('prompt', DEFAULT_PROMPT)
        prompts = _prompt if isinstance(_prompt, list) else [_prompt]

        _stop = request.json.get('stop', [])
        stops = _stop if isinstance(_stop, list) else [_stop]

        stream = request.json.get('stream', False)

        user = request.json.get('user')

        logging.debug(f'Computing {num_completions} completion{"" if num_completions == 1 else "1"} '
                      f'with up to {max_tokens} token{"" if max_tokens == 1 else "1"} each '
                      f'for user {user}')

        response_id = generate_response_id()
        created = get_timestamp()
        if stream:
            return Response(
                (f'data: {event_data}\n\n' for event_data in [
                    json.dumps({
                        'id': response_id,
                        'object': 'text_completion',
                        'created': created,
                        'choices': [{'text': '\n', 'index': 0, 'logprobs': None, 'finish_reason': None}],
                        'model': model_id
                    }),
                    json.dumps({
                        'id': response_id,
                        'object': 'text_completion',
                        'created': created,
                        'choices': [{'text': '\n', 'index': 0, 'logprobs': None, 'finish_reason': None}],
                        'model': model_id
                    }),
                    json.dumps({
                        'id': response_id,
                        'object': 'text_completion',
                        'created': created,
                        'choices': [{'text': 'This', 'index': 0, 'logprobs': None, 'finish_reason': None}],
                        'model': model_id
                    }),
                    json.dumps({
                        'id': response_id,
                        'object': 'text_completion',
                        'created': created,
                        'choices': [{'text': ' is', 'index': 0, 'logprobs': None, 'finish_reason': None}],
                        'model': model_id
                    }),
                    json.dumps({
                        'id': response_id,
                        'object': 'text_completion',
                        'created': created,
                        'choices': [{'text': ' a', 'index': 0, 'logprobs': None, 'finish_reason': None}],
                        'model': model_id
                    }),
                    json.dumps({
                        'id': response_id,
                        'object': 'text_completion',
                        'created': created,
                        'choices': [{'text': ' test', 'index': 0, 'logprobs': None, 'finish_reason': None}],
                        'model': model_id
                    }),
                    json.dumps({
                        'id': response_id,
                        'object': 'text_completion',
                        'created': created,
                        'choices': [{'text': '.', 'index': 0, 'logprobs': None, 'finish_reason': None}],
                        'model': model_id
                    }),
                    json.dumps({
                        'id': response_id,
                        'object': 'text_completion',
                        'created': created,
                        'choices': [{'text': '', 'index': 0, 'logprobs': None, 'finish_reason': FINISH_REASON_EOS}],
                        'model': model_id
                    }),
                    END_OF_STREAM,
                ]),
                mimetype='text/event-stream',
            )
        else:
            return jsonify({
                'id': response_id,
                'object': 'text_completion',
                'created': created,
                'model': model_id,
                'choices': [
                    {
                        'text': '\n\nThis is a test',
                        'index': 0,
                        'logprobs': None,
                        'finish_reason': FINISH_REASON_EOS,
                    }
                ],
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            })

    return app


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Run a clone of OpenAI\'s API Service in your Local Environment',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--host', default='0.0.0.0',
                        help='Hostname or IP to serve on')
    parser.add_argument('-p', '--port', type=int, default=8000,
                        help='Port to serve on')
    parser.add_argument('--auth-token',
                        help='Base-64--encoded authorization token (API key); '
                             'if not specified, one will be generated on startup.')
    parser.add_argument('-l', '--log-level',
                        choices=('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'), default='INFO',
                        help='Logging verbosity level threshold (to stderr)')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                        level=args.log_level)

    auth_token = args.auth_token if args.auth_token is not None else generate_auth_token()
    logging.info(f'Authorization token: {auth_token}')

    app = create_app(auth_token=auth_token)
    serve(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

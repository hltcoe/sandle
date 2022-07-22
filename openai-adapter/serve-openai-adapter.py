import logging
import secrets
from base64 import b64encode
from functools import wraps
from typing import Any, cast, Dict, Optional

import requests
from flask import Flask, jsonify, make_response, Response, request
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth, HTTPTokenAuth, MultiAuth
from waitress import serve


ModelData = Dict[str, Any]


END_OF_STREAM = '[DONE]'

MODELS = [
    {
        'id': f'facebook/opt-{size}',
        'object': 'model',
        'owned_by': 'facebook',
        'permission': [],
    } for size in ('125m', '350m', '1.3b', '2.7b', '6.7b', '13b', '30b', '66b')
]


def get_models_dict() -> Dict[str, ModelData]:
    return cast(Dict[str, ModelData], dict((m['id'], m) for m in MODELS))


def get_model_data(model_id: str) -> Optional[ModelData]:
    return get_models_dict().get(model_id)


def generate_auth_token(password_length: int = 16) -> str:
    return b64encode(secrets.token_bytes(password_length)).decode('ascii')


# flask-httpauth is super helpful and sets WWW-Authenticate to prompt the
# user for a password, but we want to handle authentication ourselves, so we
# in and remove it
def strip_www_authenticate_header(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        res = f(*args, **kwargs)
        if 'WWW-Authenticate' in res.headers:
            del res.headers['WWW-Authenticate']
        return res
    return decorated


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


def create_app(accepted_auth_token: str, backend_completions_url: str) -> Flask:
    app = Flask(__name__)

    # We use CORS just to facilitate development and debugging
    CORS(app)

    basic_auth = HTTPBasicAuth()
    token_auth = HTTPTokenAuth(scheme='Bearer')
    multi_auth = MultiAuth(basic_auth, token_auth)

    @basic_auth.verify_password
    def verify_password(username, password):
        return password == accepted_auth_token

    @token_auth.verify_token
    def verify_token(token):
        return token == accepted_auth_token

    def auth_error(status):
        return make_error_response(
            status,
            'Invalid credentials (API key or password) or no credentials provided',
            'invalid_request_error',
        )

    basic_auth.error_handler(auth_error)
    token_auth.error_handler(auth_error)

    @app.route('/v1/models')
    @strip_www_authenticate_header
    @multi_auth.login_required
    def get_models():
        return jsonify({
            'data': MODELS,
            'object': 'list'
        })

    @app.route('/v1/models/<path:model>')
    @strip_www_authenticate_header
    @multi_auth.login_required
    def get_model(model):
        model_data = get_model_data(model)
        if model_data is None:
            return make_error_response(
                401,
                'That model does not exist',
                'invalid_request_error',
            )
        else:
            return jsonify(model_data)

    @app.route('/v1/completions', methods=['POST'])
    @strip_www_authenticate_header
    @multi_auth.login_required
    def post_completions():
        model_data = get_model_data(request.json.get('model'))
        if model_data is None:
            return make_error_response(
                401,
                'That model does not exist',
                'invalid_request_error',
            )
        else:
            stream = request.json.get('stream', False)
            r = requests.post(backend_completions_url, json=request.json, stream=stream)
            headers = {}
            for passthru_header in ('X-Accel-Buffering', 'Content-Type'):
                if passthru_header in r.headers:
                    headers[passthru_header] = r.headers[passthru_header]
            return Response(r.iter_content(chunk_size=None), status=r.status_code, headers=headers)

    return app


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Run a clone of OpenAI\'s API Service in your Local Environment',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('backend_completions_url',
                        help='URL of backend text completion endpoint')
    parser.add_argument('--host', default='0.0.0.0',
                        help='Hostname or IP to serve on')
    parser.add_argument('-p', '--port', type=int, default=8000,
                        help='Port to serve on')
    parser.add_argument('-t', '--auth-token',
                        help='Base-64--encoded authorization token (API key) to accept; '
                             'if not specified, one will be generated on startup.')
    parser.add_argument('-m', '--allow-model', action='append', metavar='model',
                        choices=tuple(m['id'] for m in MODELS),
                        help='Allow only the specified model(s) to be used. '
                             'Can be provided multiple times to allow multiple models. '
                             'By default, all supported models are allowed.')
    parser.add_argument('-l', '--log-level',
                        choices=('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'), default='INFO',
                        help='Logging verbosity level threshold (to stderr)')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                        level=args.log_level)

    if args.auth_token is not None:
        auth_token = args.auth_token
        logging.info(f'Authorization token (API key): {auth_token}')
    else:
        auth_token = generate_auth_token()
        logging.info(f'Generated authorization token (API key): {auth_token}')

    if args.allow_model:
        allowed_models_set = set(args.allow_model)

        logging.info(f'Allowing only the specified model(s) to be used: {allowed_models_set}')
        models_to_remove = [m for m in MODELS if m['id'] not in allowed_models_set]
        for model_to_remove in models_to_remove:
            MODELS.remove(model_to_remove)

    app = create_app(accepted_auth_token=auth_token, backend_completions_url=args.backend_completions_url)
    serve(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

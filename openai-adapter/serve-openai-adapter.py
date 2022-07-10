import logging
import secrets
from base64 import b64encode
from functools import wraps
from typing import Any, Dict

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
    } for size in ('125m', '350m', '1.3b', '2.7b')
]


def _get_model_data(model_id: str) -> ModelData:
    [model_data] = [md for md in MODELS if md['id'] == model_id]
    return model_data


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
        return make_response((
            {
                'error': {
                    'message': 'Invalid credentials (API key or password) or no credentials provided.',
                    'type': 'invalid_request_error',
                    'param': None,
                    'code': None,
                },
            },
            status,
        ))

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

    @app.route('/v1/model/<model>')
    @strip_www_authenticate_header
    @multi_auth.login_required
    def get_model(model):
        return jsonify(_get_model_data(model))

    @app.route('/v1/completions', methods=['POST'])
    @strip_www_authenticate_header
    @multi_auth.login_required
    def post_completions():
        stream = request.json.get('stream', False)
        with requests.post(backend_completions_url, json=request.json, stream=stream) as r:
            return Response(r.iter_content(chunk_size=None),
                            status=r.status_code,
                            content_type=r.headers['content-type'])

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
    parser.add_argument('--auth-token',
                        help='Base-64--encoded authorization token (API key) to accept; '
                             'if not specified, one will be generated on startup.')
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

    app = create_app(accepted_auth_token=auth_token, backend_completions_url=args.backend_completions_url)
    serve(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

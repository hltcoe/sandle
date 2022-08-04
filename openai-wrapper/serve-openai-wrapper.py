import logging
import os
import secrets
from base64 import b64encode
from functools import wraps
from typing import Any, cast, Dict, List, Optional

import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from flask import Flask, jsonify, make_response, Response, request
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth, HTTPTokenAuth, MultiAuth
import requests
from requests.exceptions import ConnectionError, Timeout
from waitress import serve


ModelData = Dict[str, Any]


END_OF_STREAM = '[DONE]'

MODELS = [
    {
        'id': model_id,
        'object': 'model',
        'owned_by': '',
        'permission': [],
        'family': model_family,
        'size': model_size,
    } for (model_id, model_family, model_size) in (
        ('facebook/opt-125m', 'OPT', '125m'),
        ('facebook/opt-350m', 'OPT', '350m'),
        ('facebook/opt-1.3b', 'OPT', '1.3b'),
        ('facebook/opt-2.7b', 'OPT', '2.7b'),
        ('facebook/opt-6.7b', 'OPT', '6.7b'),
        ('facebook/opt-13b', 'OPT', '13b'),
        ('facebook/opt-30b', 'OPT', '30b'),
        ('facebook/opt-66b', 'OPT', '66b'),
        ('bigscience/bloom-350m', 'Bloom', '350m'),
        ('bigscience/bloom-760m', 'Bloom', '760m'),
        ('bigscience/bloom-1b3', 'Bloom', '1.3b'),
        ('bigscience/bloom-2b5', 'Bloom', '2.5b'),
        ('bigscience/bloom-6b3', 'Bloom', '6.3b'),
        ('bigscience/bloom', 'Bloom', '176b'),
    )
]


def get_models_dict() -> Dict[str, ModelData]:
    return cast(Dict[str, ModelData], dict((m['id'], m) for m in MODELS))


def get_model_data(model_id: str) -> Optional[ModelData]:
    return get_models_dict().get(model_id)


def generate_auth_token(password_length: int = 16) -> str:
    return b64encode(secrets.token_bytes(password_length)).decode('ascii')


# flask-httpauth is super helpful and sets WWW-Authenticate to prompt the
# user for a password, but we want to handle authentication ourselves, so we
# step in and remove it
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


def create_app(accepted_auth_tokens: List[str], backend_completions_url: str,
               auth_token_is_user: bool = False) -> Flask:
    app = Flask(__name__)

    # We use CORS just to facilitate development and debugging
    CORS(app)

    basic_auth = HTTPBasicAuth()
    token_auth = HTTPTokenAuth(scheme='Bearer')
    multi_auth = MultiAuth(basic_auth, token_auth)

    def _verify_token(token):
        if token in accepted_auth_tokens:
            return token if auth_token_is_user else True
        else:
            return False

    @basic_auth.verify_password
    def verify_password(username, password):
        return _verify_token(password)

    @token_auth.verify_token
    def verify_token(token):
        return _verify_token(token)

    def auth_error(status):
        return make_error_response(
            status,
            'Invalid credentials (API key or password) or no credentials provided',
            'invalid_request_error',
        )

    basic_auth.error_handler(auth_error)
    token_auth.error_handler(auth_error)

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
            f'Not allowed to {request.method} on {request.path} '
            '(HINT: Perhaps you meant to use a different HTTP method?)',
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
        try:
            model_data = get_model_data(model)
        except Exception:
            model_data = None

        if model_data is None:
            return make_error_response(
                404,
                'That model does not exist',
                'invalid_request_error',
            )
        else:
            return jsonify(model_data)

    @app.route('/v1/completions', methods=['POST'])
    @strip_www_authenticate_header
    @multi_auth.login_required
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

        model = request_json.get('model')
        try:
            model_data = get_model_data(model)
        except Exception:
            model_data = None

        if model_data is not None:
            stream = request_json.get('stream', False)
            if auth_token_is_user:
                user = multi_auth.current_user()
                sentry_sdk.set_user({'id': user} if user else None)
                request_json['user'] = user
            try:
                r = requests.post(backend_completions_url, json=request_json, stream=stream)
            except ConnectionError:
                message = 'Error connecting to backend service.'
                logging.exception(message)
                return make_error_response(502, message, 'internal_server_error')
            except Timeout:
                message = 'Timeout connecting to backend service.'
                logging.exception(message)
                return make_error_response(504, message, 'internal_server_error')
            headers = {}
            for passthru_header in ('X-Accel-Buffering', 'Content-Type'):
                if passthru_header in r.headers:
                    headers[passthru_header] = r.headers[passthru_header]
            return Response(r.iter_content(chunk_size=None), status=r.status_code, headers=headers)
        else:
            return make_error_response(
                404,
                'That model does not exist',
                'invalid_request_error',
            )

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
    parser.add_argument('-t', '--auth-token', action='append',
                        help='Base-64--encoded authorization token (API key) to accept; '
                             'can be specified more than once to accept multiple tokens. '
                             'If none are specified, one will be generated on startup.')
    parser.add_argument('-f', '--auth-token-file',
                        help='File containing base-64--encoded authorization tokens (API keys) to accept, '
                             'one per line.  Blank lines are ignored.  '
                             'Auth token file is ignored if --auth-token is specified.  '
                             'If no tokens are specified, one will be generated on startup.')
    parser.add_argument('-u', '--auth-token-is-user', action='store_true',
                        help='If true, use the authorization token as the API user. '
                             'This behavior may result in auth tokens showing up in logs.')
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

    sentry_sdk.init(
        integrations=[
            FlaskIntegration(),
        ],
        traces_sample_rate=1.0,  # a rate < 1.0 is recommended for production, yolo
    )
    sentry_sdk.set_tag('component', 'openai-wrapper')

    if args.auth_token:
        auth_tokens = args.auth_token
        logging.info('Authorization tokens (API keys) read from command-line argument')
    elif args.auth_token_file:
        with open(args.auth_token_file) as f:
            auth_tokens = [line.strip() for line in f if line]
        logging.info('Authorization tokens (API keys) read from file')
    elif os.environ.get('SANDLE_AUTH_TOKEN'):
        auth_tokens = [os.environ['SANDLE_AUTH_TOKEN']]
        logging.info('Authorization token (API key) read from environment variable')
    else:
        auth_tokens = [generate_auth_token()]
        logging.info(f'Generated authorization token (API key): {auth_tokens[0]}')

    if args.allow_model:
        allowed_models_set = set(args.allow_model)

        logging.info(f'Allowing only the specified model(s) to be used: {allowed_models_set}')
        models_to_remove = [m for m in MODELS if m['id'] not in allowed_models_set]
        for model_to_remove in models_to_remove:
            MODELS.remove(model_to_remove)

    app = create_app(
        accepted_auth_tokens=auth_tokens,
        backend_completions_url=args.backend_completions_url,
        auth_token_is_user=args.auth_token_is_user,
    )
    serve(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

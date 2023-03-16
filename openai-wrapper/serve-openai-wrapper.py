import logging
import os
import secrets
from base64 import b64encode
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Dict, Collection, List, Literal, Optional, Tuple

import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from flask import Flask, jsonify, make_response, Response, request
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth, HTTPTokenAuth, MultiAuth
from huggingface_hub import HfApi
from pydantic import BaseModel
import requests
from requests.exceptions import ConnectionError, Timeout
from waitress import serve
import click


BackendUrl = str


class ModelData(BaseModel):
    id: str
    object: Literal['model']
    owned_by: str
    permission: List


DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 8000
DEFAULT_LOG_LEVEL = 'INFO'

END_OF_STREAM = '[DONE]'


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


def create_app(accepted_auth_tokens: Collection[str],
               backend_hf: Optional[BackendUrl] = None,
               backend_llama: Optional[BackendUrl] = None,
               backend_stub: Optional[BackendUrl] = None,
               auth_token_is_user: bool = False,
               allowed_model_ids: Optional[Collection[str]] = None) -> Flask:
    @lru_cache
    def get_explicit_model_data_and_backends() -> Dict[str, Tuple[ModelData, BackendUrl]]:
        explicit_backend_urls = [
            backend_url
            for backend_url in (backend_llama, backend_stub)
            if backend_url is not None
        ]
        explicit_model_data_and_backends: Dict[str, Tuple[ModelData, BackendUrl]] = {}
        for backend_url in explicit_backend_urls:
            backend_models_url = backend_url + '/v1/models'
            try:
                r = requests.get(backend_models_url)
                if r.ok:
                    for model_data in r.json().get('data', []):
                        explicit_model_data_and_backends[model_data['id']] = (
                            ModelData.parse_obj(model_data),
                            backend_url
                        )
                else:
                    logging.error(f'HTTP {r.status_code} ({r.reason}) from backend at {backend_models_url}')
            except ConnectionError:
                logging.exception(f'Error connecting to backend at {backend_models_url}')
            except Timeout:
                logging.exception(f'Timeout connecting to backend at {backend_models_url}')
        return explicit_model_data_and_backends

    @lru_cache
    def get_implicit_model_data_and_backend(model_id: str) -> Optional[Tuple[ModelData, BackendUrl]]:
        if backend_hf is not None:
            api = HfApi()
            model_infos = [
                model_info
                for model_info in api.list_models(search=model_id)
                if model_info.modelId == model_id
            ]
            if model_infos:
                [model_info] = model_infos
                return (
                    ModelData(
                        id=model_info.modelId,
                        object='model',
                        owned_by=model_info.author if model_info.author else '',
                        permission=[],
                    ),
                    backend_hf,
                )

        return None

    def get_model_data_and_backend(model_id: str) -> Optional[Tuple[ModelData, BackendUrl]]:
        if allowed_model_ids is not None and model_id not in allowed_model_ids:
            return None

        explicit_model_data_and_backend = get_explicit_model_data_and_backends().get(model_id)
        if explicit_model_data_and_backend is not None:
            return explicit_model_data_and_backend
        else:
            return get_implicit_model_data_and_backend(model_id)

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
        model_data_and_backends = (
            get_explicit_model_data_and_backends().values()
            if allowed_model_ids is None
            else [
                model_data_and_backend
                for model_data_and_backend
                in (get_model_data_and_backend(model_id) for model_id in allowed_model_ids)
                if model_data_and_backend is not None
            ]
        )
        return jsonify({
            'data': [
                model_data.dict()
                for (model_data, _)
                in model_data_and_backends
            ],
            'object': 'list',
        })

    @app.route('/v1/models/<path:model_id>')
    @strip_www_authenticate_header
    @multi_auth.login_required
    def get_model(model_id):
        try:
            model_data_and_backend = get_model_data_and_backend(model_id)
            model_data = model_data_and_backend[0] if model_data_and_backend is not None else None
        except Exception as ex:
            logging.debug(f'Error getting data for model {model_id}', exc_info=ex)
            model_data = None

        if model_data is not None:
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

        model_id = request_json.get('model')
        try:
            model_data_and_backend = get_model_data_and_backend(model_id)
        except Exception as ex:
            logging.debug(f'Error getting data for model {model_id}', exc_info=ex)
            model_data_and_backend = None

        if model_data_and_backend is not None:
            (model_data, backend_url) = model_data_and_backend
            stream = request_json.get('stream', False)
            if auth_token_is_user:
                user = multi_auth.current_user()
                sentry_sdk.set_user({'id': user} if user else None)
                request_json['user'] = user
            try:
                r = requests.post(backend_url + '/v1/completions', json=request_json, stream=stream)
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


@click.command()
@click.option('--backend-hf', type=BackendUrl, help='HuggingFace backend URL')
@click.option('--backend-llama', type=BackendUrl, help='LLaMA backend URL')
@click.option('--backend-stub', type=BackendUrl, help='Stub backend URL')
@click.option('--host', type=str, default=DEFAULT_HOST, help='Hostname or IP to serve on')
@click.option('-p', '--port', type=int, default=DEFAULT_PORT, help='Port to serve on')
@click.option('-t', '--auth-token', type=str, multiple=True,
              help='Base-64--encoded authorization token (API key) to accept; '
                   'can be specified more than once to accept multiple tokens. '
                   'If none are specified, one will be generated on startup.')
@click.option('-f', '--auth-token-file', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help='File containing base-64--encoded authorization tokens (API keys) to accept, '
                   'one per line.  Blank lines are ignored. '
                   'Auth token file is ignored if --auth-token is specified. '
                   'If no tokens are specified, one will be generated on startup.')
@click.option('-u', '--auth-token-is-user', is_flag=True,
              help='If true, use the authorization token as a user identifier on the backend, '
                   'allowing activity to be tracked on a per-user basis. '
                   'If true, auth tokens will show up in logs.')
@click.option('-m', '--single-model', '--allow-model', type=str, multiple=True,
              help='Allow only the specified model(s) to be used. '
                   'Can be provided multiple times to allow multiple models. '
                   'By default, all available models are allowed.')
@click.option('-l', '--log-level', type=click.Choice(('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG')),
              default=DEFAULT_LOG_LEVEL, help='Logging verbosity level threshold (to stderr)')
def main(
    backend_hf: Optional[BackendUrl] = None,
    backend_llama: Optional[BackendUrl] = None,
    backend_stub: Optional[BackendUrl] = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    auth_token: Optional[List[str]] = None,
    auth_token_file: Optional[Path] = None,
    auth_token_is_user: bool = False,
    single_model: Optional[List[str]] = None,
    log_level: str = DEFAULT_LOG_LEVEL,
):
    """Run a clone of OpenAI's API Service in your Local Environment"""

    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                        level=log_level)

    sentry_sdk.init(
        integrations=[
            FlaskIntegration(),
        ],
        traces_sample_rate=0.1,
    )
    sentry_sdk.set_tag('component', 'openai-wrapper')

    all_auth_tokens = []
    if auth_token:
        all_auth_tokens = auth_token
        if all_auth_tokens:
            logging.info('Authorization tokens (API keys) read from command-line argument or environment')
        else:
            logging.info('No authorization tokens (API keys) read from command-line argument or environment')
    elif auth_token_file:
        if os.path.exists(auth_token_file):
            with open(auth_token_file) as f:
                all_auth_tokens = [line.strip() for line in f if line]
            if all_auth_tokens:
                logging.info('Authorization tokens (API keys) read from file')
            else:
                logging.warning('No authorization tokens (API keys) read from file')

    if not all_auth_tokens:
        all_auth_tokens = [generate_auth_token()]
        logging.info(f'Generated authorization token (API key): {all_auth_tokens[0]}')

    all_allowed_models = None
    if single_model:
        all_allowed_models = set(single_model)

    if all_allowed_models is not None:
        logging.info(f'Allowing only the specified model(s) to be used: {all_allowed_models}')

    if not any((backend_hf, backend_llama, backend_stub)):
        raise Exception('No backend provided')

    app = create_app(
        accepted_auth_tokens=all_auth_tokens,
        backend_hf=backend_hf,
        backend_llama=backend_llama,
        backend_stub=backend_stub,
        auth_token_is_user=auth_token_is_user,
        allowed_model_ids=all_allowed_models,
    )
    serve(app, host=host, port=port)


if __name__ == '__main__':
    main(auto_envvar_prefix='SANDLE')

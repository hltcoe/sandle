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

EXAMPLE_TEXT = '''\
I am a highly intelligent question answering bot. If you ask me a question that can be answered with one or more \
spans exactly from the given context, I will give you the answer. If you ask me a question that has no answer within \
the context, I will respond with "Unknown". An example is given below.

Context: "Thousands of people rallied in Seoul – at this mass protest, people called for President Park Geun-hye to \
step down, at another rally pro-Park protesters gathered in the South Korean capital.  A constitutional court is set \
to decide her fate later this month.  I came here out of rage towards President Park Geun-hye, said one of the \
protesters, angry at what he said were the president's constant lies.  It would have been better if this situation \
had not happened at all, admitted an anti-Park protester.  But due to the situation, people's political awareness \
has heightened, and (people) have been given an opportunity to judge for themselves, the direction of politics, he \
added. The president was impeached by the General Assembly in December over accusations of bribery and abuse of \
office.  Analysts believe the Constitutional Court will uphold the motion, which would mean fresh elections must be \
held within 60 days of the court's decision."

Q: Who is protesting?
A: Thousands of people; people; pro-Park protestors;

Q: Where is the protest?
A: Seoul; South Korean; the South Korean capital;

Q: When was the protest?
A: Unknown

Q: What was the protest against?
A: President Park Geun-hye; Park; her; the president;

Context: "Workers across Asia mark May Day Indo Asian News Service IANS India Private Limited1 May 2017 Jakarta, May \
1 (IANS) Thousands across Asia marked May Day or International Workers' Day on Monday by holding rallies and \
demanding better wages and working rights.  In Jakarta, more than 10,000 workers gathered in a rally to protest \
President Joko Widodo's labour policies, Efe news reported.  They were stopped by a road block police set up about \
500 metres from the presidential palace, manned by around 1,000 anti-riot police officers."

Q: Who is protesting?
A:\
'''


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

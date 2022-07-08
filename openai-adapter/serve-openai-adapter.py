import json
import logging
import string
import secrets
from base64 import b64encode
from functools import wraps
from typing import Any, Dict, Tuple

from flask import Flask, jsonify, make_response, Response, request
from waitress import serve


ModelData = Dict[str, Any]


END_OF_STREAM = '[DONE]'

MODELS = [
    {
        'id': model,
        'object': 'model',
        'owned_by': 'facebook',
        'permission': [],
    } for model in (
        'facebook/opt-125m',
        'facebook/opt-350m',
        'facebook/opt-1.3b',
        'facebook/opt-2.7b',
    )
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


def generate_auth_token(user: str = 'user', password_length: int = 16) -> Tuple[str, str, str]:
    alphabet = string.ascii_letters + string.digits
    password = ''.join(secrets.choice(alphabet) for _ in range(password_length))
    return (user, password, b64encode(f'{user}:{password}'))


def create_app(auth_token: str, backend_url: str) -> Flask:
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
        stream = request.json.get('stream', False)
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
                        'text': '\n\nThis is a test.',
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
    parser.add_argument('backend_url',
                        help='URL of backend language model service')
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

    if args.auth_token is not None:
        (user, password, auth_token) = generate_auth_token()
        logging.info(f'Generated authorization token user: {user}')
        logging.info(f'Generated authorization token password: {password}')
    else:
        auth_token = args.auth_token
    logging.info(f'Authorization token: {auth_token}')

    app = create_app(auth_token=auth_token, backend_url=args.backend_url)
    serve(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

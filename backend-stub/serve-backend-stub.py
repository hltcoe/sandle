import json
import logging
from random import randint
from typing import Any, Dict, Optional

from flask import Flask, jsonify, make_response, Response, request
from waitress import serve
import click


DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 8000
DEFAULT_LOG_LEVEL = 'INFO'
END_OF_STREAM = '[DONE]'

with open('models.json') as f:
    MODELS = json.load(f)


def make_api_completion() -> Dict[str, Any]:
    return {
        'id': f'stub-response-{randint(int(1e20), int(1e21))}',
        'object': 'text_completion',
        'created': 0,
        'model': 'stub',
        'choices': [
            {
                'text': ' world!',
                'index': 0,
                'logprobs': None,
                'finish_reason': 'stop',
            }
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


def create_app() -> Flask:
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
    def get_models():
        return jsonify({
            'data': MODELS,
            'object': 'list'
        })

    @app.route('/v1/completions', methods=['POST'])
    def post_completions():
        stream = request.json.get('stream', False)
        if stream:
            return Response(
                (f'data: {event_data}\n\n' for event_data in [
                    json.dumps(make_api_completion()),
                    END_OF_STREAM,
                ]),
                mimetype='text/event-stream',
                headers={'X-Accel-Buffering': 'no'},  # tell nginx not to buffer
            )
        else:
            return jsonify(make_api_completion())

    return app


@click.command()
@click.option('--host', type=str, default=DEFAULT_HOST, help='Hostname or IP to serve on')
@click.option('-p', '--port', type=int, default=DEFAULT_PORT, help='Port to serve on')
@click.option('-l', '--log-level', type=click.Choice(('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG')),
              default=DEFAULT_LOG_LEVEL, help='Logging verbosity level threshold (to stderr)')
def main(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, log_level: str = DEFAULT_LOG_LEVEL):
    """Run a stub implementation of OpenAI's /v1/completions endpoint"""

    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                        level=log_level)

    app = create_app()

    serve(app, host=host, port=port, threads=1)


if __name__ == '__main__':
    main(auto_envvar_prefix='SANDLE')

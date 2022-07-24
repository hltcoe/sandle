import json
import logging
from random import randint
from typing import Any, Dict

from flask import Flask, jsonify, Response, request
from waitress import serve


END_OF_STREAM = '[DONE]'


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


def create_app() -> Flask:
    app = Flask(__name__)

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


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Run a stub implementation of OpenAI\'s /v1/completions endpoint',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--host', default='0.0.0.0',
                        help='Hostname or IP to serve on')
    parser.add_argument('-p', '--port', type=int, default=8000,
                        help='Port to serve on')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                        level=logging.DEBUG)

    app = create_app()

    serve(app, host=args.host, port=args.port, threads=1)


if __name__ == '__main__':
    main()

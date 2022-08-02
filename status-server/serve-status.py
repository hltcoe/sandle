import logging

from flask import Flask, jsonify
from waitress import serve


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route('/status')
    def get_status():
        return jsonify({
            'code': 'ok',
            'message': 'Okay',
        })

    return app


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Run status server',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--host', default='0.0.0.0',
                        help='Hostname or IP to serve on')
    parser.add_argument('-p', '--port', type=int, default=8000,
                        help='Port to serve on')
    parser.add_argument('-l', '--log-level',
                        choices=('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'), default='INFO',
                        help='Logging verbosity level threshold (to stderr)')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                        level=args.log_level)

    app = create_app()
    serve(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

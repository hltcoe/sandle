import json
import logging

from flask import Flask, jsonify, Response, request
from transformers import AutoTokenizer, AutoModelForCausalLM
from waitress import serve


DEFAULT_MAX_TOKENS = 16
DEFAULT_NUM_COMPLETIONS = 1
END_OF_STREAM = '[DONE]'
FINISH_REASON_EOS = 'stop'
FINISH_REASON_LENGTH = 'length'


class LM:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM

    def __init__(self, repo: str = 'facebook/opt-125m'):
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo)

    def complete(self, text: str) -> str:
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors='pt')
        output = self.model.generate(inputs['input_ids'], max_length=600)
        return self.tokenizer.decode(output[0].tolist())


def create_app(model_repo: str) -> Flask:
    logging.info('Loading model')
    lm = LM(model_repo)

    logging.info('Creating app')
    app = Flask(__name__)

    @app.route('/complete', methods=['POST'])
    def complete():
        return jsonify({'text': lm.complete(request.json['text'])})

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
    parser.add_argument('--model-repo', default='facebook/opt-125m',
                        help='Repository of model to use')
    parser.add_argument('-l', '--log-level',
                        choices=('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'), default='INFO',
                        help='Logging verbosity level threshold (to stderr)')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                        level=args.log_level)

    app = create_app(model_repo=args.model_repo)
    serve(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

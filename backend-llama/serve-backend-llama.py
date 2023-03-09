import json
import logging
import os
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from time import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from uuid import uuid4

import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
import torch
from flask import Flask, jsonify, make_response, Response, request
from waitress import serve
from fairscale.nn.model_parallel.initialize import initialize_model_parallel, get_model_parallel_group
from llama import ModelArgs, Transformer, Tokenizer, LLaMA
import click


DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 8000
DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_MAX_BATCH_SIZE = 32
DEFAULT_MAX_TOKENS = 16
DEFAULT_TEMPERATURE = 1.
DEFAULT_TOP_P = 1.
DEFAULT_NUM_RETURN_SEQUENCES = 1
DEFAULT_PROMPT = 'Hello world!'
END_OF_STREAM = '[DONE]'
FINISH_REASON_EOS = 'stop'
FINISH_REASON_LENGTH = 'length'

with open('models.json') as f:
    MODELS = json.load(f)


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))

    torch.distributed.init_process_group('nccl')
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return (local_rank, world_size)


def load(
    ckpt_dir: Path,
    tokenizer_path: Path,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    checkpoints = sorted(ckpt_dir.glob('*.pth'))
    if world_size != len(checkpoints):
        raise Exception(f'Started {world_size} processes but found {len(checkpoints)} (!= {world_size}) checkpoints')
    ckpt_path = checkpoints[local_rank]
    logging.info(f'Loading model from {ckpt_dir} (tokenizer: {tokenizer_path})')
    checkpoint = torch.load(str(ckpt_path), map_location='cpu')
    with open(ckpt_dir / 'params.json', mode='r') as f:
        params = json.loads(f.read())

    model_args = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
    tokenizer = Tokenizer(model_path=str(tokenizer_path))
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)  # type: ignore
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    return LLaMA(model, tokenizer)


def generate_response_id() -> str:
    return str(uuid4())


def get_timestamp() -> int:
    return int(time())


def truncate_at_stops(text: str, stop_strings: List[str]) -> Tuple[str, bool]:
    truncated = False
    for s in stop_strings:
        index = text.find(s)
        if index >= 0:
            text = text[:index]
            truncated = True
    return (text, truncated)


class Completion(NamedTuple):
    text: str
    finish_reason: Optional[str]
    idx: int

    @classmethod
    def from_truncation(cls, truncation: Tuple[str, bool], idx: int = 0):
        (text, truncated) = truncation
        return cls(text=text, finish_reason=FINISH_REASON_EOS if truncated else FINISH_REASON_LENGTH, idx=idx)


def make_api_completions(
        response_id: str, created: int, model_id: str, completions: List[Completion]
) -> Dict[str, Any]:
    return {
        'id': response_id,
        'object': 'text_completion',
        'created': created,
        'model': model_id,
        'choices': [
            {
                'text': completion.text,
                'index': completion.idx,
                'logprobs': None,
                'finish_reason': completion.finish_reason
            }
            for completion in completions
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


class GenerateArgs(NamedTuple):
    args: Tuple[Any]
    kwargs: Dict[str, Any]


def create_app(model_id: str, generate_queue: Queue) -> Flask:
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
            (
                f'Not allowed to {request.method} on {request.path} '
                '(HINT: Perhaps you meant to use a different HTTP method?)'
            ),
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
            'data': [model_data for model_data in MODELS if model_data['id'] == model_id],
            'object': 'list'
        })

    @app.route('/v1/completions', methods=['POST'])
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

        try:
            max_tokens = int(request_json.get('max_tokens', DEFAULT_MAX_TOKENS))

            requested_model_id = request_json['model']
            if requested_model_id != model_id:
                raise Exception(f'model must be {model_id}')

            prompt = request_json.get('prompt', DEFAULT_PROMPT)
            if not isinstance(prompt, str):
                raise Exception('prompt must be a string')

            _stop = request_json.get('stop')
            if isinstance(_stop, list):
                stops = _stop
            elif isinstance(_stop, str):
                stops = [_stop]
            else:
                stops = []

            if stops:
                logging.warning('Stop sequences are implemented naively')

            num_return_sequences = int(request_json.get('n', DEFAULT_NUM_RETURN_SEQUENCES))

            stream = request_json.get('stream', False)
            if stream and num_return_sequences != 1:
                raise NotImplementedError('Streaming with more than one return sequence is not implemented')

            temperature = float(request_json.get('temperature', DEFAULT_TEMPERATURE))

            greedy_decoding = request_json.get('greedy_decoding', False)
            if greedy_decoding:
                temperature = 0

            top_p = float(request_json.get('top_p', DEFAULT_TOP_P))

            user = request_json.get('user')
            sentry_sdk.set_user({'id': user} if user else None)

            completion_log_text = 'completion' if num_return_sequences == 1 else 'completions'
            if stream:
                completion_log_text = 'streaming ' + completion_log_text
            tokens_log_text = 'token' if max_tokens == 1 else 'tokens'
            if num_return_sequences != 1:
                tokens_log_text = tokens_log_text + ' each'
            logging.debug(f'Computing {completion_log_text} of up to {max_tokens} {tokens_log_text} for user {user}')

        except Exception as ex:
            return make_error_response(
                400,
                str(ex),
                'invalid_request_error',
            )

        # Ensure generate queue is empty before putting data in
        queue_empty = False
        while not queue_empty:
            try:
                generate_queue.get_nowait()
            except Empty:
                queue_empty = True

        response_id = generate_response_id()
        created = get_timestamp()
        prompts = [prompt] * num_return_sequences
        generate_queue.put(GenerateArgs(
            args=(prompts,),
            kwargs=dict(
                max_gen_len=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ),
        ))
        api_completions = make_api_completions(
            response_id,
            created,
            model_id,
            [
                Completion.from_truncation(truncate_at_stops(raw_completion_text[len(prompt):], stop_strings=stops), i)
                for (i, raw_completion_text) in enumerate(generate_queue.get())
            ],
        )
        if stream:
            return Response(
                (f'data: {event_data}\n\n' for event_data in (json.dumps(api_completions), END_OF_STREAM)),
                mimetype='text/event-stream',
                headers={'X-Accel-Buffering': 'no'},  # tell nginx not to buffer
            )
        else:
            return jsonify(api_completions)

    return app


def serve_app(model_id: str, generate_queue: Queue, **waitress_kwargs):
    app = create_app(model_id, generate_queue)
    serve(app, **waitress_kwargs)


@click.command()
@click.argument('llama_dir', type=click.Path(file_okay=False, path_type=Path))
@click.argument('model_size', type=click.Choice(tuple(model_data['id'].split('-')[-1] for model_data in MODELS)))
@click.option('--host', type=str, default=DEFAULT_HOST, help='Hostname or IP to serve on')
@click.option('-p', '--port', type=int, default=DEFAULT_PORT, help='Port to serve on')
@click.option('-l', '--log-level', type=click.Choice(('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG')),
              default=DEFAULT_LOG_LEVEL, help='Logging verbosity level threshold (to stderr)')
@click.option('--max-seq-len', type=int, default=DEFAULT_MAX_SEQ_LEN, help='Maximum sequence length')
@click.option('--max-batch-size', type=int, default=DEFAULT_MAX_BATCH_SIZE, help='Maximum batch size')
def main(
    llama_dir: Path,
    model_size: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    log_level: str = DEFAULT_LOG_LEVEL,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE
):
    """
    Run a simplified, single-threaded clone of OpenAI's /v1/completions endpoint on the LLaMA model at
    LLAMA_DIR/MODEL_SIZE using the tokenizer at LLAMA_DIR/tokenizer.model .
    """
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(process)d] [%(name)s] %(message)s',
                        level=log_level)

    sentry_sdk.init(
        integrations=[
            FlaskIntegration(),
        ],
        traces_sample_rate=0.1,
    )
    sentry_sdk.set_tag('component', 'backend-llama')

    model_id = f'llama-{model_size}'
    ckpt_dir = llama_dir / model_size
    tokenizer_path = llama_dir / 'tokenizer.model'

    (local_rank, world_size) = setup_model_parallel()
    logging.info(f'Local rank: {local_rank}')

    if local_rank == 0:
        generate_queue = Queue()
        server_process = Process(
            target=serve_app,
            args=(model_id, generate_queue),
            kwargs=dict(host=host, port=port, threads=1),
        )
        server_process.start()
    else:
        generate_queue = None
        server_process = None

    generator = load(ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size)

    while True:
        # Rank 0 gets new generate args from queue
        if local_rank == 0:
            generate_args_list = [generate_queue.get()]
        else:
            generate_args_list = [None]
        # Rank 0 broadcasts args to other ranks
        torch.distributed.broadcast_object_list(generate_args_list, src=0, group=get_model_parallel_group())
        # All ranks process args
        [generate_args] = generate_args_list
        completed_prompts = generator.generate(*generate_args.args, **generate_args.kwargs)
        # Rank 0 puts completions onto queue
        if local_rank == 0:
            generate_queue.put(completed_prompts)

    if local_rank == 0:
        generate_queue.close()
        server_process.join()


if __name__ == '__main__':
    main(auto_envvar_prefix='SANDLE')

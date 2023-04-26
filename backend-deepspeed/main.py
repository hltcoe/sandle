"""
Run a simplified, single-threaded clone of OpenAI's /v1/completions endpoint on the specified model
in DeepSpeed.
"""

import json
import logging
from pathlib import Path
from time import time
from typing import List, Literal, Optional, Tuple, Union
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import mii
from pydantic import BaseModel, BaseSettings, Field, validator
import sentry_sdk


DEFAULT_MAX_TOKENS = 16
DEFAULT_TEMPERATURE = 1.
DEFAULT_TOP_P = 1.
DEFAULT_NUM_RETURN_SEQUENCES = 1
DEFAULT_PROMPT = 'Hello world!'
DEPLOYMENT_NAME = 'sandle_deployment'
END_OF_STREAM = '[DONE]'
FINISH_REASON_EOS = 'stop'
FINISH_REASON_LENGTH = 'length'


with open('models.json') as f:
    MODELS = json.load(f)


class Settings(BaseSettings):
    model_id: str
    model_path: Optional[Path] = None
    log_level: Literal['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'] = 'INFO'

    @validator('model_id')
    def model_id_is_configured(cls, v, **kwargs):
        if v not in [model_data['id'] for model_data in MODELS]:
            raise ValueError(f'Model {v} specified in settings is not configured in models.json.')
        return v

    class Config:
        env_prefix = 'sandle_'
        env_file = '.env'


settings = Settings()

logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(process)d] [%(name)s] %(message)s',
                    level=settings.log_level)

sentry_sdk.init(traces_sample_rate=0.1)
sentry_sdk.set_tag('component', 'backend-deepspeed')


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


class ModelData(BaseModel):
    id: str
    description: str
    object: Literal['model'] = 'model'
    owned_by: str
    permission: List = Field(default_factory=lambda: [])


class ModelList(BaseModel):
    data: List[ModelData]
    object: Literal['list'] = 'list'


class CompletionsChoice(BaseModel):
    text: str
    index: int
    logprobs: Literal[None] = None
    finish_reason: Literal[None, 'length', 'stop']

    @classmethod
    def parse_truncation(cls, truncation: Tuple[str, bool], index: int = 0):
        (text, truncated) = truncation
        return cls(text=text, index=index, finish_reason=FINISH_REASON_EOS if truncated else FINISH_REASON_LENGTH)


class CompletionsUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Completions(BaseModel):
    id: str
    object: Literal['text_completion'] = 'text_completion'
    created: int
    model_id: str
    choices: List[CompletionsChoice]
    usage: CompletionsUsage = Field(default_factory=lambda: CompletionsUsage())


class CompletionsParams(BaseModel):
    max_tokens: int = DEFAULT_MAX_TOKENS
    model: str = settings.model_id
    prompt: str = DEFAULT_PROMPT
    stop: Union[str, List[str]] = Field(default_factory=lambda: [])
    n: int = DEFAULT_NUM_RETURN_SEQUENCES
    stream: bool = False
    greedy_decoding: bool = False
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    user: Optional[str] = None

    @validator('model')
    def model_is_loaded(cls, v, **kwargs):
        if v != settings.model_id:
            raise ValueError(f'Requested model {v} but only {settings.model_id} is loaded')
        return v


def mii_query(params: CompletionsParams) -> List[str]:
    generator = mii.mii_query_handle(DEPLOYMENT_NAME)
    result = generator.query(
        {'query': [params.prompt] * params.n},
        max_new_tokens=params.max_tokens, do_sample=not params.greedy_decoding, top_p=params.top_p,
        temperature=params.temperature, num_return_sequences=params.n,
    )
    return result.response


app = FastAPI()


@app.get('/v1/models')
def get_models() -> ModelList:
    return ModelList(
        data=[ModelData(**model_data) for model_data in MODELS if model_data['id'] == settings.model_id],
    )


@app.post('/v1/completions')
def post_completions(params: CompletionsParams):
    if isinstance(params.stop, str):
        stops = [params.stop]
    else:
        stops = params.stop

    if stops:
        logging.warning('Stop sequences are implemented naively')

    if params.stream and params.n != 1:
        raise NotImplementedError('Streaming with more than one return sequence is not implemented')

    sentry_sdk.set_user({'id': params.user} if params.user else None)

    completion_log_text = 'completion' if params.n == 1 else 'completions'
    if params.stream:
        completion_log_text = 'streaming ' + completion_log_text
    tokens_log_text = 'token' if params.max_tokens == 1 else 'tokens'
    if params.n != 1:
        tokens_log_text = tokens_log_text + ' each'
    logging.debug(
        f'Computing {completion_log_text} of up to {params.max_tokens} {tokens_log_text} for user {params.user}'
    )

    raw_completions = mii_query(params)
    api_completions = Completions(
        id=generate_response_id(),
        created=get_timestamp(),
        model_id=settings.model_id,
        choices=[
            CompletionsChoice.parse_truncation(
                truncate_at_stops(raw_completion[len(params.prompt):], stop_strings=stops),
                i,
            )
            for (i, raw_completion) in enumerate(raw_completions)
        ]
    )

    if params.stream:
        return StreamingResponse(
            (f'data: {event_data}\n\n' for event_data in (json.dumps(api_completions), END_OF_STREAM)),
            media_type='text/event-stream',
            headers={'X-Accel-Buffering': 'no'},  # tell nginx not to buffer
        )
    else:
        return api_completions


mii.deploy(
    task='text-generation',
    model=settings.model_id,
    model_path=str(settings.model_path) if settings.model_path is not None else None,
    deployment_name=DEPLOYMENT_NAME,
    mii_config={'tensor_parallel': 1, 'dtype': 'fp16'},
)

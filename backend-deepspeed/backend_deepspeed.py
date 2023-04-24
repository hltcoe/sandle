from enum import Enum
import json
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, validator


DEPLOYMENT_NAME = 'sandle-deployment'

with open('models.json') as f:
    MODELS = json.load(f)


class LogLevel(str, Enum):
    @classmethod
    def _missing_(cls, value):
        return cls.__members__[value.upper()]

    CRITICAL = 'CRITICAL'
    ERROR = 'ERROR'
    WARNING = 'WARNING'
    INFO = 'INFO'
    DEBUG = 'DEBUG'


class Settings(BaseSettings):
    model_id: str
    model_path: Optional[Path] = None
    log_level: LogLevel = LogLevel.INFO

    @validator('model_id')
    def model_id_is_configured(cls, v, **kwargs):
        if v not in [model_data['id'] for model_data in MODELS]:
            raise ValueError(f'Model {v} specified in settings is not configured in models.json.')
        return v

    class Config:
        env_prefix = 'sandle_'
        env_file = '.env'


settings = Settings()

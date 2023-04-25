import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseSettings, validator


DEPLOYMENT_NAME = 'sandle_deployment'

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

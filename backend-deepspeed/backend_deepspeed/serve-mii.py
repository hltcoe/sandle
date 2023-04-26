import logging

import mii
import sentry_sdk

from util import DEPLOYMENT_NAME, settings


logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [worker] [%(process)d] [%(name)s] %(message)s',
                    level=settings.log_level)

sentry_sdk.init(traces_sample_rate=0.1)
sentry_sdk.set_tag('component', 'backend-deepspeed-worker')

mii_config = {'tensor_parallel': 1, 'dtype': 'fp16'}

mii.deploy(
    task='text-generation',
    model=settings.model_id,
    model_path=str(settings.model_path) if settings.model_path is not None else None,
    deployment_name=DEPLOYMENT_NAME,
    mii_config=mii_config,
)

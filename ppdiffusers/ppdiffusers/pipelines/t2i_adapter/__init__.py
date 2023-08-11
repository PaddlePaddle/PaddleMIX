from ...utils import (
    OptionalDependencyNotAvailable,
    is_paddle_available,
    is_paddlenlp_available, )

try:
    if not (is_paddlenlp_available() and is_paddle_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_paddle_and_paddlenlp_objects import *  # noqa F403
else:
    from .pipeline_stable_diffusion_adapter import StableDiffusionAdapterPipeline

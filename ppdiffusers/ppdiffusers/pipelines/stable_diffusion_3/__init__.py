from typing import TYPE_CHECKING

from ...utils import (
    PPDIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_paddle_available,
    is_paddlenlp_available,
)


_dummy_objects = {}
_additional_imports = {}
_import_structure = {"pipeline_output": ["StableDiffusion3PipelineOutput"]}

try:
    if not (is_paddlenlp_available() and is_paddle_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_paddle_and_paddlenlp_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_paddle_and_paddlenlp_objects))
else:
    _import_structure["pipeline_stable_diffusion_3"] = ["StableDiffusion3Pipeline"]
    _import_structure["pipeline_stable_diffusion_3_img2img"] = ["StableDiffusion3Img2ImgPipeline"]

if TYPE_CHECKING or PPDIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_paddlenlp_available() and is_paddle_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_paddle_and_paddlenlp_objects import *  # noqa F403
    else:
        from .pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
        from .pipeline_stable_diffusion_3_img2img import StableDiffusion3Img2ImgPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
    for name, value in _additional_imports.items():
        setattr(sys.modules[__name__], name, value)
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
_import_structure = {}

try:
    if not (is_paddlenlp_available() and is_paddle_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_paddle_and_paddlenlp_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_paddle_and_paddlenlp_objects))
else:
    _import_structure["pipeline_stable_diffusion_3_controlnet"] = ["StableDiffusion3ControlNetPipeline"]
    _import_structure["pipeline_stable_diffusion_3_controlnet_inpainting"] = [
        "StableDiffusion3ControlNetInpaintingPipeline"
        ]

if TYPE_CHECKING or PPDIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_paddlenlp_available() and is_paddle_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_paddle_and_paddlenlp_objects import *  # noqa F403
    else:
        from .pipeline_stable_diffusion_3_controlnet import StableDiffusion3ControlNetPipeline
        from .pipeline_stable_diffusion_3_controlnet_inpainting import StableDiffusion3ControlNetInpaintingPipeline

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
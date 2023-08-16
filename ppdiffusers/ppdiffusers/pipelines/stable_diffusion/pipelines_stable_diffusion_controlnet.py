from ...utils import deprecate
from ..controlnet.multicontrolnet import MultiControlNetModel
from ..controlnet.pipeline_controlnet import StableDiffusionControlNetPipeline
deprecate(
    'stable diffusion controlnet',
    '0.22.0',
    'Importing `StableDiffusionControlNetPipeline` or `MultiControlNetModel` from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet is deprecated. Please import `from diffusers import StableDiffusionControlNetPipeline` instead.',
    standard_warn=False,
    stacklevel=3)

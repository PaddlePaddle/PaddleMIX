from ...utils import OptionalDependencyNotAvailable, is_fastdeploy_available, is_paddle_available, is_paddlenlp_available
try:
    if not (is_paddlenlp_available() and is_paddle_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_paddle_and_paddlenlp_objects import *
else:
    from .multicontrolnet import MultiControlNetModel
    from .pipeline_controlnet import StableDiffusionControlNetPipeline
    from .pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
    from .pipeline_controlnet_inpaint import StableDiffusionControlNetInpaintPipeline
    from .pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline
if is_paddlenlp_available() and is_fastdeploy_available():
    from .pipeline_fastdeploy_stable_diffusion_controlnet import FastDeployStableDiffusionControlNetPipeline

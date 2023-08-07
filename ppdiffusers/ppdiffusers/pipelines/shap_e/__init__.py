from ...utils import OptionalDependencyNotAvailable, is_paddle_available, is_paddlenlp_available
try:
    if not (is_paddlenlp_available() and is_paddle_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_paddle_and_paddlenlp_objects import ShapEPipeline
else:
    from .camera import create_pan_cameras
    from .pipeline_shap_e import ShapEPipeline
    from .pipeline_shap_e_img2img import ShapEImg2ImgPipeline
    from .renderer import BoundingBoxVolume, ImportanceRaySampler, MLPNeRFModelOutput, MLPNeRSTFModel, ShapEParamsProjModel, ShapERenderer, StratifiedRaySampler, VoidNeRFModel

import paddle
from PIL import Image
from ppdiffusers import StableDiffusionInpaintPipeline
from paddlevlp.utils.log import logger
from .apptask import AppTask


class StableDiffusionInpaintTask(AppTask):
    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        # Default to static mode
        self._static_mode = False
        self._org_size = None
        self._resize = kwargs.get("inpainting_resize", (512, 512))

        self._construct_model(model)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """

        #bulid model
        model_instance = StableDiffusionInpaintPipeline.from_pretrained(model)

        self._model = model_instance

    def _preprocess(self, inputs):
        """
        """
        image = inputs.get("image", None)
        assert image is not None, f"The image is None"
        seg_masks = inputs.get("seg_masks", None)
        assert seg_masks is not None, f"The seg masks is None"
        inpaint_prompt = inputs.get("inpaint_prompt", None)
        assert inpaint_prompt is not None, f"The inpaint_prompt is None"

        self._org_size = image.size
        merge_mask = paddle.sum(seg_masks, axis=0).unsqueeze(0)
        merge_mask = merge_mask > 0
        mask_pil = Image.fromarray(merge_mask[0][0].cpu().numpy())

        inputs['image'] = image.resize(self._resize)
        mask_pil = mask_pil.resize(self._resize)

        inputs.pop('seg_masks', None)
        inputs['mask_pil'] = mask_pil

        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """

        result = self._model(
            inputs['inpaint_prompt'],
            image=inputs['image'],
            mask_image=inputs['mask_pil']).images[0]

        inputs.pop('mask_pil', None)
        inputs.pop('image', None)

        inputs['result'] = result

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """

        image = inputs['result'].resize(self._org_size)
        inputs['result'] = image

        return inputs

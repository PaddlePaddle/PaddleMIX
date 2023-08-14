import paddle
from ppdiffusers import StableDiffusionPipeline, VersatileDiffusionDualGuidedPipeline
from .apptask import AppTask


class StableDiffusionTask(AppTask):
    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        self._height = kwargs.get('height', 768)
        self._width = kwargs.get('width', 768)
        self._guidance_scale = kwargs.get('guidance_scale', 7.5)

        # Default to static mode
        self._static_mode = False
        self._construct_model(model)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """

        #bulid model
        model_instance = StableDiffusionPipeline.from_pretrained(model)

        self._model = model_instance

    def _preprocess(self, inputs):
        """
        """
        prompt = inputs.get("prompt", None)
        assert prompt is not None, f"The prompt is None"

        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """

        result = self._model(
            prompt=inputs['prompt'],
            guidance_scale=self._guidance_scale,
            height=self._height,
            width=self._width).images[0]

        inputs.pop('prompt', None)

        inputs['result'] = result

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """

        return inputs


class VersatileDiffusionDualGuidedTask(AppTask):
    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._text_to_image_strength = kwargs.get('text_to_image_strength',
                                                  0.75)
        # Default to static mode
        self._static_mode = False
        self._construct_model(model)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """

        #bulid model
        model_instance = VersatileDiffusionDualGuidedPipeline.from_pretrained(
            model)
        model_instance.remove_unused_weights()
        self._model = model_instance

    def _preprocess(self, inputs):
        """
        """
        prompt = inputs.get("prompt", None)
        assert prompt is not None, f"The prompt is None"
        image = inputs.get("image", None)
        assert image is not None, f"The image is None"

        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """

        result = self._model(
            prompt=inputs['prompt'],
            image=inputs['image'],
            text_to_image_strength=self._text_to_image_strength).images[0]

        inputs.pop('prompt', None)
        inputs.pop('image', None)
        inputs['result'] = result

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """

        return inputs

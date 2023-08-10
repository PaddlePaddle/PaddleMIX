import paddle
from ppdiffusers import StableDiffusionImg2ImgPipeline, StableDiffusionUpscalePipeline
from .apptask import AppTask


class StableDiffusionImg2ImgTask(AppTask):
    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        self._strength = kwargs.get('strength', 0.75)
        self._guidance_scale = kwargs.get('guidance_scale', 7.5)

        # Default to static mode
        self._static_mode = False
        self._construct_model(model)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """

        #bulid model
        model_instance = StableDiffusionImg2ImgPipeline.from_pretrained(
            model, safety_checker=None)

        self._model = model_instance

    def _preprocess(self, inputs):
        """
        """
        image = inputs.get("image", None)
        assert image is not None, f"The image is None"
        prompt = inputs.get("prompt", None)
        assert prompt is not None, f"The prompt is None"
        negative_prompt = inputs.get("negative_prompt", None)
        assert negative_prompt is not None, f"The negative_prompt is None"

        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """

        result = self._model(
            prompt=inputs['prompt'],
            negative_prompt=inputs['negative_prompt'],
            image=inputs['image'],
            guidance_scale=self._guidance_scale,
            strength=self._strength, ).images[0]

        inputs.pop('prompt', None)
        inputs.pop('negative_prompt', None)
        inputs.pop('image', None)
        inputs['result'] = result

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """

        return inputs


class StableDiffusionUpscaleTask(AppTask):
    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        # Default to static mode
        self._static_mode = False
        self._construct_model(model)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """

        #bulid model
        model_instance = StableDiffusionUpscalePipeline.from_pretrained(model)

        self._model = model_instance

    def _preprocess(self, inputs):
        """
        """
        image = inputs.get("image", None)
        assert image is not None, f"The image is None"
        prompt = inputs.get("prompt", None)
        assert prompt is not None, f"The prompt is None"

        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """

        result = self._model(
            prompt=inputs['prompt'],
            image=inputs['image'], ).images[0]

        inputs.pop('prompt', None)
        inputs.pop('image', None)
        inputs['result'] = result

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """

        return inputs

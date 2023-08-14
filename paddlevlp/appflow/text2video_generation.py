import paddle
from ppdiffusers import DPMSolverMultistepScheduler, TextToVideoSDPipeline
from .apptask import AppTask


class TextToVideoSDTask(AppTask):
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
        model_instance = TextToVideoSDPipeline.from_pretrained(model)
        model_instance.scheduler = DPMSolverMultistepScheduler.from_config(
            model_instance.scheduler.config)
        self._model = model_instance

    def _preprocess(self, inputs):
        """
        """
        prompt = inputs.get("prompt", None)
        assert prompt is not None, f"The prompt is None"
        num_inference_steps = inputs.get("num_inference_steps", 25)
        inputs['num_inference_steps'] = num_inference_steps

        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """

        result = self._model(
            prompt=inputs['prompt'],
            num_inference_steps=inputs['num_inference_steps'], ).frames

        inputs.pop('prompt', None)

        inputs['result'] = result

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """

        return inputs

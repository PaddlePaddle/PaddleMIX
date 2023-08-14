from paddlenlp import Taskflow
from paddlemix.utils.log import logger
from .apptask import AppTask


class ChatGlmTask(AppTask):
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
        model_instance = Taskflow("text2text_generation", model=model)

        self._model = model_instance

    def _preprocess(self, inputs):
        """
        """
        image = inputs.get("image", None)
        assert image is not None, f"The image is None"
        prompt = inputs.get("prompt", None)
        assert prompt is not None, f"The prompt is None"

        prompt = "Given caption,extract the main object to be replaced and marked it as 'main_object', " + \
              f"Extract the remaining part as 'other prompt', " + \
              f"Return main_object, other prompt in English" + \
              f"Given caption: {prompt}."

        inputs["prompt"] = prompt

        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """

        result = self._model(inputs["prompt"])['result'][0]

        inputs.pop('prompt', None)
        inputs['result'] = result

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """

        prompt, inpaint_prompt = inputs['result'].split('\n')[0].split(':')[
            -1].strip(), inputs['result'].split('\n')[-1].split(':')[-1].strip()

        inputs.pop('result', None)

        inputs['prompt'] = prompt
        inputs['inpaint_prompt'] = inpaint_prompt

        return inputs

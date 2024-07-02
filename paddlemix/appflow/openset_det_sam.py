# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import paddle
import paddle.nn.functional as F

from paddlemix.models.groundingdino.modeling import GroundingDinoModel
from paddlemix.models.sam.modeling import SamModel
from paddlemix.processors.groundingdino_processing import GroundingDinoProcessor
from paddlemix.processors.sam_processing import SamProcessor

from .apptask import AppTask


class OpenSetDetTask(AppTask):
    """
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        self.kwargs["openset_det_sam"] = task

        # det threshold
        self._box_threshold = kwargs.get("box_threshold", 0.3)
        self._text_threshold = kwargs.get("text_threshold", 0.25)

        # Default to static mode
        self._static_mode = kwargs.get("static_mode", False)

        self._construct_processor(model)
        self._construct_model(model)

        if self._static_mode:
            self._static_model_name = self._get_static_model_name()
            self._get_inference_model()

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        self._input_spec = [
            paddle.static.InputSpec(shape=[None, 3, None, None], name="x", dtype="float32"),  # image features
            paddle.static.InputSpec(shape=[None, None, None], name="m", dtype="int64"),  # mask
            paddle.static.InputSpec(shape=[None, None], name="input_ids", dtype="int64"),
            paddle.static.InputSpec(shape=[None, None], name="attention_mask", dtype="int64"),
            paddle.static.InputSpec(
                shape=[None, None, None],
                name="text_self_attention_masks",
                dtype="int64",
            ),
            paddle.static.InputSpec(shape=[None, None], name="position_ids", dtype="int64"),
        ]

    def _create_inputs(self, inputs):
        input_map = {}
        input_map["x"] = inputs["image_tensor"].numpy()
        input_map["m"] = np.array(inputs["mask"].numpy(), dtype="int64")

        for key in inputs["tokenized_out"].keys():
            input_map[key] = np.array(inputs["tokenized_out"][key].numpy(), dtype="int64")

            input_map[key] = np.array(inputs["tokenized_out"][key].numpy(), dtype="int64")

        for name in self.input_names:
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(input_map[name])

    def _construct_processor(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        # build processor
        self._processor = GroundingDinoProcessor.from_pretrained(model, cache_dir=self._model_dir)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """

        # build model
        model_instance = GroundingDinoModel.from_pretrained(model, cache_dir=self._model_dir)

        # Load the model parameter for the predict
        model_instance.eval()
        self._model = model_instance

    def _preprocess(self, inputs):
        """ """
        image = inputs.get("image", None)
        assert image is not None, "The image is None"
        prompt = inputs.get("prompt", None)
        assert prompt is not None, "The prompt is None"

        self._size = image.size
        image_tensor, mask, tokenized_out = self._processor(images=image, text=prompt)

        inputs["image_tensor"] = image_tensor
        inputs["mask"] = mask
        inputs["tokenized_out"] = tokenized_out

        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """

        if self._static_mode:
            self._create_inputs(inputs)
            self.predictor.run()
            pred_boxes = self.output_handle[0].copy_to_cpu()
            pred_logits = self.output_handle[1].copy_to_cpu()
            result = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
        else:
            result = self._model(
                inputs["image_tensor"],
                inputs["mask"],
                input_ids=inputs["tokenized_out"]["input_ids"],
                attention_mask=inputs["tokenized_out"]["attention_mask"],
                text_self_attention_masks=inputs["tokenized_out"]["text_self_attention_masks"],
                position_ids=inputs["tokenized_out"]["position_ids"],
            )
        inputs.pop("image_tensor", None)
        inputs.pop("mask", None)
        inputs.pop("tokenized_out", None)

        inputs["result"] = result

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """

        if self._static_mode:
            inputs["result"]["pred_logits"] = paddle.to_tensor(inputs["result"]["pred_logits"])
            inputs["result"]["pred_boxes"] = paddle.to_tensor(inputs["result"]["pred_boxes"])

        logits = F.sigmoid(inputs["result"]["pred_logits"])[0]  # (nq, 256)
        boxes = inputs["result"]["pred_boxes"][0]  # (nq, 4)

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(axis=1) > self._box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = self._processor.decode(logit > self._text_threshold)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

        H, W = self._size[1], self._size[0]
        sizes = paddle.to_tensor([W, H, W, H]).astype(boxes.dtype)
        boxes = []
        for box in zip(boxes_filt):
            box = box[0] * sizes
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = box.numpy()
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            boxes.append([x0, y0, x1, y1])

        boxes = np.array(boxes)

        inputs.pop("result", None)
        inputs.pop("prompt", None)
        inputs["labels"] = pred_phrases
        inputs["boxes"] = boxes
        return inputs

    def set_argument(self, argument: dict):
        for k, v in argument.items():
            if k == "input":
                continue
            setattr(self, f"_{k}", v)


class OpenSetSegTask(AppTask):
    """
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        self.kwargs["openset_det_sam"] = task

        self._input_type = kwargs.get("input_type", "boxs")
        # Default to static mode
        self._static_mode = kwargs.get("static_mode", False)

        self._construct_processor(model)
        self._construct_model(model)

        if self._static_mode:
            self._static_model_name = self._get_static_model_name()
            self._get_inference_model()

    def _construct_input_spec(self):
        """
        Construct the input spec for the convert dygraph model to static model.
        """
        shape = [None, 3, None, None]
        if self._input_type == "points":
            shape2 = [1, 1, 2]
        elif self._input_type == "boxs":
            shape2 = [None, 4]
        elif self._input_type == "points_grid":
            shape2 = [64, 1, 2]

        self._input_spec = [
            paddle.static.InputSpec(shape=shape, dtype="float32"),
            paddle.static.InputSpec(shape=shape2, dtype="int32"),
        ]

    def _create_inputs(self, inputs):
        input_map = {}
        input_map["img"] = inputs["image_seg"].numpy()
        input_map["prompt"] = np.array(inputs["prompt"].numpy())

        for name in self.input_names:
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(input_map[name])

    def _construct_processor(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        # build processor
        self._processor = SamProcessor.from_pretrained(model, cache_dir=self._model_dir)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """

        # build model
        model_instance = SamModel.from_pretrained(model, input_type=self._input_type, cache_dir=self._model_dir)

        # Load the model parameter for the predict
        model_instance.eval()
        self._model = model_instance

    def _preprocess(self, inputs):
        """ """
        image = inputs.get("image", None)
        assert image is not None, "The image is None"
        box_prompt = inputs.get("boxes", None)
        points_prompt = inputs.get("points", None)
        assert box_prompt is not None or points_prompt is not None, "The prompt is None"

        if box_prompt is not None:
            box_prompt = box_prompt if isinstance(box_prompt, np.ndarray) else np.array(box_prompt)
        if points_prompt is not None:
            points_prompt = np.array([points_prompt])

        image_seg, prompt = self._processor(
            image,
            input_type=self._input_type,
            box=box_prompt,
            point_coords=points_prompt,
        )

        inputs["image_seg"] = image_seg
        inputs["prompt"] = prompt

        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """

        if self._static_mode:

            if self._input_type == "boxs":
                inputs["prompt"] = inputs["prompt"].reshape([-1, 4])

            self._create_inputs(inputs)
            self.predictor.run()
            result = self.output_handle[0].copy_to_cpu()

        else:
            result = self._model(img=inputs["image_seg"], prompt=inputs["prompt"])

        inputs.pop("image_seg", None)

        inputs["result"] = result

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """

        seg_masks = self._processor.postprocess_masks(inputs["result"])
        inputs["seg_masks"] = seg_masks
        inputs.pop("result", None)
        return inputs

    def set_argument(self, argument: dict):
        for k, v in argument.items():
            if k == "input":
                continue
            setattr(self, f"_{k}", v)

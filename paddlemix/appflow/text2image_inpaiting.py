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

import paddle
from PIL import Image

from ppdiffusers import StableDiffusionInpaintPipeline

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

        # build model
        model_instance = StableDiffusionInpaintPipeline.from_pretrained(model)

        self._model = model_instance

    def _preprocess(self, inputs):
        """ """
        image = inputs.get("image", None)
        assert image is not None, "The image is None"
        seg_masks = inputs.get("seg_masks", None)
        assert seg_masks is not None, "The seg masks is None"
        inpaint_prompt = inputs.get("inpaint_prompt", None)
        assert inpaint_prompt is not None, "The inpaint_prompt is None"

        self._org_size = image.size
        if isinstance(seg_masks, paddle.Tensor):
            merge_mask = paddle.sum(seg_masks, axis=0).unsqueeze(0)
            merge_mask = merge_mask > 0
            mask_pil = Image.fromarray(merge_mask[0][0].cpu().numpy())
        else:
            mask_pil = seg_masks

        inputs["image"] = image.resize(self._resize)
        mask_pil = mask_pil.resize(self._resize)

        inputs.pop("seg_masks", None)
        inputs["mask_pil"] = mask_pil

        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """

        result = self._model(
            inputs["inpaint_prompt"],
            image=inputs["image"],
            mask_image=inputs["mask_pil"],
        ).images[0]

        inputs.pop("mask_pil", None)
        inputs.pop("image", None)

        inputs["result"] = result

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """

        image = inputs["result"].resize(self._org_size)
        inputs["result"] = image

        return inputs

# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import copy
from typing import List, Optional

import paddle

from ..models.llava.constants import IMAGE_TOKEN_INDEX
from ..models.llava.mm_utils import (
    expand2square,
    get_conversation,
    load_image,
    preprocess_multimodal,
    tokenizer_image_token,
)
from .base_processing import ProcessorMixin

__all__ = ["LlavaProcessor"]


class LlavaProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, **kwargs):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        image_paths: List[str] = None,
        prompt: Optional[str] = None,
        mode=None,
        **kwargs,
    ):
        image_aspect_ratio = kwargs.get("image_aspect_ratio", "pad")
        data_dict = {}
        images = []
        for image_path in image_paths:
            image = load_image(image_path)
            if image_aspect_ratio == "pad":
                image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))

            image = self.image_processor(image, return_tensors="pd")["pixel_values"][0]
            images.append(image)

        if mode == "train":
            if len(images) > 0:
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in prompt]), self.data_args)
                data_dict["images"] = images
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])
                data_dict["images"] = paddle.zeros(shape=[3, "crop_size[height]", "crop_size[width]"])

            data_dict = get_conversation(sources, self.tokenizer, has_image=len(images) > 0)
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0], images=data_dict["images"]
            )

        else:
            if len(images) > 0:
                image = paddle.stack(x=images, axis=0)
                data_dict["images"] = image

            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pd"
            ).unsqueeze(0)

            data_dict["input_ids"] = input_ids

        return data_dict

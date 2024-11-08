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

from typing import Optional

import paddle

from ..models.llava.constants import IMAGE_TOKEN_INDEX
from ..models.llava.mm_utils import (
    expand2square,
    is_valid_video_filename,
    load_image,
    process_anyres_image,
    sample_frames,
    tokenizer_image_token,
)
from ..models.llava.train_utils import get_conversation
from .base_processing import ProcessorMixin

__all__ = ["LlavaNextProcessor"]


class LlavaNextProcessor(ProcessorMixin):

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, **kwargs):
        super().__init__(image_processor, tokenizer)
        self.max_len = kwargs.get("max_length", 2048)
        self.image_aspect_ratio = kwargs.get("image_aspect_ratio", "square")
        self.version = kwargs.get("version", "v1")

    def __call__(
        self,
        record: Optional[dict] = None,
        mode=None,
        **kwargs,
    ):
        if record is not None:
            image_paths = record["image"] if "image" in record.keys() else []
            prompt = record["conversations"] if "conversations" in record.keys() else None

        image_aspect_ratio = self.image_aspect_ratio

        data_dict = {}
        images = []

        for image_path in image_paths:
            if is_valid_video_filename(image_path):
                images += sample_frames(image_path, record.get("num_frames", 16))
            else:
                image = load_image(image_path)

                if image_aspect_ratio == "pad":
                    image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
                    image = self.image_processor(image, return_tensors="pd")["pixel_values"][0]

                elif image_aspect_ratio == "anyres":
                    image = process_anyres_image(
                        image, self.image_processor, self.image_processor.image_grid_pinpoints
                    )

                else:
                    image = self.image_processor.preprocess(image, return_tensors="pd")["pixel_values"][0]
                images.append(image)

        if mode == "train":

            data_dict = get_conversation(self.version, prompt, self.tokenizer, has_image=len(images) > 0)
            data_dict = dict(
                input_ids=data_dict["input_ids"][0][: self.max_len].tolist(),
                labels=data_dict["labels"][0][: self.max_len].tolist(),
            )

            if len(images) > 0:
                images = paddle.stack(x=images, axis=0)
                data_dict["images"] = images
            else:
                crop_size = self.image_processor.crop_size
                data_dict["images"] = paddle.zeros(shape=[3, crop_size["height"], crop_size["width"]])

        else:
            if len(images) > 0:
                image = paddle.stack(x=images, axis=0)
                data_dict["images"] = image

            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pd"
            ).unsqueeze(0)

            data_dict["input_ids"] = input_ids[: self.max_len]

        return data_dict

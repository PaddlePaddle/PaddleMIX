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
"""
Processor class for QWEN-VL.
"""
from typing import List, Optional, Union

import numpy as np
import paddle
import requests
from paddle.vision.transforms import functional as F
from paddlenlp.transformers.tokenizer_utils_base import TensorType
from PIL import Image

from .base_processing import ProcessorMixin
from .image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from .processing_utils import BaseImageProcessor

__all__ = [
    "QwenVLProcessor",
    "QwenVLImageProcessor",
]


class QwenVLProcessor(ProcessorMixin):

    attributes = ["tokenizer"]
    tokenizer_class = "QWenTokenizer"

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer)
        self.image_start_id = kwargs.get("image_start_id", 151857)
        self.image_processor = QwenVLImageProcessor()

    def __call__(
        self,
        query: List[dict] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):

        if query is None:
            raise ValueError("You have to specify query.")
        images = []
        for ele in query:
            if "image" in ele:
                images.append(ele["image"])

        query = self.tokenizer.from_list_format(query)

        inputs = self.tokenizer(query, return_tensors=return_tensors)
        inputs["images"] = None

        if len(images) > 0:
            inputs["images"] = self.image_processor(images)

        return inputs

    def decode(self, pred: Union[List, paddle.Tensor]):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """

        return self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


class QwenVLImageProcessor(BaseImageProcessor):
    def __init__(
        self,
        image_size: int = 448,
        image_mean: Optional[Union[float, List[float]]] = [0.48145466, 0.4578275, 0.40821073],
        image_std: Optional[Union[float, List[float]]] = [0.26862954, 0.26130258, 0.27577711],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.image_size = image_size, image_size
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    def __call__(self, image_paths: List[str]):

        images = []
        for image_path in image_paths:
            if image_path.startswith("http://") or image_path.startswith("https://"):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            image = image.convert("RGB")
            images.append(self.image_transform(image))

        images = paddle.stack(x=images, axis=0)

        return images

    def image_transform(self, image):

        image = F.resize(image, size=self.image_size, interpolation="bicubic")
        tensor_normalize = paddle.vision.transforms.Normalize(
            mean=self.image_mean, std=self.image_std, data_format="HWC"
        )
        image = tensor_normalize(np.array(image) / 255.0)
        image = F.to_tensor(image)

        return image

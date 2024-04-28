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
"""
Processor class for InternLM-XComposer2.
"""
import copy
from typing import List, Optional, Union

import paddle
import paddle.vision.transforms as transforms
import requests
from paddlenlp.transformers.tokenizer_utils_base import TensorType
from PIL import Image

from .base_processing import ProcessorMixin
from .processing_utils import BaseImageProcessor, BaseTextProcessor

__all__ = ["InternLMXComposer2Processor", "InternLMXComposer2ImageProcessor", "InternLMXComposer2TextProcessor"]


class InternLMXComposer2Processor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "InternLMXComposer2Tokenizer"

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer)
        img_size = kwargs.get("img_size", 224)
        self.max_length = kwargs.get("max_length", 4096)
        self.image_processor = InternLMXComposer2ImageProcessor(img_size)
        self.text_processor = InternLMXComposer2TextProcessor()

    def __call__(
        self,
        query: List[dict] = None,
        record: List[dict] = None,
        mode: str = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        if query is None and record is None:
            raise ValueError("You have to specify query or record.")
        if query is None:
            query = record

        if mode == "train":
            inputs = self.train_preprocess(query)

        else:  # TODO: what's this for? chat?
            images = []
            for ele in query:
                if "image" in ele:
                    images.append(ele["image"])

            inputs = self.tokenizer(query, return_tensors=return_tensors)
            if len(images) > 0:
                inputs["images"] = self.image_processor(images)

        return inputs

    def train_preprocess(self, sources):
        import re

        image_pattern = re.compile(r"<img>.*</img>")
        image_path = []

        if isinstance(sources, dict) and "conversations" in sources.keys():
            sources = sources["conversations"][0]

            sources = self.text_processor(sources)
        if "<img>" in sources:
            result = image_pattern.findall(sources)
            for ele in result:
                image_path.append(ele[5:-6])

        inputs = dict(text_input=sources)

        text = inputs["text_input"]
        if len(image_path) > 0:
            text_tokens, text = self.interleav_wrap(text, image_path)
        else:  # TODO: adjust for pure text input
            text_tokens = self.tokenizer(
                text,
                return_tensors="pd",
                padding="longest",
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )
        inputs = {
            "input_tokens": text_tokens,
            "input_text": text,
        }
        if len(image_path) > 0:
            inputs["images"] = self.image_processor(image_path)

        return inputs

    def interleav_wrap(self, text, img_path_list):
        img_path_list = [f"<img>{p}</img>" for p in img_path_list]
        parts = text
        for img_path in img_path_list:
            parts = parts.replace(img_path, "<ImageHere>")
        text = parts
        parts = text.split("<ImageHere>")

        wrap_tokens = []
        need_bos = True
        for idx, part in enumerate(parts):
            if len(part) > 0:
                part_tokens = self.tokenizer(part, return_tensors="pd", padding="longest", add_special_tokens=need_bos)
                if need_bos:
                    need_bos = False
                wrap_tokens.append(part_tokens)
        return wrap_tokens, text

    def text2emb(self, text, add_special=False):
        to_regress_tokens = self.tokenizer(
            text,
            return_tensors="pd",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=add_special,
        )
        targets = self.mask_human_targets(to_regress_tokens.input_ids)
        return to_regress_tokens, targets

    def mask_human_targets(self, input_ids, pure=False):
        target_batch = []
        for bs in range(input_ids.shape[0]):
            ids = input_ids[bs]
            targets = copy.deepcopy(ids)
            end_count = 0
            last_eoa = 0
            for i, temp_id in enumerate(ids):
                if temp_id == 92542:
                    if end_count % 2 == 0:
                        targets[last_eoa : i + 6] = -100
                    else:
                        last_eoa = i + 1
                    end_count += 1
                elif temp_id == 2:
                    targets[i + 1 :] = -100
                    break
            if temp_id != 2 and end_count % 2 == 0:
                targets[last_eoa + 1 :] = -100
            target_batch.append(targets.unsqueeze(axis=0))
        target_batch = paddle.concat(x=target_batch, axis=0)
        return target_batch

    def batch_decode(self, *args, **kwargs):

        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """

        return self.tokenizer.batch_decode(*args, **kwargs)

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


class InternLMXComposer2ImageProcessor(BaseImageProcessor):
    def __init__(self, image_size=224, **kwargs):
        super().__init__(**kwargs)
        mean = 0.48145466, 0.4578275, 0.40821073
        std = 0.26862954, 0.26130258, 0.27577711
        self.normalize = transforms.Normalize(mean, std)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation="bicubic"),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

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


class InternLMXComposer2TextProcessor(BaseTextProcessor):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def __call__(self, sources):
        END_HUMAN = "[UNUSED_TOKEN_145]\n"
        END_BOT = "[UNUSED_TOKEN_145]\n"
        conversation = (
            "[UNUSED_TOKEN_146]user\n"
            + sources[0].strip()
            + END_HUMAN
            + "[UNUSED_TOKEN_146]assistant\n"
            + sources[1].strip()
            + END_BOT
            + "</s>"
        )
        return conversation

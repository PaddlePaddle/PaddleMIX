# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.utils.import_utils import import_module

from .constant import (
    IMAGE_PROCESSOR_MAPPING,
    SUPPORTED_MODELS,
    TOKENIZER_MAPPING,
    VL_PROCESSOR_MAPPING,
)


def get_processor(model_name, model_path, **kwargs):
    if model_name in SUPPORTED_MODELS.keys():
        img_processor_cls = import_module(f"paddlemix.processors.{IMAGE_PROCESSOR_MAPPING[model_name]}")
        image_processor = img_processor_cls()
        tokenizer = import_module(f"paddlemix.models.{TOKENIZER_MAPPING[model_name]}").from_pretrained(
            model_path, padding_side="left"
        )
        vl_processor_cls = import_module(f"paddlemix.processors.{VL_PROCESSOR_MAPPING[model_name]}")
        processor = vl_processor_cls(image_processor, tokenizer)

        pad_token_id = processor.tokenizer.pad_token_id
        processor.pad_token_id = pad_token_id
        processor.eos_token_id = processor.tokenizer.eos_token_id
        if kwargs.get("max_pixels", None):
            processor.image_processor.max_pixels = kwargs["max_pixels"]
        if kwargs.get("min_pixels", None):
            processor.image_processor.min_pixels = kwargs["min_pixels"]
    else:
        raise ValueError(f"Invalid model: {model_name}")
    return processor, tokenizer


# TODO
def get_tokenizer():
    pass

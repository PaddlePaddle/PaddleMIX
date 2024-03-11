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

from paddlenlp.transformers.t5.tokenizer import T5Tokenizer as PPNLPT5Tokenizer

__all__ = ["T5Tokenizer"]


class T5Tokenizer(PPNLPT5Tokenizer):
    model_input_names = [
        "input_ids",
        "attention_mask",
    ]

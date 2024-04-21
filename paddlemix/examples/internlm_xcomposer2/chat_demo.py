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

import argparse

import paddle

from paddlemix.models.internlm_xcomposer2.modeling import InternLMXComposer2ForCausalLM
from paddlemix.models.internlm_xcomposer2.tokenizer import InternLMXComposer2Tokenizer

paddle.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--from_pretrained", type=str, default="internlm/internlm-xcomposer2-7b", help="pretrained ckpt and tokenizer"
)
args = parser.parse_args()

MODEL_PATH = args.from_pretrained
# init model and tokenizer
model = InternLMXComposer2ForCausalLM.from_pretrained(MODEL_PATH).eval()
tokenizer = InternLMXComposer2Tokenizer.from_pretrained(MODEL_PATH)

text = "<ImageHere>Please describe this image in detail."
image = "../image1.jpg"
with paddle.no_grad():
    response, _ = model.chat(tokenizer, query=text, image=image, history=[], do_sample=False)
print(response)

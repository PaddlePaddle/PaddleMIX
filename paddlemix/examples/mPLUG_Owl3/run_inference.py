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

from paddlenlp.transformers import Qwen2Tokenizer
from PIL import Image

from paddlemix.models.mPLUGOwl3.configuration_mplugowl3 import mPLUGOwl3Config
from paddlemix.models.mPLUGOwl3.modeling_mplugowl3 import mPLUGOwl3Model

parser = argparse.ArgumentParser()
parser.add_argument("--dtype", type=str, default="bfloat16")
args = parser.parse_args()


model_path = "mPLUG/mPLUG-Owl3-7B-241101"

config = mPLUGOwl3Config.from_pretrained(model_path)
model = mPLUGOwl3Model.from_pretrained(model_path, dtype=args.dtype).eval()
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
processor = model.init_processor(tokenizer)

# image = Image.new('RGB', (500, 500), color='red')
image = Image.open("paddlemix/demo_images/examples_image1.jpg").convert("RGB")

messages = [{"role": "user", "content": """<|image|>Describe this image."""}, {"role": "assistant", "content": ""}]

inputs = processor(messages, images=[image], videos=None)
inputs["pixel_values"] = inputs["pixel_values"].cast(args.dtype)

inputs.update(
    {
        "tokenizer": tokenizer,
        "max_new_tokens": 512,  #
        "decode_text": True,
    }
)

res = model.generate(**inputs)
print("output:\n", res)

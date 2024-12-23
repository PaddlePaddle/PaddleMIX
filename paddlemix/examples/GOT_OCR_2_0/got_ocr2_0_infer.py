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
from paddlenlp.transformers import QWenTokenizer

from paddlemix.models.GOT.GOT_ocr_2_0 import GOTQwenForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name_or_path", type=str, default="stepfun-ai/GOT-OCR2_0", help="pretrained ckpt and tokenizer"
)
parser.add_argument("--image_file", type=str, default="paddlemix/demo_images/hospital.jpeg")
parser.add_argument("--multi_crop", action="store_true")
parser.add_argument("--ocr_type", type=str, default="plain", choices=["ocr", "format"])
parser.add_argument("--box", type=str, default="")
parser.add_argument("--color", type=str, default="")
parser.add_argument("--render", action="store_true")
parser.add_argument("--dtype", type=str, default="bfloat16")

args = parser.parse_args()
model_name_or_path = args.model_name_or_path

tokenizer = QWenTokenizer.from_pretrained(model_name_or_path)
model = GOTQwenForCausalLM.from_pretrained(
    model_name_or_path, dtype=args.dtype, pad_token_id=tokenizer.eos_token_id
).eval()

# input test image
image_file = args.image_file
with paddle.no_grad():
    if args.multi_crop:
        # multi-crop OCR:
        res = model.chat_crop(tokenizer, image_file, ocr_type=args.ocr_type, dtype=args.dtype)
    else:
        # plain texts OCR
        # format texts OCR
        res = model.chat(tokenizer, image_file, ocr_type=args.ocr_type, dtype=args.dtype)
    print("output:\n", res)

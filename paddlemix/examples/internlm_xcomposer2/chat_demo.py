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


def main(args):
    if args.image_path is not None:
        args.text = "<ImageHere>" + args.text
    MODEL_PATH = args.model_name_or_path
    # init model and tokenizer
    model = InternLMXComposer2ForCausalLM.from_pretrained(MODEL_PATH).eval()
    tokenizer = InternLMXComposer2Tokenizer.from_pretrained(MODEL_PATH)

    with paddle.no_grad():
        response, _ = model.chat(tokenizer, query=args.text, image=args.image_path, history=[], do_sample=False)
    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="internlm/internlm-xcomposer2-7b",
        help="pretrained ckpt and tokenizer",
    )
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()
    main(args)

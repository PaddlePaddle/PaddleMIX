# -*- coding: utf-8 -*-

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

# @Time    : 2025/4/19 下午8:37
# @Author  : zhaop-l(zhaopuzxjc@126.com)
import argparse

from paddlenlp.transformers import CLIPImageProcessor, Qwen2Tokenizer
from PIL import Image

from paddlemix.models.points_qwen2_5 import POINTSChatModel


def main(args):
    model_path = args.model_path

    model = POINTSChatModel.from_pretrained(model_path)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    image_processor = CLIPImageProcessor.from_pretrained(model_path)

    image_path = args.image_file
    pil_image = Image.open(image_path)
    question = args.question

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_beams": 1,
    }
    res = model.chat(pil_image, question, tokenizer, image_processor, True, generation_config)

    print(f"User: {question}\nAssistant: {res}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/POINTS-Qwen-2-5-7B-Chat_pd")
    parser.add_argument("--question", type=str, default="please describe the image in detail")
    parser.add_argument("--image_file", type=str, default="paddlemix/demo_images/examples_image2.jpg")
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()
    main(args)

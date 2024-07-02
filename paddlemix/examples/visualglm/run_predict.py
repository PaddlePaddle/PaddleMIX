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

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["FLAGS_use_cuda_managed_memory"] = "true"

import requests
from PIL import Image

from paddlemix import VisualGLMForConditionalGeneration, VisualGLMProcessor
from paddlemix.utils.downloader import is_url


def predict(args):
    # load VisualGLM moel and processor
    model = VisualGLMForConditionalGeneration.from_pretrained(args.pretrained_name_or_path, dtype="float16")
    model.eval()
    processor = VisualGLMProcessor.from_pretrained(args.pretrained_name_or_path)
    print("load processor and model done!")
    image_path = args.image_path
    if is_url(image_path):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)
    generate_kwargs = {
        "max_length": 1024,
        "min_length": 10,
        "num_beams": 1,
        "top_p": 1.0,
        "top_k": 1,
        "repetition_penalty": 1.2,
        "temperature": 0.8,
        "decode_strategy": "sampling",
        "eos_token_id": processor.tokenizer.eos_token_id,
    }

    # Epoch 1
    query = "写诗描述一下这个场景"
    history = []
    inputs = processor(image, query)

    generate_ids, _ = model.generate(**inputs, **generate_kwargs)
    responses = processor.get_responses(generate_ids)
    history.append([query, responses[0]])
    print(responses)

    # Epoch 2
    query = "这部电影的导演是谁？"
    inputs = processor(image, query, history=history)
    generate_ids, _ = model.generate(**inputs, **generate_kwargs)
    responses = processor.get_responses(generate_ids)
    history.append([query, responses[0]])
    print(responses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_name_or_path",
        default="THUDM/visualglm-6b",
        type=str,
        help="The dir name of visualglm checkpoint.",
    )
    parser.add_argument(
        "--image_path",
        default="https://paddlenlp.bj.bcebos.com/data/images/mugs.png",
        type=str,
        help="",
    )
    args = parser.parse_args()

    predict(args)

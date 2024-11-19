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
import paddle.vision.transforms as T
from paddlenlp.transformers import Llama3Tokenizer, LlamaTokenizer, Qwen2Tokenizer
from PIL import Image

from paddlemix.datasets.internvl_dataset import dynamic_preprocess
from paddlemix.models.internvl2.internlm2 import InternLM2Tokenizer
from paddlemix.models.internvl2.internvl_chat import InternVLChatModel

paddle.set_grad_enabled(False)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            # T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation="bicubic"),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = paddle.stack(pixel_values)
    return pixel_values


def load_tokenizer(model_path):
    import re

    match = re.search(r"\d+B", model_path)
    if match:
        model_size = match.group()
    else:
        model_size = "2B"

    if model_size in ["1B"]:
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    elif model_size in ["2B", "8B", "26B"]:
        tokenizer = InternLM2Tokenizer.from_pretrained(model_path)
    elif model_size in ["40B"]:
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
    elif model_size in ["76B"]:
        tokenizer = Llama3Tokenizer.from_pretrained(model_path)
    else:
        raise ValueError

    return tokenizer


def main(args):
    if args.image_path is not None and args.image_path != "None":
        pixel_values = load_image(args.image_path, max_num=12).to(paddle.bfloat16)
        args.text = "<image>\n" + args.text

    else:
        pixel_values = None

    # init model and tokenizer
    MODEL_PATH = args.model_name_or_path
    model_size = MODEL_PATH.split("-")[-1]
    print(f"model size: {model_size}")
    tokenizer = load_tokenizer(MODEL_PATH)
    print("tokenizer:\n", tokenizer)
    print("len(tokenizer): ", len(tokenizer))

    model = InternVLChatModel.from_pretrained(MODEL_PATH).eval()

    generation_config = dict(max_new_tokens=1024, do_sample=False)

    with paddle.no_grad():
        response, history = model.chat(
            tokenizer, pixel_values, args.text, generation_config, history=None, return_history=True
        )
        print(f"User: {args.text}\nAssistant: {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="OpenGVLab/InternVL2-8B",
        help="pretrained ckpt and tokenizer",
    )
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--text", type=str, default="Please describe the image shortly.", required=True)
    args = parser.parse_args()
    main(args)

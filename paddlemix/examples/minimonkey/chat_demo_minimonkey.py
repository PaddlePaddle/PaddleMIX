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
from PIL import Image

from paddlemix.datasets.internvl_dataset import dynamic_preprocess, dynamic_preprocess2
from paddlemix.models.internvl2.internlm2 import InternLM2Tokenizer
from paddlemix.models.internvl2.internvl_chat import MiniMonkeyChatModel

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


def load_image(image_file, input_size=448, min_num=1, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images, target_aspect_ratio = dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        min_num=min_num,
        max_num=max_num,
        return_target_aspect_ratio=True,
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = paddle.stack(pixel_values)
    return pixel_values, target_aspect_ratio


def load_image2(image_file, input_size=448, min_num=1, max_num=12, target_aspect_ratio=None):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess2(
        image,
        image_size=input_size,
        use_thumbnail=True,
        min_num=min_num,
        max_num=max_num,
        prior_aspect_ratio=target_aspect_ratio,
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = paddle.stack(pixel_values)
    return pixel_values


def main(args):
    assert args.image_path is not None and args.image_path != "None"
    pixel_values, target_aspect_ratio = load_image(args.image_path, min_num=4, max_num=12)
    pixel_values = pixel_values.to(paddle.bfloat16)
    pixel_values2 = load_image2(args.image_path, min_num=3, max_num=7, target_aspect_ratio=target_aspect_ratio).to(
        paddle.bfloat16
    )
    pixel_values = paddle.concat([pixel_values2[:-1], pixel_values[:-1], pixel_values2[-1:]], 0)
    pixel_values = pixel_values.to(paddle.bfloat16)

    args.text = "<image>\n" + args.text

    # init model and tokenizer
    MODEL_PATH = args.model_name_or_path
    tokenizer = InternLM2Tokenizer.from_pretrained(MODEL_PATH)
    # TODO:
    tokenizer.added_tokens_encoder = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "<|plugin|>": 92538,
        "<|interpreter|>": 92539,
        "<|action_end|>": 92540,
        "<|action_start|>": 92541,
        "<|im_end|>": 92542,
        "<|im_start|>": 92543,
        "<img>": 92544,
        "</img>": 92545,
        "<IMG_CONTEXT>": 92546,
        "<quad>": 92547,
        "</quad>": 92548,
        "<ref>": 92549,
        "</ref>": 92550,
        "<box>": 92551,
        "</box>": 92552,
    }
    tokenizer.added_tokens_decoder = {v: k for k, v in tokenizer.added_tokens_encoder.items()}

    print("tokenizer:\n", tokenizer)
    print("len(tokenizer): ", len(tokenizer))

    model = MiniMonkeyChatModel.from_pretrained(MODEL_PATH).eval()

    generation_config = dict(max_new_tokens=512, do_sample=False)

    with paddle.no_grad():
        response, history = model.chat(
            tokenizer,
            pixel_values,
            target_aspect_ratio,
            args.text,
            generation_config,
            use_scm=True,
            history=None,
            return_history=True,
        )
        print(f"User: {args.text}\nAssistant: {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="HUST-VLRLab/Mini-Monkey",
        help="pretrained ckpt and tokenizer",
    )
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--text", type=str, default="Please describe the image shortly.", required=True)
    args = parser.parse_args()
    main(args)

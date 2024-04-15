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

from paddlemix.auto import AutoModelMIX
from paddlemix.utils.log import logger


def export(args):
    compute_dtype = "float16" if args.fp16 else "bfloat16"
    if not paddle.amp.is_bfloat16_supported():
        logger.warning("bfloat16 is not supported on your device,change to float32")
        compute_dtype = "float32"

    model = AutoModelMIX.from_pretrained(args.model_name_or_path, dtype=compute_dtype)
    vision_tower = model.get_vision_tower()
    vision_tower.load_model()
    model.eval()
    # convert to static graph with specific input description
    model = paddle.jit.to_static(
        model.encode_images,
        input_spec=[
            paddle.static.InputSpec(shape=[None, 3, 336, 336], dtype="float32"),  # images
        ],
    )

    # save to static model
    paddle.jit.save(model, args.save_path)
    print(f"static model has been to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="paddlemix/llava/llava-v1.5-7b",
        type=str,
        help="The dir name of llava checkpoint.",
    )
    parser.add_argument(
        "--save_path",
        default="./checkpoints/encode_image/clip",
        type=str,
        help="The saving path of static llava vision.",
    )
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    export(args)

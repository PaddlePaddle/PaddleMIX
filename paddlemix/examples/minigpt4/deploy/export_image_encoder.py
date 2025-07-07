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

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["FLAGS_use_cuda_managed_memory"] = "true"

import paddle

from paddlemix import MiniGPT4ForConditionalGeneration


def export(args):
    model = MiniGPT4ForConditionalGeneration.from_pretrained(args.minigpt4_13b_path, vit_dtype="float16")
    model.eval()

    # convert to static graph with specific input description
    model = paddle.jit.to_static(
        model.encode_images,
        input_spec=[
            paddle.static.InputSpec(shape=[None, 3, None, None], dtype="float32"),  # images
        ],
    )

    # save to static model
    paddle.jit.save(model, args.save_path)
    print(f"static model has been to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--minigpt4_13b_path",
        default="your minigpt4 dir path",
        type=str,
        help="The dir name of minigpt4 checkpoint.",
    )
    parser.add_argument(
        "--save_path",
        default="./checkpoints/encode_image/encode_image",
        type=str,
        help="The saving path of static minigpt4.",
    )
    args = parser.parse_args()

    export(args)

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import paddle
import yaml

from paddlemix.models.sam.modeling import SamModel
from paddlemix.utils.log import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Export Inference Model.")
    parser.add_argument(
        "--model_type",
        choices=["Sam/SamVitH-1024", "Sam/SamVitB", "Sam/SamVitL"],
        required=True,
        help="The model type.",
        type=str,
    )
    parser.add_argument(
        "--input_type",
        choices=["boxs", "points", "points_grid"],
        required=True,
        help="The model type.",
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        help="The directory for saving the exported inference model",
        type=str,
        default="./output/inference_model",
    )
    parser.add_argument(
        "--input_img_shape",
        nargs="+",
        help="Export the model with fixed input shape, e.g., `--input_img_shape 1 3 512 1024`.",
        type=int,
        default=[1, 3, 1024, 1024],
    )

    return parser.parse_args()


def main(args):

    os.environ["PADDLESEG_EXPORT_STAGE"] = "True"

    model = SamModel.from_pretrained(args.model_type, input_type=args.input_type)

    shape = [None, 3, None, None] if args.input_img_shape is None else args.input_img_shape
    if args.input_type == "points":
        shape2 = [1, 1, 2]
    elif args.input_type == "boxs":
        shape2 = [None, 4]
    elif args.input_type == "points_grid":
        shape2 = [64, 1, 2]

    input_spec = [
        paddle.static.InputSpec(shape=shape, dtype="float32"),
        paddle.static.InputSpec(shape=shape2, dtype="int32"),
    ]
    model.eval()
    model = paddle.jit.to_static(model, input_spec=input_spec)
    save_path = f"{args.model_type}_{args.input_type}"
    paddle.jit.save(model, os.path.join(save_path, "model"))

    # TODO add test config
    deploy_info = {
        "Deploy": {
            "model": "model.pdmodel",
            "params": "model.pdiparams",
            "input_img_shape": shape,
            "input_prompt_shape": shape2,
            "input_prompt_type": args.input_type,
            "model_type": args.model_type,
            "output_dtype": "float32",
        }
    }
    msg = "\n---------------Deploy Information---------------\n"
    msg += str(yaml.dump(deploy_info))
    logger.info(msg)

    yml_file = os.path.join(save_path, "deploy.yaml")
    with open(yml_file, "w") as file:
        yaml.dump(deploy_info, file)

    logger.info(f"The inference model is saved in {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

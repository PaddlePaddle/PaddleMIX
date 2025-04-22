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
from llama_inference_model import LlamaForClipInferenceModel

from paddlemix.models.llava.language_model.llava_llama import (
    LlavaConfig,
    LlavaLlamaForCausalLM,
)
from paddlemix.utils.log import logger


def export_encode_text(model, config, compute_dtype):

    # save to static model
    save_path = args.save_path + "/encode_text/llama"
    model.to_static(save_path, config, compute_dtype)
    logger.info(f"static model has been to {save_path}")


def export_encode_image(model, compute_dtype):
    paddle.save(model.llama.image_newline, args.save_path + "/encode_image/clip/image_newline.pdparams")
    # convert to static graph with specific input description

    model = paddle.jit.to_static(
        model.encode_images,
        input_spec=[
            paddle.static.InputSpec(shape=[None, 3, 336, 336], dtype=compute_dtype),  # images
        ],
    )

    # save to static model
    save_path = args.save_path + "/encode_image/clip"
    paddle.jit.save(model, save_path)
    logger.info(f"static model has been to {save_path}")


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
        default="./llava_static",
        type=str,
        help="The saving path of static llava vision.",
    )
    parser.add_argument("--encode_image", action="store_true")
    parser.add_argument("--encode_text", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    compute_dtype = "float16" if args.fp16 else "bfloat16"
    if not paddle.amp.is_bfloat16_supported() and compute_dtype == "bfloat16":
        logger.warning("bfloat16 is not supported on your device,change to float32")
        compute_dtype = "float32"

    if args.encode_image:

        model = LlavaLlamaForCausalLM.from_pretrained(args.model_name_or_path, dtype=compute_dtype)
        vision_tower = model.get_vision_tower()
        vision_tower.load_model()
        model.eval()

        export_encode_image(model, compute_dtype)

    elif args.encode_text:

        config = LlavaConfig.from_pretrained(args.model_name_or_path)
        config.tensor_parallel_degree = 1
        config.tensor_parallel_rank = 0
        config.weight_only_quant_bits = -1
        config.quant_type = "none"
        config.cachekv_int8_type = "none"
        config.append_attn = False

        model = LlamaForClipInferenceModel.from_pretrained(args.model_name_or_path, config=config)

        model.to(dtype=compute_dtype)
        model.eval()

        export_encode_text(model, config, compute_dtype)

    else:
        logger.info("please specify the task to export,--encode_image or --encode_text")

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
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["FLAGS_use_cuda_managed_memory"] = "true"

import paddle
from paddlenlp.transformers import LlamaForCausalLM
from paddlenlp.transformers.opt.modeling import OPTForCausalLM
from paddlenlp.transformers.t5.modeling import T5ForConditionalGeneration

from paddlemix.utils.log import logger


def merge(args):
    model_dict = {}
    # load the first item: vision_model
    visual_encoder_state_dict = paddle.load(args.vision_name_or_path)
    for n, p in visual_encoder_state_dict.items():
        if n.startswith("visual_encoder"):
            model_dict[n] = p
        else:
            model_dict["visual_encoder." + n] = p
    logger.info("[1/2] load visual_encoder done!")
    # load the second item: Qformer
    visual_encoder_state_dict = paddle.load(args.vision_name_or_path)
    for n, p in visual_encoder_state_dict.items():
        if n.startswith("Qformer"):
            model_dict[n] = p
        else:
            model_dict["Qformer." + n] = p
    logger.info("[1/2] load Qformer done!")
    if args.llm_path:
        # load the second item: llm model
        if "opt" in args.llm_path:
            llm_model = OPTForCausalLM.from_pretrained(args.llm_path)
        elif "llama" in args.llm_path:
            llm_model = LlamaForCausalLM.from_pretrained(args.llm_path)
        elif "t5" in args.llm_path:
            llm_model = T5ForConditionalGeneration.from_pretrained(args.llm_path)
        else:
            ValueError(f"The LLM model {args.llm_path} is not supported.")

        for n, p in llm_model.named_parameters():
            new_name = "language_model." + n
            model_dict[new_name] = p
    logger.info("load extra language_model done!")

    save_path = os.path.join(args.save_path, "model_state.pdparams")
    paddle.save(model_dict, save_path)
    logger.info("The checkpoint of blip2 has been saved to :{}".format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vision_name_or_path",
        default="blip2-stage2/eva_vit_g/model_state.pdparams",
        type=str,
        help="The dir name of visual_encoder.",
    )
    parser.add_argument(
        "--bridge_name_or_path",
        default="blip2-stage2/Qformer/model_state.pdparams",
        type=str,
        help="The checkpoint path of Qformer.",
    )
    parser.add_argument(
        "--llm_path",
        default=None,
        type=str,
        help="The checkpoint path of language model.",
    )
    parser.add_argument(
        "--save_path",
        default="/save/to/dirname",
        type=str,
        help="The saving path of blip2.",
    )
    args = parser.parse_args()

    args.vision_name_or_path = os.path.join(args.vision_name_or_path, "model_state.pdparams")
    if not os.path.exists(args.vision_name_or_path):
        raise ValueError("Not found the file: {}".format(args.vision_name_or_path))
    if not os.path.isdir(args.bridge_name_or_path):
        raise ValueError("It is not a directory: {}".format(args.bridge_name_or_path))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    merge(args)

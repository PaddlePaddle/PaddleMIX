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
from paddlemix.utils.log import logger
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["FLAGS_use_cuda_managed_memory"] = "true"

import paddle
import torch

from paddlenlp.transformers import LlamaForCausalLM
from paddlenlp.transformers.opt.modeling import OPTForCausalLM


def merge(args):
    model_dict = {}
    # load the first item: vision_model
    state_dict = paddle.load(args.blip2_path)
    for n, p in state_dict.items():
        if n.startswith("vision_model") or n.startswith(
                "qformer") or n == "query_tokens":
            model_dict[n] = p
    logger.info("[1/3] load ViT, qformer and query_tokens done!")

    # load the second item: llm model
    if "opt" in args.llm_name:
        llm_model = OPTForCausalLM.from_pretrained(args.llm_path)
    elif "llama" in args.llm_name:
        llm_model = LlamaForCausalLM.from_pretrained(args.llm_path)
    else:
        ValueError(f"The LLM model {args.llm_name} is not supported.")

    for n, p in llm_model.named_parameters():
        new_name = "language_model." + n
        model_dict[new_name] = p
    logger.info("[2/3] load language_model done!")

    # load the third item: blip2
    llm_state_dict = torch.load(args.llm_path)
    for n, p in llm_state_dict["model"].items():
        if n.startswith(args.llm_name + "_model.model"):
            new_name = n.replace(args.llm_name + "_model.model",
                                 "language_model." + args.llm_name)
            new_p = paddle.to_tensor(p.cpu().numpy())
            model_dict[new_name] = new_p

        if n.startswith(args.llm_name + args.llm_name + "_proj"):
            new_name = n.replace(args.llm_name + "_proj", "language_projection")
            if n.endswith("weight"):
                new_p = paddle.to_tensor(p.cpu().numpy()).transpose([1, 0])
            else:
                new_p = paddle.to_tensor(p.cpu().numpy())
            model_dict[new_name] = new_p

    logger.info(
        "[3/3] load language_projection, some llm weights from blip2 done!")

    save_path = os.path.join(args.save_path, "model_state.pdparams")
    paddle.save(model_dict, save_path)
    logger.info("The checkpoint of blip2 has been saved to :{}".format(
        save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--blip2_path",
        default="/blip2/dirname",
        type=str,
        help="The dir name of blip2-flan-t5-xxl.")
    parser.add_argument(
        "--llm_name", default="opt", type=str, help="Thename of llm model.")
    parser.add_argument(
        "--llm_path",
        default="/llm/dirname",
        type=str,
        help="The dir name of llm model.")
    parser.add_argument(
        "--blip2_path",
        default="/blip2/prerained_blip2.pth",
        type=str,
        help="The checkpoint path of blip2.")
    parser.add_argument(
        "--save_path",
        default="/save/to/dirname",
        type=str,
        help="The saving path of blip2.")
    args = parser.parse_args()

    args.blip2_path = os.path.join(args.blip2_path, "model_state.pdparams")
    if not os.path.exists(args.blip2_path):
        raise ValueError("Not found the file: {}".format(args.blip2_path))
    if not os.path.isdir(args.llm_path):
        raise ValueError("It is not a directory: {}".format(args.llm_path))
    if not os.path.exists(args.llm_path):
        raise ValueError("Not found the file: {}".format(args.llm_path))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    merge(args)

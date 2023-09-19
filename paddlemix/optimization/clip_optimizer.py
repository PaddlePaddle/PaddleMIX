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

import json
import logging
import re

import paddle


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def get_num_layer_for_transformer(param_name, num_max_layer):
    layer_0 = {
        "patch_embed",
        "pos_embed",
        "cls_token",
        "mask_token",
        "conv1",
        "positional_embedding",
        "token_embedding",
        "transformer.embeddings.word_embeddings",
        "transformer.embeddings.position_embeddings",
        "transformer.embeddings.token_type_embeddings",
    }
    if any(l in param_name for l in layer_0):
        return 0
    block_regex = re.compile("blocks\\.([0-9]+)\\.")
    match_block = block_regex.search(param_name)
    layer_regex = re.compile("layer\\.([0-9]+)\\.")
    match_layer = layer_regex.search(param_name)
    if match_block is not None:
        return int(match_block.group(1)) + 1
    elif match_layer is not None:
        return int(match_layer.group(1)) + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_transformer(var_name, len(self.values))


def get_parameters(args, model, assigner, tower):
    filter_parameters = []
    skip = set()
    if tower == "visual":
        lr = args.visual_lr if args.visual_lr is not None else args.learning_rate
        weight_decay = args.visual_wd if args.visual_wd is not None else args.weight_decay
        filter_parameters = [[name, param] for name, param in model.named_parameters() if "visual." in name]
        if hasattr(model, "visual"):
            if hasattr(model.visual, "no_weight_decay"):
                skip = set.union(skip, model.visual.no_weight_decay())
        skip = [("visual." + n) for n in skip]
    elif tower == "text":
        lr = args.text_lr if args.text_lr is not None else args.learning_rate
        weight_decay = args.text_wd if args.text_wd is not None else args.weight_decay
        filter_parameters = [[name, param] for name, param in model.named_parameters() if "text." in name]
        if hasattr(model, "text"):
            if hasattr(model.text, "no_weight_decay"):
                skip = set.union(skip, model.text.no_weight_decay())
        skip = [("text." + n) for n in skip]
    else:
        lr = args.learning_rate
        weight_decay = args.weight_decay
        exclude = lambda n: "visual." not in n and "text." not in n
        filter_parameters = [[n, p] for n, p in model.named_parameters() if exclude(n)]
        if hasattr(model, "no_weight_decay"):
            skip = set.union(skip, model.no_weight_decay())
    get_num_layer = assigner.get_layer_id if assigner is not None else None
    get_layer_scale = assigner.get_scale if assigner is not None else None
    parameter_group_names = {}
    parameter_group_vars = {}
    wdkey = "weight_decay"
    if args.optimizer == "lamb":
        wdkey = "lamb_weight_decay"
    for name, param in filter_parameters:
        if not not param.stop_gradient:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or name in skip:
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = tower + "_" + "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None
        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.0
            parameter_group_names[group_name] = {
                "group": tower,
                wdkey: this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "learning_rate": lr * scale,
            }
            parameter_group_vars[group_name] = {
                "group": tower,
                wdkey: this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "learning_rate": lr * scale,
            }
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    if is_master(args):
        logging.info(f"Tower = {tower}")
        logging.info(f"Skip weight decay name marked in tower-{tower}: {skip}")
        logging.info(f"Num of parameters group in tower-{tower}: {len(parameter_group_vars.values())}")
        logging.info(f"Param groups = {json.dumps(parameter_group_names, indent=2)}")
    return list(parameter_group_vars.values())


def get_assigner(args, model):
    visual_ld = args.visual_ld if args.visual_ld else args.layer_decay
    text_ld = args.text_ld if args.text_ld else args.layer_decay
    if visual_ld < 1.0:
        visual_num_layers = model.visual.get_num_layers()
        assigner_visual = LayerDecayValueAssigner(
            list(visual_ld ** (visual_num_layers + 1 - i) for i in range(visual_num_layers + 2))
        )
    else:
        assigner_visual = None
    if text_ld < 1.0 and hasattr(model, "text"):
        text_num_layers = model.text.get_num_layers()
        assigner_text = LayerDecayValueAssigner(
            list(text_ld ** (text_num_layers + 1 - i) for i in range(text_num_layers + 2))
        )
    else:
        assigner_text = None
    if assigner_visual is not None:
        logging.info("Assigned visual values = %s" % str(assigner_visual.values))
    if assigner_text is not None:
        logging.info("Assigned text values = %s" % str(assigner_text.values))
    return assigner_visual, assigner_text


def get_all_parameters(args, model):
    assigner_visual, assigner_text = get_assigner(args, model)
    parameters = []
    visual_parameters = get_parameters(args, model, assigner_visual, "visual")
    other_parameters = get_parameters(args, model, None, "other")
    parameters.extend(visual_parameters)
    parameters.extend(other_parameters)
    if hasattr(model, "text"):
        text_parameters = get_parameters(args, model, assigner_text, "text")
        parameters.extend(text_parameters)
    if len(parameters) == 0:
        parameters = model.parameters()
    return parameters


def print_optim(optimizer):
    for param_group in optimizer._param_groups:
        print(param_group["group"], param_group["learning_rate"], param_group["lr_scale"])


def create_optimizer(args, model, lr_scheduler=None, return_params=False):
    optimizer_args = dict(beta1=args.adam_beta1, beta2=args.adam_beta2)
    if lr_scheduler is not None:
        learning_rate = lr_scheduler
    else:
        learning_rate = 1.0

    if args.optimizer == "lamb":
        optimizer_args["learning_rate"] = learning_rate
        optimizer_args["lamb_weight_decay"] = args.weight_decay
        base_optimizer = paddle.optimizer.Lamb
    else:
        optimizer_args["learning_rate"] = learning_rate
        base_optimizer = paddle.optimizer.AdamW
    if args.fp16_opt_level == "O2":
        optimizer_args["multi_precision"] = True
    # if args.max_grad_norm:
    #     grad_clip = paddle.nn.ClipGradByGlobalNorm(
    #         clip_norm=args.max_grad_norm)
    #     optimizer_args['grad_clip'] = grad_clip
    parameters = get_all_parameters(args, model)
    optimizer = base_optimizer(parameters=parameters, **optimizer_args)
    if is_master(args):
        print(f"Optimizer: {args.optimizer}")
        print(f"Optimizer config: {optimizer_args}")
    if return_params:
        return optimizer, parameters
    return optimizer


def create_optimizer_simple(args, model, lr_scheduler=None):
    if lr_scheduler is not None:
        learning_rate = lr_scheduler
    else:
        learning_rate = args.learning_rate

    exclude = lambda n, p: p.rank() < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.stop_gradient is False]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.stop_gradient is False]
    optimizer = paddle.optimizer.AdamW(
        parameters=[
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": args.weight_decay},
        ],
        learning_rate=learning_rate,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        epsilon=args.adam_epsilon,
    )
    return optimizer

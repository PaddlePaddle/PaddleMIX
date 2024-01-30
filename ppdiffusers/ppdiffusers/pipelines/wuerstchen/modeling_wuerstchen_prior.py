# Copyright (c) 2023 Dominic Rampas MIT License
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import math
from typing import Dict, Union

import paddle
import paddle.nn as nn
from paddle.distributed.fleet.utils import recompute

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import UNet2DConditionLoadersMixin
from ...models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from ...models.lora import LoRACompatibleConv, LoRACompatibleLinear
from ...models.modeling_utils import ModelMixin
from ...utils import USE_PEFT_BACKEND, recompute_use_reentrant
from .modeling_wuerstchen_common import (
    AttnBlock,
    ResBlock,
    TimestepBlock,
    WuerstchenLayerNorm,
)


class WuerstchenPrior(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    unet_name = "prior"
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, c_in=16, c=1280, c_cond=1024, c_r=64, depth=16, nhead=16, dropout=0.1):
        super().__init__()
        conv_cls = nn.Conv2D if USE_PEFT_BACKEND else LoRACompatibleConv
        linear_cls = nn.Linear if USE_PEFT_BACKEND else LoRACompatibleLinear

        self.c_r = c_r
        self.projection = conv_cls(c_in, c, kernel_size=1)
        self.cond_mapper = nn.Sequential(
            linear_cls(c_cond, c),
            nn.LeakyReLU(0.2),
            linear_cls(c, c),
        )

        self.blocks = nn.LayerList()
        for _ in range(depth):
            self.blocks.append(ResBlock(c, dropout=dropout))
            self.blocks.append(TimestepBlock(c, c_r))
            self.blocks.append(AttnBlock(c, c, nhead, self_attn=True, dropout=dropout))
        norm_elementwise_affine = False
        norm_elementwise_affine_kwargs = {} if norm_elementwise_affine else dict(weight_attr=False, bias_attr=False)
        self.out = nn.Sequential(
            WuerstchenLayerNorm(c, epsilon=1e-6, **norm_elementwise_affine_kwargs),
            conv_cls(c, c_in * 2, kernel_size=1),
        )

        self.gradient_checkpointing = False
        self.set_default_attn_processor()

    @property
    # Copied from ppdiffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: paddle.nn.Layer, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from ppdiffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]], _remove_lora=False
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: paddle.nn.Layer, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor, _remove_lora=_remove_lora)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"), _remove_lora=_remove_lora)

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from ppdiffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor, _remove_lora=True)

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = (paddle.arange(half_dim).cast("float32") * -emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = paddle.concat([emb.sin(), emb.cos()], axis=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = paddle.concat(emb, paddle.zeros([emb.shape[0], 1]), axis=-1)
        return emb.cast(dtype=r.dtype)

    def forward(self, x, r, c):
        x_in = x
        x = self.projection(x)
        c_embed = self.cond_mapper(c)
        r_embed = self.gen_r_embedding(r)

        if self.gradient_checkpointing and not x.stop_gradient:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs = {} if recompute_use_reentrant() else {"use_reentrant": False}
            for block in self.blocks:
                if isinstance(block, AttnBlock):
                    x = recompute(create_custom_forward(block), x, c_embed, **ckpt_kwargs)
                elif isinstance(block, TimestepBlock):
                    x = recompute(create_custom_forward(block), x, r_embed, **ckpt_kwargs)
                else:
                    x = recompute(create_custom_forward(block), x, **ckpt_kwargs)
        else:
            for block in self.blocks:
                if isinstance(block, AttnBlock):
                    x = block(x, c_embed)
                elif isinstance(block, TimestepBlock):
                    x = block(x, r_embed)
                else:
                    x = block(x)
        a, b = self.out(x).chunk(2, axis=1)
        return (x_in - a) / ((1 - b).abs() + 1e-5)

# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
from paddle.distributed.fleet.utils import recompute

from ..configuration_utils import ConfigMixin, register_to_config
from ..models.attention_processor import AttentionProcessor
from ..models.modeling_utils import ModelMixin
from ..utils import (
    USE_PEFT_BACKEND,
    logging,
    recompute_use_reentrant,
    scale_lora_layers,
    unscale_lora_layers,
    use_old_recompute,
)
from .controlnet import BaseOutput, ControlNetConditioningEmbedding, zero_module
from .embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    FluxPosEmbed,
)
from .transformer_flux import FluxSingleTransformerBlock, FluxTransformerBlock

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FluxControlNetOutput(BaseOutput):
    controlnet_block_samples: Tuple[paddle.Tensor]
    controlnet_single_block_samples: Tuple[paddle.Tensor]


class FluxControlNetModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: List[int] = [16, 56, 56],
        num_mode: int = None,
        conditioning_embedding_channels: int = None,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # Positional embedding
        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        # Time-text embedding
        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

        # Context embedding
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        # Transformer blocks
        self.transformer_blocks = nn.LayerList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.LayerList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_single_layers)
            ]
        )

        # ControlNet blocks
        self.controlnet_blocks = nn.LayerList(
            [zero_module(nn.Linear(self.inner_dim, self.inner_dim)) for _ in range(len(self.transformer_blocks))]
        )

        self.controlnet_single_blocks = nn.LayerList(
            [
                zero_module(nn.Linear(self.inner_dim, self.inner_dim))
                for _ in range(len(self.single_transformer_blocks))
            ]
        )

        # Union mode handling
        self.union = num_mode is not None
        if self.union:
            self.controlnet_mode_embedder = nn.Embedding(num_mode, self.inner_dim)

        # Input hint block
        if conditioning_embedding_channels is not None:
            self.input_hint_block = ControlNetConditioningEmbedding(
                conditioning_embedding_channels=conditioning_embedding_channels, block_out_channels=(16, 16, 16, 16)
            )
            self.controlnet_x_embedder = nn.Linear(in_channels, self.inner_dim)
        else:
            self.input_hint_block = None
            self.controlnet_x_embedder = zero_module(nn.Linear(in_channels, self.inner_dim))

        self.gradient_checkpointing = False

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        processors = {}

        def fn_recursive_add_processors(name: str, module: nn.Layer, processors: dict):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)
        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        count = len(self.attn_processors.keys())
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(f"Expected {count} processors, got {len(processor)}")

        def fn_recursive_attn_processor(name: str, module: nn.Layer, processor):
            if hasattr(module, "set_processor"):
                if isinstance(processor, dict):
                    module.set_processor(processor.pop(f"{name}.processor"))
                else:
                    module.set_processor(processor)
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    @classmethod
    def from_transformer(
        cls,
        transformer,
        num_layers: int = 4,
        num_single_layers: int = 10,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        load_weights_from_transformer=True,
    ):
        config = dict(transformer.config)
        config["num_layers"] = num_layers
        config["num_single_layers"] = num_single_layers
        config["attention_head_dim"] = attention_head_dim
        config["num_attention_heads"] = num_attention_heads
        controlnet = cls(**config)

        if load_weights_from_transformer:
            controlnet.pos_embed.load_dict(transformer.pos_embed.state_dict())
            controlnet.time_text_embed.load_dict(transformer.time_text_embed.state_dict())
            controlnet.context_embedder.load_dict(transformer.context_embedder.state_dict())
            controlnet.x_embedder.load_dict(transformer.x_embedder.state_dict())
            controlnet.transformer_blocks.load_dict(transformer.transformer_blocks.state_dict(), strict=False)
            controlnet.single_transformer_blocks.load_dict(
                transformer.single_transformer_blocks.state_dict(), strict=False
            )

            controlnet.controlnet_x_embedder = zero_module(controlnet.controlnet_x_embedder)
        return controlnet

    def create_custom_forward(self, module, return_dict=None):
        def custom_forward(*inputs):
            if return_dict is not None:
                return module(*inputs, return_dict=return_dict)
            return module(*inputs)

        return custom_forward

    def forward(
        self,
        hidden_states: paddle.Tensor,
        controlnet_cond: paddle.Tensor,
        controlnet_mode: paddle.Tensor = None,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: paddle.Tensor = None,
        pooled_projections: paddle.Tensor = None,
        timestep: paddle.Tensor = None,
        img_ids: paddle.Tensor = None,
        txt_ids: paddle.Tensor = None,
        guidance: paddle.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[FluxControlNetOutput, Tuple]:

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # Input embedding
        hidden_states = self.x_embedder(hidden_states)
        # Process conditional input
        if self.input_hint_block is not None:
            controlnet_cond = self.input_hint_block(controlnet_cond)

            batch_size, channels, height_pw, width_pw = controlnet_cond.shape
            height = height_pw // self.config.patch_size
            width = width_pw // self.config.patch_size
            controlnet_cond = controlnet_cond.reshape(
                batch_size, channels, height, self.config.patch_size, width, self.config.patch_size
            )
            controlnet_cond = controlnet_cond.permute(0, 2, 4, 1, 3, 5)
            controlnet_cond = controlnet_cond.reshape(batch_size, height * width, -1)

        # Add conditional embedding
        controlnet_x_embed = self.controlnet_x_embedder(controlnet_cond)
        hidden_states += controlnet_x_embed
        # Time embeddings
        timestep = timestep.astype(hidden_states.dtype) * 1000

        if guidance is not None:
            guidance = guidance.astype(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )

        # Context embedding
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d paddle.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d paddle Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d paddle.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d paddle Tensor"
            )
            img_ids = img_ids[0]

        # Union mode handling
        if self.union:
            # union mode
            if controlnet_mode is None:
                raise ValueError("`controlnet_mode` cannot be `None` when applying ControlNet-Union")
            # union mode emb
            controlnet_mode_emb = self.controlnet_mode_embedder(controlnet_mode)

            encoder_hidden_states = paddle.concat([controlnet_mode_emb, encoder_hidden_states], axis=1)

            txt_ids = paddle.concat([txt_ids[:1], txt_ids], axis=0)
        # Positional embedding
        ids = paddle.concat([txt_ids, img_ids], axis=0)

        image_rotary_emb = self.pos_embed(ids)

        # Transformer blocks processing
        block_samples = ()

        for i, block in enumerate(self.transformer_blocks):

            if self.gradient_checkpointing and self.training and not use_old_recompute():
                ckpt_kwargs = {} if recompute_use_reentrant() else {"use_reentrant": False}
                hidden_states = recompute(
                    self.create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

            block_samples += (hidden_states,)

        # Combine encoder states
        hidden_states = paddle.concat([encoder_hidden_states, hidden_states], axis=1)
        # Single transformer blocks processing
        single_block_samples = ()

        for i, block in enumerate(self.single_transformer_blocks):

            if self.gradient_checkpointing and self.training:
                ckpt_kwargs = {} if recompute_use_reentrant() else {"use_reentrant": False}
                hidden_states = recompute(
                    self.create_custom_forward(block), hidden_states, temb, image_rotary_emb, **ckpt_kwargs
                )
            else:
                hidden_states = block(hidden_states=hidden_states, temb=temb, image_rotary_emb=image_rotary_emb)

            single_block_sample = hidden_states[:, encoder_hidden_states.shape[1] :]
            single_block_samples += (single_block_sample,)

        # controlnet block
        controlnet_block_samples = ()

        for i, (block_sample, controlnet_block) in enumerate(zip(block_samples, self.controlnet_blocks)):
            block_sample = controlnet_block(block_sample)

            controlnet_block_samples = controlnet_block_samples + (block_sample,)

        controlnet_single_block_samples = ()

        for i, (single_block_sample, controlnet_block) in enumerate(
            zip(single_block_samples, self.controlnet_single_blocks)
        ):
            single_block_sample = controlnet_block(single_block_sample)

            controlnet_single_block_samples = controlnet_single_block_samples + (single_block_sample,)

        # scaling
        controlnet_block_samples = [sample * conditioning_scale for sample in controlnet_block_samples]
        controlnet_single_block_samples = [sample * conditioning_scale for sample in controlnet_single_block_samples]

        controlnet_block_samples = None if len(controlnet_block_samples) == 0 else controlnet_block_samples
        controlnet_single_block_samples = (
            None if len(controlnet_single_block_samples) == 0 else controlnet_single_block_samples
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (controlnet_block_samples, controlnet_single_block_samples)

        return FluxControlNetOutput(
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
        )


class FluxMultiControlNetModel(ModelMixin):
    r"""
    `FluxMultiControlNetModel` wrapper class for Multi-FluxControlNetModel

    This module is a wrapper for multiple instances of the `FluxControlNetModel`. The `forward()` API is designed to be
    compatible with `FluxControlNetModel`.

    Args:
        controlnets (`List[FluxControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. You must set multiple
            `FluxControlNetModel` as a list.
    """

    def __init__(self, controlnets):
        super().__init__()
        self.nets = nn.LayerList(controlnets)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        controlnet_cond: List[paddle.Tensor],
        controlnet_mode: List[paddle.Tensor],
        conditioning_scale: List[float],
        encoder_hidden_states: paddle.Tensor = None,
        pooled_projections: paddle.Tensor = None,
        timestep: paddle.Tensor = None,
        img_ids: paddle.Tensor = None,
        txt_ids: paddle.Tensor = None,
        guidance: paddle.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[FluxControlNetOutput, Tuple]:
        if len(self.nets) == 1 and self.nets[0].union:
            controlnet = self.nets[0]

            for i, (image, mode, scale) in enumerate(zip(controlnet_cond, controlnet_mode, conditioning_scale)):
                block_samples, single_block_samples = controlnet(
                    hidden_states=hidden_states,
                    controlnet_cond=image,
                    controlnet_mode=mode[:, None],
                    conditioning_scale=scale,
                    timestep=timestep,
                    guidance=guidance,
                    pooled_projections=pooled_projections,
                    encoder_hidden_states=encoder_hidden_states,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=return_dict,
                )

                # merge samples
                if i == 0:
                    control_block_samples = block_samples
                    control_single_block_samples = single_block_samples
                else:
                    control_block_samples = [
                        control_block_sample + block_sample
                        for control_block_sample, block_sample in zip(control_block_samples, block_samples)
                    ]

                    control_single_block_samples = [
                        control_single_block_sample + block_sample
                        for control_single_block_sample, block_sample in zip(
                            control_single_block_samples, single_block_samples
                        )
                    ]

        # Regular Multi-ControlNets
        # load all ControlNets into memories
        else:
            for i, (image, mode, scale, controlnet) in enumerate(
                zip(controlnet_cond, controlnet_mode, conditioning_scale, self.nets)
            ):
                block_samples, single_block_samples = controlnet(
                    hidden_states=hidden_states,
                    controlnet_cond=image,
                    controlnet_mode=mode[:, None],
                    conditioning_scale=scale,
                    timestep=timestep,
                    guidance=guidance,
                    pooled_projections=pooled_projections,
                    encoder_hidden_states=encoder_hidden_states,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=return_dict,
                )

                # merge samples
                if i == 0:
                    control_block_samples = block_samples
                    control_single_block_samples = single_block_samples
                else:
                    if block_samples is not None and control_block_samples is not None:
                        control_block_samples = [
                            control_block_sample + block_sample
                            for control_block_sample, block_sample in zip(control_block_samples, block_samples)
                        ]
                    if single_block_samples is not None and control_single_block_samples is not None:
                        control_single_block_samples = [
                            control_single_block_sample + block_sample
                            for control_single_block_sample, block_sample in zip(
                                control_single_block_samples, single_block_samples
                            )
                        ]

        return control_block_samples, control_single_block_samples

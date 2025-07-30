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


from typing import Any, Dict, Optional, Union

import paddle
from cache_functions import cache_init_hunyuan, cal_type

from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput

# from ..loaders import FromOriginalModelMixin, PeftAdapterMixin
from ppdiffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

logger = logging.get_logger(__name__)


def taylorseer_hunyuan_forward(
    self,
    hidden_states: paddle.Tensor,
    timestep: paddle.Tensor,
    encoder_hidden_states: paddle.Tensor,
    encoder_attention_mask: paddle.Tensor,
    pooled_projections: paddle.Tensor,
    guidance: paddle.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[paddle.Tensor, Dict[str, paddle.Tensor]]:
    if attention_kwargs is None:
        attention_kwargs = {}
    if attention_kwargs.get("cache_dic", None) is None:
        attention_kwargs["cache_dic"], attention_kwargs["current"] = cache_init_hunyuan(self)

    cal_type(attention_kwargs["cache_dic"], attention_kwargs["current"])

    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    elif attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
        logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

    batch_size, num_channels, num_frames, height, width = tuple(hidden_states.shape)
    p, p_t = self.config.patch_size, self.config.patch_size_t
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p

    # 1. RoPE
    image_rotary_emb = self.rope(hidden_states)

    # 2. Conditional embeddings
    temb = self.time_text_embed(timestep, guidance, pooled_projections)
    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

    # 3. Attention mask preparation
    latent_sequence_length = tuple(hidden_states.shape)[1]
    condition_sequence_length = tuple(encoder_hidden_states.shape)[1]
    sequence_length = latent_sequence_length + condition_sequence_length
    attention_mask = paddle.zeros(shape=[batch_size, sequence_length]).to(paddle.bool)  # [B, N]

    effective_condition_sequence_length = encoder_attention_mask.sum(axis=1, dtype="int32")  # [B,]
    effective_sequence_length = latent_sequence_length + effective_condition_sequence_length
    for i in range(batch_size):
        attention_mask[(i), : effective_sequence_length[i]] = 1
    # [B, 1, 1, N], for broadcasting across attention heads
    attention_mask = attention_mask.unsqueeze(axis=1).unsqueeze(axis=1)

    # 4. Transformer blocks
    if self.training and self.gradient_checkpointing:

        def create_custom_forward(module, return_dict=None):
            def custom_forward(*inputs):
                if return_dict is not None:
                    return module(*inputs, return_dict=return_dict)
                else:
                    return module(*inputs)

            return custom_forward

        ckpt_kwargs = {}

        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = paddle.distributed.fleet.utils.recompute(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                **ckpt_kwargs,
            )
            hidden_states, encoder_hidden_states = paddle.distributed.fleet.utils.recompute(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                **ckpt_kwargs,
            )

        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = paddle.distributed.fleet.utils.recompute(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                **ckpt_kwargs,
            )

    else:
        attention_kwargs["current"]["stream"] = "double_stream"
        for index_block, block in enumerate(self.transformer_blocks):
            attention_kwargs["current"]["layer"] = index_block
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb, attention_kwargs
            )
        attention_kwargs["current"]["stream"] = "single_stream"
        for index_block, block in enumerate(self.single_transformer_blocks):
            attention_kwargs["current"]["layer"] = index_block
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb, attention_kwargs
            )

    # 5. Output projection
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        [batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p]
    )
    hidden_states = hidden_states.transpose(perm=[0, 4, 1, 5, 2, 6, 3, 7])
    hidden_states = (
        hidden_states.flatten(start_axis=6, stop_axis=7)
        .flatten(start_axis=4, stop_axis=5)
        .flatten(start_axis=2, stop_axis=3)
    )

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)
    attention_kwargs["current"]["step"] += 1
    if not return_dict:
        return (hidden_states,)

    return Transformer2DModelOutput(sample=hidden_states)

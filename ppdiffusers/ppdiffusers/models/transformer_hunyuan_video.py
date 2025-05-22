# Copyright 2024 The Hunyuan Team and The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, List, Optional, Tuple, Union

import inspect
import math
import paddle
import paddle.nn.functional as F
# from paddle_utils import *

from ..configuration_utils import ConfigMixin, register_to_config
# from ..loaders import FromOriginalModelMixin, PeftAdapterMixin
from ..utils import logging, USE_PEFT_BACKEND
from ..utils.paddle_utils import maybe_allow_in_graph, dim2perm
from .attention import FeedForward
from .attention_processor import Attention, AttentionProcessor
from .cache_utils import CacheMixin
from .embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    get_1d_rotary_pos_embed,
)
from .modeling_outputs import Transformer2DModelOutput
from .modeling_utils import ModelMixin
from .normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class HunyuanVideoAttnProcessor2_0:
    def __init__(self):
        if not hasattr(paddle.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                "HunyuanVideoAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        image_rotary_emb: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = paddle.concat(x=[hidden_states, encoder_hidden_states], axis=1)


        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(axis=2, shape=(attn.heads, -1))
        query = query.transpose(perm=dim2perm(query.ndim,1,2))

        key = key.unflatten(axis=2, shape=(attn.heads, -1))
        key = key.transpose(perm=dim2perm(key.ndim,1,2))

        value = value.unflatten(axis=2, shape=(attn.heads, -1))
        value = value.transpose(perm=dim2perm(value.ndim,1,2))

        # 2. QK normalization
        if attn.norm_q is not None:
            if 'begin_norm_axis' in inspect.signature(attn.norm_q.forward).parameters:
                query = attn.norm_q(query, begin_norm_axis=len(query.shape)-1)
            else:
                query = attn.norm_q(query)
        if attn.norm_k is not None:
            if 'begin_norm_axis' in inspect.signature(attn.norm_k.forward).parameters:
                key = attn.norm_k(key, begin_norm_axis=len(query.shape)-1)
            else:
                key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            from .embeddings import apply_rotary_emb

            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = paddle.concat(
                    x=[
                        apply_rotary_emb(query[:, :, : -tuple(encoder_hidden_states.shape)[1]], image_rotary_emb),
                        query[:, :, -tuple(encoder_hidden_states.shape)[1] :],
                    ],
                    axis=2,
                )
                key = paddle.concat(
                    x=[
                        apply_rotary_emb(key[:, :, : -tuple(encoder_hidden_states.shape)[1]], image_rotary_emb),
                        key[:, :, -tuple(encoder_hidden_states.shape)[1] :],
                    ],
                    axis=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(axis=2, shape=(attn.heads, -1))
            encoder_query = encoder_query.transpose(perm=dim2perm(encoder_query.ndim,1,2))

            encoder_key = encoder_key.unflatten(axis=2, shape=(attn.heads, -1))
            encoder_key = encoder_key.transpose(perm=dim2perm(encoder_key.ndim,1,2))

            encoder_value = encoder_value.unflatten(axis=2, shape=(attn.heads, -1))
            encoder_value = encoder_value.transpose(perm=dim2perm(encoder_value.ndim,1,2))

            if attn.norm_added_q is not None:
                if 'begin_norm_axis' in inspect.signature(attn.norm_added_q.forward).parameters:
                    encoder_query = attn.norm_added_q(encoder_query, begin_norm_axis=len(encoder_query.shape)-1)
                else:
                    encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                if 'begin_norm_axis' in inspect.signature(attn.norm_added_k.forward).parameters:
                    encoder_key = attn.norm_added_k(encoder_key, begin_norm_axis=len(encoder_key.shape)-1)
                else:
                    encoder_key = attn.norm_added_k(encoder_key)

            query = paddle.concat(x=[query, encoder_query], axis=2)
            key = paddle.concat(x=[key, encoder_key], axis=2)
            value = paddle.concat(x=[value, encoder_value], axis=2)

        # 5. Attention

        if attention_mask.dtype == paddle.bool:
            L, S = query.shape[-2], key.shape[-2]
            attention_mask_tmp = paddle.zeros([1,1,L, S], dtype=query.dtype)
            attention_mask_tmp = attention_mask_tmp.masked_fill(attention_mask.logical_not(), float("-inf"))
            attention_mask = attention_mask_tmp

        hidden_states = paddle.nn.functional.scaled_dot_product_attention(
            query=query.transpose([0,2,1,3]),
            key=key.transpose([0,2,1,3]),
            value=value.transpose([0,2,1,3]),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose([0,2,1,3])

        hidden_states = hidden_states.transpose(
            perm=dim2perm(hidden_states.ndim, 1, 2)
        ).flatten(start_axis=2, stop_axis=3)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -tuple(encoder_hidden_states.shape)[1]],
                hidden_states[:, -tuple(encoder_hidden_states.shape)[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class HunyuanVideoPatchEmbed(paddle.nn.Layer):
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        patch_size = ((patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size)
        self.proj = paddle.nn.Conv3D(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.flatten(start_axis=2)
        hidden_states = hidden_states.transpose(perm=dim2perm(hidden_states.ndim,1,2))  # BCFHW -> BNC
        return hidden_states


class HunyuanVideoAdaNorm(paddle.nn.Layer):
    def __init__(self, in_features: int, out_features: Optional[int] = None) -> None:
        super().__init__()

        out_features = out_features or 2 * in_features
        self.linear = paddle.nn.Linear(in_features=in_features, out_features=out_features)
        self.nonlinearity = paddle.nn.Silu()

    def forward(
        self, temb: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(chunks=2, axis=1)
        gate_msa, gate_mlp = gate_msa.unsqueeze(axis=1), gate_mlp.unsqueeze(axis=1)
        return gate_msa, gate_mlp


class HunyuanVideoIndividualTokenRefinerBlock(paddle.nn.Layer):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim
        self.norm1 = paddle.nn.LayerNorm(
            normalized_shape=hidden_size,
            weight_attr=True,
            bias_attr=True,
            epsilon=1e-06,
        )
        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
        )
        self.norm2 = paddle.nn.LayerNorm(
            normalized_shape=hidden_size,
            weight_attr=True,
            bias_attr=True,
            epsilon=1e-06,
        )
        self.ff = FeedForward(
            hidden_size,
            mult=mlp_width_ratio,
            activation_fn="linear-silu",
            dropout=mlp_drop_rate,
        )
        self.norm_out = HunyuanVideoAdaNorm(hidden_size, 2 * hidden_size)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        temb: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
        )

        gate_msa, gate_mlp = self.norm_out(temb)
        hidden_states = hidden_states + attn_output * gate_msa

        ff_output = self.ff(self.norm2(hidden_states))
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


class HunyuanVideoIndividualTokenRefiner(paddle.nn.Layer):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        self.refiner_blocks = paddle.nn.LayerList(
            sublayers=[
                HunyuanVideoIndividualTokenRefinerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    attention_bias=attention_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        temb: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> None:
        self_attn_mask = None
        if attention_mask is not None:
            batch_size = tuple(attention_mask.shape)[0]
            seq_len = tuple(attention_mask.shape)[1]
            attention_mask = attention_mask.to(hidden_states.place).to(paddle.bool)
            self_attn_mask_1 = attention_mask.view([batch_size, 1, 1, seq_len]).tile(repeat_times=[1, 1, seq_len, 1])
            self_attn_mask_2 = self_attn_mask_1.transpose(perm=dim2perm(self_attn_mask_1.ndim, 2, 3))
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).to(paddle.bool)
            self_attn_mask[:, :, :, 0] = True

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, self_attn_mask)

        return hidden_states


class HunyuanVideoTokenRefiner(paddle.nn.Layer):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=hidden_size, pooled_projection_dim=in_channels
        )
        self.proj_in = paddle.nn.Linear(in_features=in_channels, out_features=hidden_size, bias_attr=True)
        self.token_refiner = HunyuanVideoIndividualTokenRefiner(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_width_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            attention_bias=attention_bias,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        timestep: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        if attention_mask is None:
            pooled_projections = hidden_states.mean(axis=1)
        else:
            original_dtype = hidden_states.dtype
            mask_float = attention_mask.astype(dtype="float32").unsqueeze(axis=-1)
            pooled_projections = (hidden_states * mask_float).sum(axis=1) / mask_float.sum(axis=1)
            pooled_projections = pooled_projections.to(original_dtype)

        temb = self.time_text_embed(timestep, pooled_projections)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.token_refiner(hidden_states, temb, attention_mask)

        return hidden_states


class HunyuanVideoRotaryPosEmbed(paddle.nn.Layer):
    def __init__(self, patch_size: int, patch_size_t: int, rope_dim: List[int], theta: float = 256.0) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        batch_size, num_channels, num_frames, height, width = tuple(hidden_states.shape)
        rope_sizes = [num_frames // self.patch_size_t, height // self.patch_size, width // self.patch_size]

        axes_grids = []
        for i in range(3):
            # Note: The following line diverges from original behaviour. We create the grid on the device, whereas
            # original implementation creates it on CPU and then moves it to device. This results in numerical
            # differences in layerwise debugging outputs, but visually it is the same.
            grid = paddle.arange(start=0, end=rope_sizes[i], dtype="float32")
            axes_grids.append(grid)
        grid = paddle.meshgrid(*axes_grids)
        grid = paddle.stack(x=grid, axis=0)

        freqs = []
        for i in range(3):
            freq = get_1d_rotary_pos_embed(self.rope_dim[i], grid[i].reshape([-1]), self.theta, use_real=True)
            freqs.append(freq)

        freqs_cos = paddle.concat(x=[f[0] for f in freqs], axis=1)  # (W * H * T, D / 2)
        freqs_sin = paddle.concat(x=[f[1] for f in freqs], axis=1)  # (W * H * T, D / 2)
        return freqs_cos, freqs_sin


class HunyuanVideoSingleTransformerBlock(paddle.nn.Layer):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim
        mlp_dim = int(hidden_size * mlp_ratio)

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            bias=True,
            processor=HunyuanVideoAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-06,
            pre_only=True,
        )

        self.norm = AdaLayerNormZeroSingle(hidden_size, norm_type="layer_norm")
        self.proj_mlp = paddle.nn.Linear(in_features=hidden_size, out_features=mlp_dim)
        self.act_mlp = paddle.nn.GELU(approximate=True)
        self.proj_out = paddle.nn.Linear(in_features=hidden_size + mlp_dim, out_features=hidden_size)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor,
        temb: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        image_rotary_emb: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
    ) -> paddle.Tensor:
        text_seq_length = tuple(encoder_hidden_states.shape)[1]
        hidden_states = paddle.concat(x=[hidden_states, encoder_hidden_states], axis=1)

        residual = hidden_states

        # 1. Input normalization
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        # 2. Attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        attn_output = paddle.concat(x=[attn_output, context_attn_output], axis=1)

        # 3. Modulation and residual connection
        hidden_states = paddle.concat(x=[attn_output, mlp_hidden_states], axis=2)
        hidden_states = gate.unsqueeze(axis=1) * self.proj_out(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states


class HunyuanVideoTransformerBlock(paddle.nn.Layer):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = AdaLayerNormZero(hidden_size)
        self.norm1_context = AdaLayerNormZero(hidden_size)

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            added_kv_proj_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            context_pre_only=False,
            bias=True,
            processor=HunyuanVideoAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-06,
        )

        self.norm2 = paddle.nn.LayerNorm(
            normalized_shape=hidden_size,
            weight_attr=False,
            bias_attr=False,
            epsilon=1e-06,
        )
        self.ff = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")
        self.norm2_context = paddle.nn.LayerNorm(
            normalized_shape=hidden_size,
            weight_attr=False,
            bias_attr=False,
            epsilon=1e-06,
        )
        self.ff_context = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor,
        temb: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        freqs_cis: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # 2. Joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        # 3. Modulation and residual connection
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(axis=1)
        encoder_hidden_states = (encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(axis=1))

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = (norm_hidden_states * (1 + scale_mlp[:, (None)]) + shift_mlp[:, (None)])
        norm_encoder_hidden_states = (norm_encoder_hidden_states * (1 + c_scale_mlp[:, (None)]) + c_shift_mlp[:, (None)])

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(axis=1) * ff_output
        encoder_hidden_states = (encoder_hidden_states + c_gate_mlp.unsqueeze(axis=1) * context_ff_output)

        return hidden_states, encoder_hidden_states


class HunyuanVideoTransformer3DModel(ModelMixin, ConfigMixin, CacheMixin):  # FromOriginalModelMixin, PeftAdapterMixin
    r"""
    A Transformer model for video-like data used in [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo).

    Args:
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        num_attention_heads (`int`, defaults to `24`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        num_layers (`int`, defaults to `20`):
            The number of layers of dual-stream blocks to use.
        num_single_layers (`int`, defaults to `40`):
            The number of layers of single-stream blocks to use.
        num_refiner_layers (`int`, defaults to `2`):
            The number of layers of refiner blocks to use.
        mlp_ratio (`float`, defaults to `4.0`):
            The ratio of the hidden layer size to the input size in the feedforward network.
        patch_size (`int`, defaults to `2`):
            The size of the spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of the tmeporal patches to use in the patch embedding layer.
        qk_norm (`str`, defaults to `rms_norm`):
            The normalization to use for the query and key projections in the attention layers.
        guidance_embeds (`bool`, defaults to `True`):
            Whether to use guidance embeddings in the model.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        pooled_projection_dim (`int`, defaults to `768`):
            The dimension of the pooled projection of the text embeddings.
        rope_theta (`float`, defaults to `256.0`):
            The value of theta to use in the RoPE layer.
        rope_axes_dim (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions of the axes to use in the RoPE layer.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        guidance_embeds: bool = True,
        text_embed_dim: int = 4096,
        pooled_projection_dim: int = 768,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int] = (16, 56, 56),
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Latent and condition embedders
        self.x_embedder = HunyuanVideoPatchEmbed((patch_size_t, patch_size, patch_size), in_channels, inner_dim)
        self.context_embedder = HunyuanVideoTokenRefiner(
            text_embed_dim, num_attention_heads, attention_head_dim, num_layers=num_refiner_layers
        )
        self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(inner_dim, pooled_projection_dim)

        # 2. RoPE
        self.rope = HunyuanVideoRotaryPosEmbed(patch_size, patch_size_t, rope_axes_dim, rope_theta)

        # 3. Dual stream transformer blocks
        self.transformer_blocks = paddle.nn.LayerList(
            sublayers=[
                HunyuanVideoTransformerBlock(
                    num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Single stream transformer blocks
        self.single_transformer_blocks = paddle.nn.LayerList(
            sublayers=[
                HunyuanVideoSingleTransformerBlock(
                    num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                )
                for _ in range(num_single_layers)
            ]
        )

        # 5. Output projection
        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-06)
        self.proj_out = paddle.nn.Linear(
            in_features=inner_dim,
            out_features=patch_size_t * patch_size * patch_size * out_channels,
        )
        self.gradient_checkpointing = False

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: paddle.nn.Layer, processors: Dict[str, AttentionProcessor],):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        """
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
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
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
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        elif (
            attention_kwargs is not None
            and attention_kwargs.get("scale", None) is not None
        ):
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

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
        attention_mask = paddle.zeros(
            shape=[batch_size, sequence_length]).to(paddle.bool) # [B, N]

        effective_condition_sequence_length = encoder_attention_mask.sum(axis=1, dtype="int32")  # [B,]
        effective_sequence_length = (latent_sequence_length + effective_condition_sequence_length)
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
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
                )
            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
                )

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            [batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p]
        )
        hidden_states = hidden_states.transpose(perm=[0, 4, 1, 5, 2, 6, 3, 7])
        hidden_states = (hidden_states.flatten(start_axis=6, stop_axis=7).flatten(start_axis=4, stop_axis=5).flatten(start_axis=2, stop_axis=3))

        if USE_PEFT_BACKEND:
           # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)

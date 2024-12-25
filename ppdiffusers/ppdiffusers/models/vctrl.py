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

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import paddle

from ppdiffusers.configuration_utils import ConfigMixin, register_to_config
from ppdiffusers.models.attention import FeedForward
from ppdiffusers.models.attention_processor import Attention
from ppdiffusers.models.embeddings import TimestepEmbedding, Timesteps, apply_rotary_emb
from ppdiffusers.models.modeling_utils import ModelMixin
from ppdiffusers.utils import BaseOutput


@dataclass
class VCtrlModelOutput(BaseOutput):
    vctrl_block_samples: Tuple[paddle.Tensor]


class VCtrlPatchEmbed(paddle.nn.Layer):
    def __init__(self, patch_size: int = 2, in_channels: int = 16, embed_dim: int = 1920, bias: bool = True) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias_attr=bias,
        )

    def forward(self, image_embeds: paddle.Tensor):
        """
        Args:
            image_embeds (`paddle.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        batch, num_frames, channels, height, width = tuple(image_embeds.shape)
        image_embeds = image_embeds.reshape([-1, channels, height, width])
        image_embeds = self.proj(image_embeds)
        image_embeds = image_embeds.reshape([batch, num_frames, *tuple(image_embeds.shape)[1:]])

        image_embeds = image_embeds.flatten(start_axis=3)
        perm = list(range(len(image_embeds.shape)))
        perm = [perm[0], perm[1], perm[3], perm[2]]
        image_embeds = image_embeds.transpose(perm)
        image_embeds = image_embeds.flatten(start_axis=1, stop_axis=2)
        return image_embeds


class VCtrlLayerNormZero(paddle.nn.Layer):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-05,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.silu = paddle.nn.Silu()
        self.linear = paddle.nn.Linear(in_features=conditioning_dim, out_features=3 * embedding_dim, bias_attr=bias)
        self.norm = paddle.nn.LayerNorm(
            normalized_shape=embedding_dim, epsilon=eps, weight_attr=elementwise_affine, bias_attr=elementwise_affine
        )

    def forward(self, hidden_states: paddle.Tensor, temb: paddle.Tensor) -> paddle.Tensor:
        shift, scale, gate = self.linear(self.silu(temb)).chunk(chunks=3, axis=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        return hidden_states, gate[:, None, :]


class VCtrlAttnProcessor2_0:
    """
    Processor for implementing scaled dot-product attention for the VCtrl model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(paddle.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("VCtrlAttnProcessor2_0 requires Paddle >2.5.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        image_rotary_emb: Optional[paddle.Tensor] = None,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        batch_size, sequence_length, _ = tuple(hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.reshape([batch_size, attn.heads, -1, tuple(attention_mask.shape)[-1]])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = tuple(key.shape)[-1]
        head_dim = inner_dim // attn.heads

        query = query.reshape([batch_size, -1, attn.heads, head_dim])
        perm = list(range(len(query.shape)))
        perm = [perm[0], perm[2], perm[1], perm[3]]
        query = query.transpose(perm)

        key = key.reshape([batch_size, -1, attn.heads, head_dim])
        perm = list(range(len(key.shape)))
        perm = [perm[0], perm[2], perm[1], perm[3]]
        key = key.transpose(perm)

        value = value.reshape([batch_size, -1, attn.heads, head_dim])
        perm = list(range(len(value.shape)))
        perm = [perm[0], perm[2], perm[1], perm[3]]
        value = value.transpose(perm)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # NOTE: There is diff between paddle's and torch's sdpa
        # paddle needs input: [batch_size, seq_len, num_heads, head_dim]
        # torch needs input: [batch_size, num_heads, seq_len, head_dim]

        hidden_states = paddle.nn.functional.scaled_dot_product_attention_(
            query.transpose([0, 2, 1, 3]),
            key.transpose([0, 2, 1, 3]),
            value.transpose([0, 2, 1, 3]),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states = hidden_states.reshape([batch_size, -1, attn.heads * head_dim])
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class VCtrlBlock(paddle.nn.Layer):
    """
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-05,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = VCtrlLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-06,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=VCtrlAttnProcessor2_0(),
        )
        self.norm2 = VCtrlLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        temb: paddle.Tensor,
        image_rotary_emb: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
    ) -> paddle.Tensor:

        norm_hidden_states, gate_msa = self.norm1(hidden_states, temb)

        attn_hidden_states = self.attn1(hidden_states=norm_hidden_states, image_rotary_emb=image_rotary_emb)

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        norm_hidden_states, gate_ff = self.norm2(hidden_states, temb)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_ff * ff_output
        return hidden_states


class VCtrlModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        timestep_activation_fn: str = "silu",
        patch_size: int = 2,
        in_channels: int = 16,
        extra_conditioning_channels: int = 1,
        num_layers: int = 6,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-05,
        use_rotary_positional_embeddings: bool = True,
        image_to_video: bool = False,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)
        self.sample_patch_embed = VCtrlPatchEmbed(
            patch_size, in_channels * 2 if image_to_video else in_channels, inner_dim, bias=True
        )
        self.cond_patch_embed = VCtrlPatchEmbed(
            patch_size, in_channels + extra_conditioning_channels, inner_dim, bias=True
        )
        self.transformer_blocks = paddle.nn.LayerList(
            sublayers=[
                VCtrlBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: paddle.nn.Layer, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)
            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def forward(
        self,
        sample: paddle.Tensor,
        timestep: Union[paddle.Tensor, float, int],
        v_cond: paddle.Tensor,
        v_cond_scale: float = 1.0,
        image_rotary_emb: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        return_dict: bool = True,
    ) -> Union[VCtrlModelOutput, Tuple[Tuple[paddle.Tensor, ...], paddle.Tensor]]:
        dtype = sample.dtype
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        t_emb = t_emb.to(dtype=dtype)
        t_emb = self.time_embedding(t_emb)

        sample = self.sample_patch_embed(sample)
        v_cond = self.cond_patch_embed(v_cond)

        mean_latents, std_latents = paddle.mean(x=sample, axis=(1, 2), keepdim=True), paddle.std(
            x=sample.to(dtype="float32"), axis=(1, 2), keepdim=True
        ).to(dtype=dtype)
        mean_control, std_control = paddle.mean(x=v_cond, axis=(1, 2), keepdim=True), paddle.std(
            x=v_cond.to(dtype="float32"), axis=(1, 2), keepdim=True
        ).to(dtype=dtype)

        v_cond = (v_cond - mean_control) * (std_latents / (std_control + 1e-05)) + mean_latents

        hidden_states = sample + v_cond
        hidden_states = hidden_states.to(dtype=dtype)

        features = []
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                hidden_states = paddle.distributed.fleet.utils.recompute(
                    create_custom_forward(block), hidden_states, t_emb, image_rotary_emb, **ckpt_kwargs
                )
            else:
                hidden_states = block(hidden_states=hidden_states, temb=t_emb, image_rotary_emb=image_rotary_emb)

            features.append(hidden_states)
        features = [(feature * v_cond_scale) for feature in features]

        if not return_dict:
            return features
        return VCtrlModelOutput(vctrl_block_samples=features)

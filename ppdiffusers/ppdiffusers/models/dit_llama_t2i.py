# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.functional.flash_attention import (
    flash_attention,
    scaled_dot_product_attention,
)

from ..configuration_utils import ConfigMixin, register_to_config
from .dit_llama import FeedForward, FinalLayer, TimestepEmbedder, TypePromote, modulate
from .modeling_utils import ModelMixin
from .transformer_2d import Transformer2DModelOutput


class Attention(nn.Layer):
    def __init__(self, dim, n_heads, n_kv_heads, qk_norm=True, fused_attn=True, y_dim=0):
        """
        Initialize the Attention module.

        Args:
            dim (int): Number of input dimensions.
            n_heads (int): Number of heads.
            n_kv_heads (Optional[int]): Number of kv heads, if using GQA.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (nn.Linear): Linear transformation for queries.
            wk (nn.Linear): Linear transformation for keys.
            wv (nn.Linear): Linear transformation for values.
            wo (nn.Linear): Linear transformation for output.
            cache_k (paddle.Tensor): Cached keys for attention.
            cache_v (paddle.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_local_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias_attr=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias_attr=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias_attr=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias_attr=False)

        if y_dim > 0:
            self.wk_y = nn.Linear(y_dim, self.n_kv_heads * self.head_dim, bias_attr=False)
            self.wv_y = nn.Linear(y_dim, self.n_kv_heads * self.head_dim, bias_attr=False)
            self.gate = nn.Parameter(paddle.zeros([self.n_local_heads]))

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.n_local_heads * self.head_dim)
            self.k_norm = nn.LayerNorm(self.n_local_kv_heads * self.head_dim)
            if y_dim > 0:
                self.ky_norm = nn.LayerNorm(self.n_local_kv_heads * self.head_dim)
            else:
                self.ky_norm = nn.Identity()
        else:
            self.q_norm = self.k_norm = nn.Identity()
            self.ky_norm = nn.Identity()

        self.fused_attn = fused_attn
        self.scale = self.head_dim**-0.5

    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        """
        Reshape frequency tensor for broadcasting it with another tensor.

        This function reshapes the frequency tensor to have the same shape as
        the target tensor 'x' for the purpose of broadcasting the frequency
        tensor during element-wise operations.

        Args:
            freqs_cis (paddle.Tensor): Frequency tensor to be reshaped.
            x (paddle.Tensor): Target tensor for broadcasting compatibility.

        Returns:
            paddle.Tensor: Reshaped frequency tensor.

        Raises:
            AssertionError: If the frequency tensor doesn't match the expected
                shape.
            AssertionError: If the target tensor 'x' doesn't have the expected
                number of dimensions.
        """
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert tuple(freqs_cis.shape) == (tuple(x.shape)[1], tuple(x.shape)[-1])
        shape = [(d if i == 1 or i == ndim - 1 else 1) for i, d in enumerate(tuple(x.shape))]
        return freqs_cis.reshape([*shape])

    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        """
        Apply rotary embeddings to input tensors using the given frequency
        tensor.

        This function applies rotary embeddings to the given query 'xq' and
        key 'xk' tensors using the provided frequency tensor 'freqs_cis'. The
        input tensors are reshaped as complex numbers, and the frequency tensor
        is reshaped for broadcasting compatibility. The resulting tensors
        contain rotary embeddings and are returned as real tensors.

        Args:
            xq (paddle.Tensor): Query tensor to apply rotary embeddings.
            xk (paddle.Tensor): Key tensor to apply rotary embeddings.
            freqs_cis (paddle.Tensor): Precomputed frequency tensor for complex
                exponentials.

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: Tuple of modified query tensor
                and key tensor with rotary embeddings.
        """
        with paddle.amp.auto_cast(enable=False):
            xq_ = paddle.as_complex(xq.cast("float32").reshape([*tuple(xq.shape)[:-1], -1, 2]))
            xk_ = paddle.as_complex(xk.cast("float32").reshape([*tuple(xk.shape)[:-1], -1, 2]))
            freqs_cis = Attention.reshape_for_broadcast(freqs_cis, xq_)
            xq_out = paddle.as_real(xq_ * freqs_cis).flatten(start_axis=3)
            xk_out = paddle.as_real(xk_ * freqs_cis).flatten(start_axis=3)
            return xq_out.cast(xq.dtype), xk_out.cast(xk.dtype)

    def forward(self, x, freqs_cis, y, y_mask):
        """
        Forward pass of the attention module.

        Args:
            x (paddle.Tensor): Input tensor.
            freqs_cis (paddle.Tensor): Precomputed frequency tensor.

        Returns:
            paddle.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = tuple(x.shape)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.reshape([bsz, seqlen, self.n_local_heads, self.head_dim])
        xk = xk.reshape([bsz, seqlen, self.n_local_kv_heads, self.head_dim])
        xv = xv.reshape([bsz, seqlen, self.n_local_kv_heads, self.head_dim])

        xq, xk = Attention.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq, xk = xq.cast(dtype), xk.cast(dtype)

        n_rep = self.n_local_heads // self.n_local_kv_heads

        if dtype in [paddle.float16, paddle.bfloat16]:
            output, _ = flash_attention(
                xq,
                xk,
                xv,
                dropout=0.0,
                causal=False,
                return_softmax=False,
            )
        else:
            if n_rep > 1:
                xk = xk.unsqueeze(axis=3).tile([1, 1, 1, n_rep, 1]).flatten(start_axis=2, stop_axis=3)
                xv = xv.unsqueeze(axis=3).tile([1, 1, 1, n_rep, 1]).flatten(start_axis=2, stop_axis=3)
            if self.fused_attn:
                output = F.scaled_dot_product_attention_(
                    xq,
                    xk,
                    xv,
                    dropout_p=0.0,
                    is_causal=False,
                )
            else:
                q = xq.transpose([0, 2, 1, 3]) * self.scale
                attn = q @ xk.transpose([0, 2, 1, 3]).transpose([0, 1, 3, 2])
                attn = F.softmax(attn, axis=-1)
                output = attn @ xv.transpose([0, 2, 1, 3])
                output = output.transpose([0, 2, 1, 3])

        output = output.flatten(start_axis=-2)

        if hasattr(self, "wk_y"):
            yk = self.ky_norm(self.wk_y(y)).reshape([bsz, -1, self.n_local_kv_heads, self.head_dim])
            yv = self.wv_y(y).reshape([bsz, -1, self.n_local_kv_heads, self.head_dim])
            n_rep = self.n_local_heads // self.n_local_kv_heads

            y_mask = y_mask.reshape([bsz, 1, 1, -1]).expand([bsz, self.n_local_heads, seqlen, -1])

            if dtype in [paddle.float16, paddle.bfloat16]:
                output_y = scaled_dot_product_attention(
                    xq,
                    yk,
                    yv,
                    attn_mask=y_mask.cast(dtype),  # no need to transpose
                )
            else:
                if n_rep > 1:
                    yk = yk.unsqueeze(3).tile([1, 1, 1, n_rep, 1]).flatten(2, 3)
                    yv = yv.unsqueeze(3).tile([1, 1, 1, n_rep, 1]).flatten(2, 3)

                output_y = F.scaled_dot_product_attention_(
                    xq,
                    yk,
                    yv,
                    attn_mask=y_mask,
                )

            output_y = output_y * self.gate.tanh().reshape([1, 1, -1, 1])
            output_y = output_y.flatten(-2)
            output = output + output_y

        return self.wo(output)


class TransformerBlock(nn.Layer):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        multiple_of: int,
        mlp_ratio: float,
        ffn_dim_multiplier: float,
        norm_eps: float,
        qk_norm: bool,
        fused_attn: bool,
        y_dim: int,
    ) -> None:
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            dim (int): Embedding dimension of the input features.
            n_heads (int): Number of attention heads.
            n_kv_heads (Optional[int]): Number of attention heads in key and
                value features (if using GQA), or set to None for the same as
                query.
            multiple_of (int): Value to ensure hidden dimension is a multiple
                of this value in the FeedForward block.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden
                dimension in the FeedForward block. Defaults to None.
            norm_eps (float): A small value added to the norm layer
                denominators to avoid division-by-zero.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.
            adaLN_modulation (nn.Sequential): A small network to generate
                feature modulation factors.

        """
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm, fused_attn, y_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.feed_forward = FeedForward(
            dim=dim, hidden_dim=mlp_hidden_dim, multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier
        )
        self.layer_id = layer_id
        self.attention_norm = nn.LayerNorm(dim, epsilon=norm_eps, bias_attr=False)
        self.ffn_norm = nn.LayerNorm(dim, epsilon=norm_eps, bias_attr=False)

        self.adaLN_modulation = nn.Sequential(
            nn.Silu(),
            nn.Linear(min(dim, 1024), 6 * dim),
        )
        self.attention_y_norm = nn.LayerNorm(y_dim, epsilon=norm_eps, bias_attr=False)

    def forward(self, x, y, y_mask, freqs_cis, adaln_input=None):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (paddle.Tensor): Input tensor.
            freqs_cis (paddle.Tensor): Precomputed cosine and sine frequencies.
            mask (paddle.Tensor, optional): Masking tensor for attention.
                Defaults to None.

        Returns:
            paddle.Tensor: Output tensor after applying attention and
                feedforward layers.

        """
        y = y.cast(x.dtype)
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(
                6, axis=1
            )
            h = x + gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa), freqs_cis, self.attention_y_norm(y), y_mask
            )
            out = h + gate_mlp.unsqueeze(1) * self.feed_forward(modulate(self.ffn_norm(h), shift_mlp, scale_mlp))
        else:
            h = x + self.attention(self.attention_norm(x), freqs_cis, self.attention_y_norm(y), y_mask)
            out = h + self.feed_forward(self.ffn_norm(h))
        return out


class DiTLLaMAT2IModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    _use_memory_efficient_attention_xformers = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        out_channels: int = 8,
        max_seq_len: int = 4224,
        num_layers: int = 32,
        num_attention_heads: int = 16,
        attention_head_dim: int = 96,
        mlp_ratio: float = 4.0,
        n_kv_heads=None,
        multiple_of: int = 256,
        ffn_dim_multiplier=None,
        norm_eps: float = 1e-05,
        learn_sigma: bool = True,
        qk_norm: bool = True,
        cap_feat_dim: int = 4096,
        rope_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        dim = attention_head_dim * num_attention_heads

        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps
        self.learn_sigma = learn_sigma
        self.qk_norm = qk_norm

        self.gradient_checkpointing = True
        self.fused_attn = True

        self.x_embedder = nn.Linear(in_channels * patch_size**2, dim)
        self.t_embedder = TimestepEmbedder(min(dim, 1024))
        self.cap_embedder = nn.Sequential(
            nn.LayerNorm(cap_feat_dim),
            nn.Linear(cap_feat_dim, min(dim, 1024)),
        )

        # 2. Define transformers blocks
        self.layers = nn.LayerList(
            [
                TransformerBlock(
                    layer_id=idx,
                    dim=dim,
                    n_heads=num_attention_heads,
                    n_kv_heads=n_kv_heads,
                    multiple_of=multiple_of,
                    mlp_ratio=mlp_ratio,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    fused_attn=self.fused_attn,
                    y_dim=cap_feat_dim,
                )
                for idx in range(num_layers)
            ]
        )

        # 3. Define output layers
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)
        self.freqs_cis = self.precompute_freqs_cis(
            dim // num_attention_heads, max_seq_len, rope_scaling_factor=rope_scaling_factor
        )
        self.eol_token = self.create_parameter(shape=[dim])
        self.pad_token = self.create_parameter(shape=[dim])

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def enable_gradient_checkpointing(self, value=True):
        self.gradient_checkpointing = value

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[str] = None):
        self._use_memory_efficient_attention_xformers = True
        self.fused_attn = True

    def unpatchify(self, x, img_size, return_tensor=False):
        """
        Args:
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        pH = pW = self.patch_size
        if return_tensor:
            H, W = img_size[0]
            B = x.shape[0]
            L = (H // pH) * (W // pW + 1)  # one additional for eol
            x = x[:, :L].reshape([B, H // pH, W // pW + 1, pH, pW, self.out_channels])
            x = x[:, :, :-1]
            x = x.transpose([0, 5, 1, 3, 2, 4]).flatten(4, 5).flatten(2, 3)
            return x
        else:
            imgs = []
            for i in range(x.shape[0]):
                H, W = img_size[i]
                L = (H // pH) * (W // pW + 1)
                imgs.append(
                    x[i][:L]
                    .reshape([H // pH, W // pW + 1, pH, pW, self.out_channels])[:, :-1, :, :, :]
                    .transpose([4, 0, 2, 1, 3])
                    .flatten(3, 4)
                    .flatten(1, 2)
                )
        return imgs

    def patchify_and_embed(self, x):
        if isinstance(x, paddle.Tensor):
            pH = pW = self.patch_size
            B, C, H, W = x.shape[:]
            x = x.reshape([B, C, H // pH, pH, W // pW, pW]).transpose([0, 2, 4, 1, 3, 5]).flatten(3)
            x = self.x_embedder(x)

            x = paddle.concat(
                [
                    x,
                    self.eol_token.reshape([1, 1, 1, -1]).expand([B, H // pH, 1, -1]),
                ],
                axis=2,
            )
            x = x.flatten(1, 2)

            if x.shape[1] < self.max_seq_len:
                x = paddle.concat(
                    [
                        x,
                        self.pad_token.reshape([1, 1, -1]).expand([B, self.max_seq_len - x.shape[1], -1]),
                    ],
                    axis=1,
                )
            return x, [(H, W)] * B
        else:
            pH = pW = self.patch_size
            x_embed = []
            img_size = []
            for img in x:
                C, H, W = img.shape[:]
                img_size.append((H, W))
                img = img.reshape([C, H // pH, pH, W // pW, pW]).transpose([1, 3, 0, 2, 4]).flatten(2)
                img = self.x_embedder(img)
                img = paddle.concat(
                    [
                        img,
                        self.eol_token.reshape([1, 1, -1]).expand([H // pH, 1, -1]),
                    ],
                    axis=1,
                )
                img = img.flatten(0, 1)
                if img.shape[0] < self.max_seq_len:
                    img = paddle.concat(
                        [
                            img,
                            self.pad_token.reshape([1, -1]).expand([self.max_seq_len - img.shape[0], -1]),
                        ],
                        axis=0,
                    )
                x_embed.append(img)
            x_embed = paddle.stack(x_embed, axis=0)
            return x_embed, img_size

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, rope_scaling_factor: float = 1.0):

        """
        Precompute the frequency tensor for complex exponentials (cis) with
        given dimensions.

        This function calculates a frequency tensor with complex exponentials
        using the given dimension 'dim' and the end index 'end'. The 'theta'
        parameter scales the frequencies. The returned tensor contains complex
        values in complex64 data type.

        Args:
            dim (int): Dimension of the frequency tensor.
            end (int): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation.
                Defaults to 10000.0.

        Returns:
            paddle.Tensor: Precomputed frequency tensor with complex
                exponentials.
        """
        freqs = 1.0 / theta ** (paddle.arange(start=0, end=dim, step=2)[: dim // 2].cast("float32") / dim)
        t = paddle.arange(end=end, dtype=paddle.float32)
        t = t / rope_scaling_factor
        input_0, vec2_0 = TypePromote(t, freqs)
        freqs = paddle.outer(input_0, vec2_0).cast("float32")
        freqs_cis = paddle.complex(
            paddle.ones_like(freqs) * paddle.cos(freqs), paddle.ones_like(freqs) * paddle.sin(freqs)
        )
        return freqs_cis

    def forward(
        self,
        hidden_states: paddle.Tensor,
        timestep: paddle.Tensor,
        cap_feats: paddle.Tensor,
        cap_mask: paddle.Tensor,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states: (N, C, H, W) tensor of spatial inputs (images or latent
                representations of images)
            timestep: (N,) tensor of diffusion timesteps
            class_labels: (N,) tensor of class labels
        """
        hidden_states = hidden_states.cast(self.dtype)
        timestep = timestep.cast(self.dtype)

        # 1. Input
        x_is_tensor = isinstance(hidden_states, paddle.Tensor)
        hidden_states, img_size = self.patchify_and_embed(hidden_states)

        t = self.t_embedder(timestep).cast(self.dtype)
        cap_mask_float = cap_mask.cast("float32").unsqueeze(-1)
        cap_feats_pool = (cap_feats * cap_mask_float).sum(axis=1) / cap_mask_float.sum(axis=1)
        cap_emb = self.cap_embedder(cap_feats_pool.cast(self.dtype))
        adaln_input = t + cap_emb

        # 2. Blocks
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing:
                hidden_states = paddle.distributed.fleet.utils.recompute(
                    layer, hidden_states, cap_feats, cap_mask, self.freqs_cis[: hidden_states.shape[1]], adaln_input
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    cap_feats,
                    cap_mask,
                    self.freqs_cis[: hidden_states.shape[1]],
                    adaln_input,
                )

        # 3. Output
        hidden_states = self.final_layer(hidden_states, adaln_input)
        output = self.unpatchify(hidden_states, img_size, return_tensor=x_is_tensor)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

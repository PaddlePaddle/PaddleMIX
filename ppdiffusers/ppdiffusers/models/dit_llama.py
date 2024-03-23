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

import math
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.functional.flash_attention import flash_attention

from ..configuration_utils import ConfigMixin, register_to_config
from .embeddings import LabelEmbedding
from .modeling_utils import ModelMixin
from .transformer_2d import Transformer2DModelOutput


def TypePromote(x, y):
    TYPE_PROMOTE_DICT = {
        "INT16FP16": "float16",
        "INT16FP32": "float32",
        "INT16FP64": "float64",
        "INT32FP16": "float32",
        "INT32FP32": "float32",
        "INT32FP64": "float64",
        "INT64FP16": "float64",
        "INT64FP32": "float64",
        "INT64FP64": "float64",
    }
    if x.dtype.name + y.dtype.name in TYPE_PROMOTE_DICT:
        promote_type = TYPE_PROMOTE_DICT[x.dtype.name + y.dtype.name]
    elif y.dtype.name + x.dtype.name in TYPE_PROMOTE_DICT:
        promote_type = TYPE_PROMOTE_DICT[y.dtype.name + x.dtype.name]
    else:
        return x, y
    return x.cast(promote_type), y.cast(promote_type)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(axis=1)) + shift.unsqueeze(axis=1)


class TimestepEmbedder(nn.Layer):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.Silu(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = paddle.exp(x=-math.log(max_period) * paddle.arange(start=0, end=half, dtype="float32") / half)
        args = t[:, (None)].astype(dtype="float32") * freqs[None]
        embedding = paddle.concat(x=[paddle.cos(x=args), paddle.sin(x=args)], axis=-1)
        if dim % 2:
            embedding = paddle.concat(x=[embedding, paddle.zeros_like(x=embedding[:, :1])], axis=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.cast(self.mlp[0].weight.dtype))
        return t_emb


class Attention(nn.Layer):
    def __init__(self, dim, n_heads, n_kv_heads, qk_norm=True, fused_attn=True):
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

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.n_local_heads * self.head_dim)
            self.k_norm = nn.LayerNorm(self.n_local_kv_heads * self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

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

    def forward(self, x, freqs_cis):
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
            n_rep = self.n_local_heads // self.n_local_kv_heads
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
        return self.wo(output)


class FeedForward(nn.Layer):
    def __init__(self, dim, hidden_dim, multiple_of=256, ffn_dim_multiplier=None):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple
                of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden
                dimension. Defaults to None.

        Attributes:
            w1 (nn.Linear): Linear transformation for the first
                layer.
            w2 (nn.Linear): Linear transformation for the second layer.
            w3 (nn.Linear): Linear transformation for the third
                layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = int(multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of))

        self.w1 = nn.Linear(dim, hidden_dim, bias_attr=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias_attr=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias_attr=False)

    def forward(self, x):
        xw1 = F.silu(self.w1(x))
        xw3 = self.w3(x)
        output = self.w2(xw1 * xw3)
        return output


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
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm, fused_attn)
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

    def forward(self, x, freqs_cis, adaln_input=None):
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
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(
                6, axis=1
            )
            h = x + gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa), freqs_cis
            )
            out = h + gate_mlp.unsqueeze(1) * self.feed_forward(modulate(self.ffn_norm(h), shift_mlp, scale_mlp))
        else:
            h = x + self.attention(self.attention_norm(x), freqs_cis)
            out = h + self.feed_forward(self.ffn_norm(h))
        return out


class FinalLayer(paddle.nn.Layer):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = paddle.nn.LayerNorm(hidden_size, weight_attr=False, bias_attr=False, epsilon=1e-06)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.Silu(), nn.Linear(min(hidden_size, 1024), 2 * hidden_size))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiTLLaMA2DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    _use_memory_efficient_attention_xformers = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 32,  # image_size // 8
        patch_size: int = 2,
        in_channels: int = 4,
        out_channels: int = 8,
        num_layers: int = 32,
        num_attention_heads: int = 16,
        attention_head_dim: int = 96,
        mlp_ratio: float = 4.0,
        n_kv_heads=None,
        multiple_of: int = 256,
        ffn_dim_multiplier=None,
        norm_eps: float = 1e-05,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.sample_size = sample_size
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
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.learn_sigma = learn_sigma
        self.qk_norm = qk_norm

        self.gradient_checkpointing = True
        self.fused_attn = True

        self.x_embedder = nn.Linear(in_channels * patch_size**2, dim)
        self.t_embedder = TimestepEmbedder(min(dim, 1024))
        self.y_embedder = LabelEmbedding(num_classes, min(dim, 1024), class_dropout_prob)

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
                )
                for idx in range(num_layers)
            ]
        )

        # 3. Define output layers
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)
        self.freqs_cis = self.precompute_freqs_cis(dim // num_attention_heads, 4096)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def enable_gradient_checkpointing(self, value=True):
        self.gradient_checkpointing = value

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[str] = None):
        self._use_memory_efficient_attention_xformers = True
        self.fused_attn = True

    def unpatchify(self, x):
        """
        Args:
            x: (N, T, patch_size**2 * C)
            imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(tuple(x.shape)[1] ** 0.5)
        assert h * w == tuple(x.shape)[1]

        x = x.reshape(shape=([tuple(x.shape)[0], h, w, p, p, c]))
        x = paddle.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=([tuple(x.shape)[0], c, h * p, h * p]))
        return imgs

    def patchify(self, x):
        B, C, H, W = tuple(x.shape)
        assert (H, W) == (self.sample_size, self.sample_size)
        pH = pW = self.patch_size
        x = x.reshape([B, C, H // pH, pH, W // pW, pW])
        x = x.transpose([0, 2, 4, 1, 3, 5]).flatten(start_axis=-3).flatten(start_axis=1, stop_axis=2)
        return x

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
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
        t = paddle.arange(end=end)
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
        class_labels: paddle.Tensor,
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
        hidden_states = self.patchify(hidden_states)
        x = self.x_embedder(hidden_states)
        t = self.t_embedder(timestep)
        y = self.y_embedder(class_labels)
        adaln_input = t + y

        # 2. Blocks
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing:
                x = paddle.distributed.fleet.utils.recompute(layer, x, self.freqs_cis[: x.shape[1]], adaln_input)
            else:
                x = layer(
                    x,
                    self.freqs_cis[: x.shape[1]],
                    adaln_input,
                )

        # 3. Output
        hidden_states = self.final_layer(x, adaln_input)
        output = self.unpatchify(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

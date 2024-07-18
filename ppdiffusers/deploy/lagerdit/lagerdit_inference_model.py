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
from dataclasses import dataclass
from typing import Any, Dict, Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute
from paddle.nn.functional.flash_attention import flash_attention

from ppdiffusers.configuration_utils import ConfigMixin, register_to_config
from ppdiffusers.utils import recompute_use_reentrant, use_old_recompute
from ppdiffusers.models.embeddings import LabelEmbedding
from ppdiffusers.models.modeling_utils import ModelMixin
from ppdiffusers.models.transformer_2d import Transformer2DModelOutput
from paddle.nn.initializer import Constant
from paddle.nn.functional.flash_attention import flash_attn_unpadded
from paddle.framework import LayerHelper, in_dynamic_mode

from paddle.incubate.nn.functional import (
    fused_layer_norm,
    fused_rms_norm,
)
from paddle.incubate.nn.functional import fused_linear


def fused_act_bias_wrapper(
    x,
    bias=None,
    dequant_scales=None,
    shift=None,
    smooth=None,
    act_method="gelu",
    compute_dtype="default",
    quant_scale=-1,
    quant_round_type=0,
    quant_max_bound=0,
    quant_min_bound=0,
):
    if in_dynamic_mode():
        return paddle._C_ops.fused_bias_act(
            x,
            bias,
            dequant_scales,
            shift,
            smooth,
            act_method,
            compute_dtype,
            quant_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )
    helper = LayerHelper("fused_bias_act")
    if x.dtype == "int32":
        if compute_dtype == "bf16":
            dtype = "uint16"
        elif compute_dtype == "fp16":
            dtype = "float16"
        elif compute_dtype == "fp32":
            dtype = "float32"
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

    inputs = {}
    inputs["x"] = x
    if bias is not None:
        inputs["bias"] = bias
    if dequant_scales is not None:
        inputs["dequant_scales"] = dequant_scales

    if shift is not None:
        inputs["shift"] = shift

    if smooth is not None:
        inputs["smooth"] = smooth

    attrs = {
        "act_method": act_method,
        "compute_dtype": compute_dtype,
        "quant_scale": quant_scale,
        "quant_round_type": quant_round_type,
        "quant_max_bound": quant_max_bound,
        "quant_min_bound": quant_min_bound,
    }

    helper.append_op(
        type="fused_bias_act",
        inputs=inputs,
        outputs={"out": out},
        attrs=attrs,
    )
    return out


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
    bs, dim = scale.shape
    x = x.reshape([bs,-1,dim])
    out = x * (1 + scale.unsqueeze(axis=1)) + shift.unsqueeze(axis=1)
    return out.reshape([-1,dim])


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

        n_rep = self.n_local_heads // self.n_local_kv_heads
        if n_rep >= 1:
            xk = xk.unsqueeze(axis=3).tile([1, 1, 1, n_rep, 1]).flatten(start_axis=2, stop_axis=3)
            xv = xv.unsqueeze(axis=3).tile([1, 1, 1, n_rep, 1]).flatten(start_axis=2, stop_axis=3)

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

class FinalLayer(paddle.nn.Layer):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = paddle.nn.LayerNorm(hidden_size, weight_attr=False, bias_attr=False, epsilon=1e-06)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.Silu(),
            nn.Linear(min(hidden_size, 1024), 2 * hidden_size)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DitFusedMultiTransformerLayersConfig:
    def __init__(self,
                embed_dim,
                num_heads,
                dim_feedforward,
                weight_only_quant_bits=-1,  # -1 means use Half precision.
                dropout_rate=0.0,
                activation="geglu",
                norm_type="layernorm",
                use_neox_rotary_style=False,
                normalize_before=True,
                ln_scale_attrs=None,
                ln_bias_attrs=None,
                qkv_weight_attrs=None,
                qkv_weight_scale_attrs=None,
                qkv_bias_attrs=None,
                q_norm_weight_attrs=None,
                q_norm_bias_attrs=None,
                k_norm_weight_attrs=None,
                k_norm_bias_attrs=None,
                linear_weight_attrs=None,
                linear_weight_scale_attrs=None,
                linear_bias_attrs=None,
                ffn_ln_scale_attrs=None,
                ffn_ln_bias_attrs=None,
                ffn1_weight_attrs=None,
                ffn1_weight_scale_attrs=None,
                ffn1_bias_attrs=None,
                ffn2_weight_attrs=None,
                ffn2_weight_scale_attrs=None,
                ffn2_bias_attrs=None,
                qkv_out_scale_attrs=None,
                linear_out_scale_attrs=None,
                ffn1_out_scale_attrs=None,
                ffn2_out_scale_attrs=None,
                epsilon=1e-5,
                residual_alpha=1.0,
                num_layers=-1,
                nranks=1,
                trans_qkvw=True,
                ring_id=-1,
                skip=False
                ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.norm_type = norm_type


        self.ln_scale_attrs = ln_scale_attrs
        self.ln_bias_attrs = ln_bias_attrs

        self.qkv_weight_attrs = qkv_weight_attrs
        self.qkv_weight_scale_attrs = qkv_weight_scale_attrs
        self.qkv_bias_attrs = qkv_bias_attrs

        self.q_norm_weight_attrs=q_norm_weight_attrs
        self.q_norm_bias_attrs=q_norm_bias_attrs

        self.k_norm_weight_attrs=k_norm_weight_attrs
        self.k_norm_bias_attrs=k_norm_bias_attrs


        self.linear_weight_attrs = linear_weight_attrs
        self.linear_weight_scale_attrs = linear_weight_scale_attrs
        self.linear_bias_attrs = linear_bias_attrs

        ### ffn

        self.ffn_ln_scale_attrs = ffn_ln_scale_attrs
        self.ffn_ln_bias_attrs = ffn_ln_bias_attrs
        ##### ffn1
        self.ffn1_weight_attrs = ffn1_weight_attrs
        self.ffn1_weight_scale_attrs = ffn1_weight_scale_attrs
        self.ffn1_bias_attrs = ffn1_bias_attrs

        ##### ffn2
        self.ffn2_weight_attrs = ffn2_weight_attrs
        self.ffn2_weight_scale_attrs = ffn2_weight_scale_attrs
        self.ffn2_bias_attrs = ffn2_bias_attrs

        self.qkv_out_scale_attrs = qkv_out_scale_attrs
        self.linear_out_scale_attrs = linear_out_scale_attrs
        self.ffn1_out_scale_attrs = ffn1_out_scale_attrs
        self.ffn2_out_scale_attrs = ffn2_out_scale_attrs


        self.epsilon = epsilon
        self.residual_alpha = residual_alpha
        self.num_layers = num_layers
        self.nranks = nranks
        self.trans_qkvw = trans_qkvw
        self.ring_id = ring_id
        pass
class DiTFusedMultiTransformerLayers(nn.Layer):
    def __init__(self, 
                config: DitFusedMultiTransformerLayersConfig, export = False):
        super().__init__()
        self.num_layers = config.num_layers
        assert config.embed_dim > 0, "Expected embed_dim to be greater than 0, " "but received {}".format(
            config.embed_dim
        )
        assert config.num_heads > 0, "Expected nhead to be greater than 0, " "but received {}".format(config.num_heads)
        assert config.dim_feedforward > 0, "Expected dim_feedforward to be greater than 0, but received {}".format(
            config.dim_feedforward
        )
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.export = export
        self.mlp_hidden_dim = config.dim_feedforward
        self.dim_feedforward = config.dim_feedforward
        # self._dtype = self._helper.get_default_dtype()
        self._dtype = "bfloat16"
        self._norm_weight_dtype = 'float32'
        self.create_params_type = self._dtype
        self.activation = config.activation
        self._epsilon = config.epsilon
        self._ring_id = config.ring_id
        self.nranks = config.nranks
        self.norm_type = config.norm_type
        if self.norm_type == "layernorm":
            self.norm_func = fused_layer_norm
        elif self.norm_type == "rmsnorm":
            self.norm_func = fused_rms_norm
        else:
            raise NotImplementedError(f"Only support norm type of [layernorm, rmsnorm], but got {self.norm_type}")
        self.linear = fused_linear

        

        # prepare parameters


        self.ln_scales, self.ln_biases = [], []
        
        self.qkv_weights, self.qkv_biases = [], []
        
        self.q_norm_weights, self.q_norm_biases = [], []
        self.k_norm_weights, self.k_norm_biases = [], []

        self.linear_weights, self.linear_biases = [], []

        self.ffn_ln_scales, self.ffn_ln_biases = [], []
        self.ffn1_weights, self.ffn1_biases = [], []
        self.ffn2_weights, self.ffn2_biases = [], []


        fmt_blocks_addLN_modulate_linear_weight_attr = paddle.ParamAttr(name="fmt_adaLN_modulate_linear.weight")
        fmt_blocks_addLN_modulate_linear_bias_attr = paddle.ParamAttr(name="fmt_adaLN_modulate_linear.bias")
        self.adaLN_modulate_linear_weight = paddle.create_parameter(
            attr= fmt_blocks_addLN_modulate_linear_weight_attr,
            default_initializer=Constant(value=1.0),
            shape=[min(config.embed_dim, 1024), 6*config.embed_dim * self.num_layers],
            dtype=self._dtype,
        )
        self.adaLN_modulate_linear_bias = paddle.create_parameter(
            attr=fmt_blocks_addLN_modulate_linear_bias_attr,
            default_initializer=Constant(value=1.0),
            shape=[6*config.embed_dim * self.num_layers],
            dtype=self._dtype,
        )


        self._add_parameter(self.adaLN_modulate_linear_weight)
        self._add_parameter(self.adaLN_modulate_linear_bias)


        for i in range(self.num_layers):

            ln_scale_attr = self.get_attr(config.ln_scale_attrs, i)
            ln_bias_attr = self.get_attr(config.ln_bias_attrs, i)
            qkv_weight_attr = self.get_attr(config.qkv_weight_attrs, i)
            qkv_bias_attr = self.get_attr(config.qkv_bias_attrs, i)
            q_norm_weight_attr = self.get_attr(config.q_norm_weight_attrs, i)
            q_norm_bias_attr = self.get_attr(config.q_norm_bias_attrs, i)
            k_norm_weight_attr = self.get_attr(config.k_norm_weight_attrs, i)
            k_norm_bias_attr = self.get_attr(config.k_norm_bias_attrs, i)
            linear_weight_attr = self.get_attr(config.linear_weight_attrs, i)
            linear_bias_attr = self.get_attr(config.linear_bias_attrs, i)


            ffn_ln_scale_attr = self.get_attr(config.ffn_ln_scale_attrs, i)
            ffn_ln_bias_attr = self.get_attr(config.ffn_ln_bias_attrs, i)
            ffn1_weight_attr = self.get_attr(config.ffn1_weight_attrs, i)
            ffn1_bias_attr = self.get_attr(config.ffn1_bias_attrs, i)
            ffn2_weight_attr = self.get_attr(config.ffn2_weight_attrs, i)
            ffn2_bias_attr = self.get_attr(config.ffn2_bias_attrs, i)



            ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[config.embed_dim],
                default_initializer=Constant(value=1.0),
                dtype=self._norm_weight_dtype,
            )
            ln_bias = None
            if ln_bias_attr:
                ln_bias = self.create_parameter(
                    attr=ln_bias_attr,
                    shape=[config.embed_dim],
                    is_bias=True,
                    dtype=self._norm_weight_dtype,
                )


            q_norm_weight = self.create_parameter(
                attr=q_norm_weight_attr,
                shape=[config.embed_dim],
                default_initializer=Constant(value=1.0),
                dtype=self._norm_weight_dtype,
            )
            q_norm_bias = None
            if q_norm_bias_attr:
                q_norm_bias = self.create_parameter(
                    attr=q_norm_bias_attr,
                    shape=[config.embed_dim],
                    is_bias=True,
                    dtype=self._norm_weight_dtype,
                )
            k_norm_weight = self.create_parameter(
                attr=k_norm_weight_attr,
                shape=[config.embed_dim],
                default_initializer=Constant(value=1.0),
                dtype=self._norm_weight_dtype,
            )
            k_norm_bias = None
            if k_norm_bias_attr:
                k_norm_bias = self.create_parameter(
                    attr=k_norm_bias_attr,
                    shape=[config.embed_dim],
                    is_bias=True,
                    dtype=self._norm_weight_dtype,
                )

            self.init_weight_shape(config)
            qkv_weight = self.create_parameter(
                shape=self.qkv_weight_shape,
                attr=qkv_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            qkv_bias = None
            if qkv_bias_attr:
                qkv_bias = self.create_parameter(
                    shape=[3 * self.num_heads * self.head_dim],
                    attr=qkv_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            linear_weight = self.create_parameter(
                shape=self.linear_weight_shape,
                attr=linear_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            linear_bias = None
            if linear_bias_attr:
                linear_bias = self.create_parameter(
                    shape=[config.embed_dim],
                    attr=linear_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )


            ffn_ln_scale = self.create_parameter(
                shape=[config.embed_dim],
                attr=ffn_ln_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0),
                dtype=self._norm_weight_dtype,
            )
            ffn_ln_bias = None
            if ffn_ln_bias_attr:
                ffn_ln_bias = self.create_parameter(
                    shape=[config.embed_dim],
                    attr=ffn_ln_bias_attr,
                    is_bias=True,
                    dtype=self._norm_weight_dtype,
                )
            # TODO(wangbojun), ffn shape need refine 
            ffn1_weight = self.create_parameter(
                shape=self.ffn1_weight_shape,
                attr=ffn1_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            ffn1_bias = None
            if ffn1_bias_attr:
                ffn1_bias = self.create_parameter(
                    shape=[dim_feedforward * 2] if config.activation.endswith("glu") else [dim_feedforward],
                    attr=ffn1_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            ffn2_weight = self.create_parameter(
                shape=self.ffn2_weight_shape,
                attr=ffn2_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            ffn2_bias = None
            if ffn2_bias_attr:
                ffn2_bias = self.create_parameter(
                    shape=[config.embed_dim],
                    attr=ffn2_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )





            self.ln_scales.append(ln_scale)
            self.ln_biases.append(ln_bias)
            
            self.qkv_weights.append(qkv_weight)
            self.qkv_biases.append(qkv_bias)
            self.q_norm_weights.append(q_norm_weight)
            self.q_norm_biases.append(q_norm_bias)
            self.k_norm_weights.append(k_norm_weight)
            self.k_norm_biases.append(k_norm_bias)
            self.linear_weights.append(linear_weight)
            self.linear_biases.append(linear_bias)



            self.ffn_ln_scales.append(ffn_ln_scale)
            self.ffn_ln_biases.append(ffn_ln_bias)
            self.ffn1_weights.append(ffn1_weight)
            self.ffn1_biases.append(ffn1_bias)
            self.ffn2_weights.append(ffn2_weight)
            self.ffn2_biases.append(ffn2_bias)

            

            self._add_parameter(ln_scale)
            self._add_parameter(ln_bias)

            self._add_parameter(q_norm_weight)
            self._add_parameter(q_norm_bias)
            self._add_parameter(k_norm_weight)
            self._add_parameter(k_norm_bias)

            self._add_parameter(qkv_weight)
            self._add_parameter(qkv_bias)
            self._add_parameter(linear_weight)
            self._add_parameter(linear_bias)

            self._add_parameter(ffn_ln_scale)
            self._add_parameter(ffn_ln_bias)
            self._add_parameter(ffn1_weight)
            self._add_parameter(ffn1_bias)
            self._add_parameter(ffn2_weight)
            self._add_parameter(ffn2_bias)
        pass

    def norm_func_wrap(self, x, norm_weight=None, norm_bias=None, epsilon=1e-8, begin_norm_axis=1, bias=None, residual=None):
        if self.export:
            return self.norm_func(x=x, norm_weight=norm_weight, norm_bias=norm_bias, epsilon=epsilon, begin_norm_axis=begin_norm_axis, bias=bias, residual=residual)
        else:
            return self.norm_func(x=x, norm_weight=norm_weight, norm_bias=norm_bias, epsilon=epsilon, begin_norm_axis=begin_norm_axis, bias=bias, residual=residual)[0]
    
    def get_attr(self, attrs, idx):
        if isinstance(attrs, (list, tuple)):
            assert (
                len(attrs) == self.num_layers
            ), f"length of attrs is {len(attrs)} is not equal to self.num_layers {self.num_layers}"
            return attrs[idx]
        return attrs

    def _add_parameter(self, param):
        if param is None:
            return
        assert param.name not in self._parameters
        self._parameters[param.name] = param


    def init_weight_shape(self, config):
        self.qkv_weight_shape = (
            [3 * self.num_heads * self.head_dim, self.embed_dim]
            if config.trans_qkvw
            else [self.embed_dim * 3 * self.num_heads, self.head_dim]
        )
        self.linear_weight_shape = [self.num_heads * self.head_dim, self.embed_dim]        
        ffn2_hidden_features = int(2 * self.dim_feedforward / 3)
        #TODO(wangbojun), ffn2 swiglu multiof need check or get from``
        ffn2_multiple_of = 256
        ffn2_hidden_features = ffn2_multiple_of * ((ffn2_hidden_features + ffn2_multiple_of - 1) // ffn2_multiple_of)
        
        self.ffn1_weight_shape = (
            [self.embed_dim, ffn2_hidden_features * 2]
            if self.activation.endswith("glu")
            else [self.embed_dim, ffn2_hidden_features]
        )
        self.ffn2_weight_shape = [ffn2_hidden_features, self.embed_dim]


    def compute_layernorm_before_qkv(self, src, i):
        ln_out = self.norm_func_wrap(x=src, norm_weight=self.ln_scales[i], norm_bias=self.ln_biases[i], epsilon=self._epsilon, begin_norm_axis=1)
        return ln_out

    def compute_qkv_linear(self, ln_out, i):
        if float(paddle.version.cuda()) < 11.6:
            qkv_out = paddle.matmul(ln_out, self.qkv_weights[i], False, True)
            if self.qkv_biases[i] is not None:
                qkv_out = paddle.add(qkv_out, self.qkv_biases[i])
            return qkv_out
        else:
            # This method requires CUDA version >= 11.6.
            return self.linear(ln_out, self.qkv_weights[i], self.qkv_biases[i], transpose_weight=True)
    def compute_adaLN_modulate(self, input):
        fused_res = fused_linear(input, self.adaLN_modulate_linear_weight, self.adaLN_modulate_linear_bias)
        return paddle.split(fused_res, num_or_sections=self.num_layers * 6, axis=1)
        


    def compute_qkv_with_modulate(self, src, residual_input, shift_msa, scale_msa, i, modulate):
        ln_out = self.compute_layernorm_before_qkv(src, i)

        modulate_out = modulate(ln_out, shift_msa, scale_msa)
        qkv_out = self.compute_qkv_linear(modulate_out, i)
        return qkv_out, residual_input
    
    def apply_rotary_emb(self, xq, xk, freqs_cis):
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
            seq_len, dim = freqs_cis.shape
            xq_ = xq_.reshape([-1,seq_len, self.num_heads, dim])
            xk_ = xk_.reshape([-1,seq_len, self.num_heads, dim])
            freqs_cis = Attention.reshape_for_broadcast(freqs_cis, xq_)
            
            xq_out = paddle.as_real(xq_ * freqs_cis).flatten(start_axis=3)
            xk_out = paddle.as_real(xk_ * freqs_cis).flatten(start_axis=3)
            xq_out = xq_out.reshape([-1, self.num_heads, self.head_dim])

            xk_out = xk_out.reshape([-1, self.num_heads, self.head_dim])
            return xq_out.cast(xq.dtype), xk_out.cast(xk.dtype)


    def compute_fmha(
        self,
        q_out,
        k_out,
        v_out,
        freqs_cis,
        cu_seq_lens,
        seq_len,
        max_seq_len_q,
        max_seq_len_kv
    ):
        """
        qkv: bsz, seq_len, 3, numhead, headsize ->
        q_out: bsz, numhead, seq_len, headsize
        kv_out: 2, bsz, numhead, seq_len, headsize
        """
        #TODO wangbojun rope
        q_out, k_out = self.apply_rotary_emb(q_out,k_out,freqs_cis)
        qktv_out, _ = flash_attn_unpadded(
            # q_out.unsqueeze(0),k_out.unsqueeze(0),v_out.unsqueeze(0),
            q_out,k_out,v_out,
            cu_seq_lens,
            cu_seq_lens,
            max_seq_len_q,
            max_seq_len_kv,
            1.0/math.sqrt(self.head_dim),
            training=False
        )
        '''
         qktv_out, _ = flash_attn_unpadded(q_out.unsqueeze(0),k_out.unsqueeze(0),v_out.unsqueeze(0), cu_seq_lens, cu_seq_lens, max_seq_len_q, max_seq_len_kv, 1.0/math.sqrt(self.head_dim), training=False)
        '''
        # qktv_out = qktv_out.squeeze(axis=0)
        qktv_out_reshape = qktv_out.reshape([0,-1])

        return qktv_out_reshape

    def compute_out_linear(self, fmha_out, i):
        return paddle.matmul(fmha_out, self.linear_weights[i])
    def compute_qk_norm(self, qkv_out, i):
        qkv_out = qkv_out.reshape([0,3,self.num_heads, self.head_dim])
        q = qkv_out[:,0,:,:].reshape([-1, self.num_heads, self.head_dim])
        k = qkv_out[:,1,:,:].reshape([-1, self.num_heads, self.head_dim])
        v = qkv_out[:,2,:,:].reshape([-1, self.num_heads, self.head_dim])
        q_norm_out = self.norm_func_wrap(x=q, norm_weight=self.q_norm_weights[i], norm_bias=self.q_norm_biases[i], epsilon=1e-6)
        k_norm_out = self.norm_func_wrap(x=k, norm_weight=self.k_norm_weights[i], norm_bias=self.k_norm_biases[i], epsilon=1e-6)
        return q_norm_out, k_norm_out, v
    def compute_attn(
        self,
        qkv_out,
        freqs_cis,
        cu_seq_lens,
        seq_len,
        max_seq_len_q,
        max_seq_len_kv,
        i
    ):
        # fmha compute

        q_norm_out, k_norm_out, v_out = self.compute_qk_norm(qkv_out, i)
        fmha_out = self.compute_fmha(
            q_norm_out, k_norm_out, v_out,
            freqs_cis,
            cu_seq_lens,
            seq_len,
            max_seq_len_q,
            max_seq_len_kv
        )
        out_linear_out = self.compute_out_linear(fmha_out, i)
        return out_linear_out

    def compute_ffn_layernorm(self, out_linear_out, i):
        norm_out = self.norm_func_wrap(
            out_linear_out,
            norm_weight=self.ffn_ln_scales[i],
            norm_bias=self.ffn_ln_biases[i],
            epsilon=self._epsilon,
            begin_norm_axis=1,
            bias=self.linear_biases[i],
            # residual=residual_input,
        )
        return norm_out

    def compute_activation(self, ffn1_out, i):
        #TODO(wangbojun)
        return fused_act_bias_wrapper(ffn1_out, self.ffn1_biases[i], act_method="swiglu")

    def compute_ffn1(self, tmp_out, i):
        return paddle.matmul(tmp_out, self.ffn1_weights[i])
    def compute_ffn2(self, ffn1_out, i):
        return paddle.matmul(ffn1_out, self.ffn2_weights[i])

    def forward(self, x, freqs_cis, adaln_input=None,
                cu_seq_lens=None,
                img_seq_lens=None,
                max_seq_len_q=None,
                max_seq_len_kv=None):
        x=x.reshape([-1,self.embed_dim])
        silu_out = nn.functional.silu(adaln_input)
        adaln_input_modulation = self.compute_adaLN_modulate(silu_out)
        for i in range(self.num_layers):
            if adaln_input is not None:
                index = i * 6 
                residual_input = x
                shift_msa = adaln_input_modulation[index]
                scale_msa = adaln_input_modulation[index + 1]
                gate_msa = adaln_input_modulation[index + 2]
                shift_mlp = adaln_input_modulation[index + 3]
                scale_mlp = adaln_input_modulation[index + 4]
                gate_mlp = adaln_input_modulation[index + 5]
                qkv_out, _ = self.compute_qkv_with_modulate(x, residual_input, shift_msa, scale_msa, i, modulate)
                out_linear_out = self.compute_attn(
                    qkv_out,
                    freqs_cis,
                    cu_seq_lens,
                    img_seq_lens,
                    max_seq_len_q,
                    max_seq_len_kv,
                    i
                )
                bs,dim = gate_msa.shape
                out_linear_out=out_linear_out.reshape([bs,-1,dim])
                gate_msa_out = gate_msa.unsqueeze(1) * out_linear_out
                h = residual_input + gate_msa_out.reshape([-1,dim])
                ffn_ln_out = self.compute_ffn_layernorm(h, i)
                # ffn1 matmul
                ffn1_out = self.compute_ffn1(modulate(ffn_ln_out, shift_mlp, scale_mlp), i)
                ffn1_out = self.compute_activation(ffn1_out, i)
                # ffn2 matmul
                ffn2_out = self.compute_ffn2(ffn1_out, i)
                # out = h + gate_mlp.unsqueeze(1) * ffn2_out
                out = h.reshape([-1,max_seq_len_q, dim]) + gate_mlp.unsqueeze(1) * ffn2_out.reshape([-1,max_seq_len_q, dim])
                x=out.reshape([-1,self.embed_dim])
            else:
                pass
            pass
        return x

class LagerDitInferenceModel(ModelMixin, ConfigMixin):
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
        export: bool = False,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        dim = attention_head_dim * num_attention_heads
        self.emb_dim = dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.dim_ffn = mlp_ratio * dim
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

        self.export = export

        # 2. Define transformers blocks
        # TODO(wangbojun)


        fmt_blocks_attn_ln_scale_attrs = [paddle.ParamAttr(name="fmt_blocks.{}.attn_ln_scale".format(i)) for i in range(self.num_layers)]


        # no attn norm and ffn norm bias for dit
        # fmt_blocks_attn_ln_bias_attrs = [paddle.ParamAttr(name="fmt_blocks.{}.attn_ln_bias".format(i)) for i in range(self.num_layers)]

        fmt_blocks_attn_qkv_weight_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks.{}.attn_qkv_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]


        fmt_blocks_attn_q_norm_weight_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks.{}.attn_q_norm_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        fmt_blocks_attn_q_norm_bias_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks.{}.attn_q_norm_bias".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]

        fmt_blocks_attn_k_norm_weight_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks.{}.attn_k_norm_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        fmt_blocks_attn_k_norm_bias_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks.{}.attn_k_norm_bias".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]


        fmt_blocks_attn_out_proj_weight_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks.{}.out_proj_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        fmt_blocks_ffn_ln_scale_attrs = [
            paddle.ParamAttr(name="fmt_blocks.{}.ffn_ln_scale".format(i)) for i in range(self.num_layers)
        ]
        fmt_blocks_ffn1_weight_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks.{}.ffn1_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]
        fmt_blocks_ffn2_weight_attrs = [
            paddle.ParamAttr(
                name="fmt_blocks.{}.ffn2_weight".format(i), initializer=paddle.nn.initializer.Constant(value=0)
            )
            for i in range(self.num_layers)
        ]

        self.fmt_config = DitFusedMultiTransformerLayersConfig(
            embed_dim = self.emb_dim,
            num_heads = self.num_attention_heads,
            dim_feedforward = self.dim_ffn,
            weight_only_quant_bits=-1, #todo(wangbojun)
            dropout_rate=0.0,
            activation="geglu",
            norm_type="layernorm",
            use_neox_rotary_style=False,
            normalize_before=True,
            ln_scale_attrs=fmt_blocks_attn_ln_scale_attrs,
            qkv_weight_attrs=fmt_blocks_attn_qkv_weight_attrs,
            q_norm_weight_attrs=fmt_blocks_attn_q_norm_weight_attrs,
            q_norm_bias_attrs=fmt_blocks_attn_q_norm_bias_attrs,
            k_norm_weight_attrs=fmt_blocks_attn_k_norm_weight_attrs,
            k_norm_bias_attrs=fmt_blocks_attn_k_norm_bias_attrs,
            linear_weight_attrs=fmt_blocks_attn_out_proj_weight_attrs,
            ffn_ln_scale_attrs=fmt_blocks_ffn_ln_scale_attrs,
            ffn1_weight_attrs=fmt_blocks_ffn1_weight_attrs,
            ffn2_weight_attrs=fmt_blocks_ffn2_weight_attrs,
            epsilon=1e-5,
            residual_alpha=1.0,
            num_layers=self.num_layers,
            nranks=1,
            trans_qkvw=True,
            ring_id=-1,
            skip=False
        )
        self.fmt_layer = DiTFusedMultiTransformerLayers(
            self.fmt_config,
            export=self.export,
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
                          #012345 
        # x = paddle.einsum("nhwpqc->nchpwq", x.cast("float32")).cast("bfloat16")
        # [TODO wangbojun], need check
        x = x.transpose([0,5,1,3,2,4])
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
    def __set_value(self, weight,state_dict, params_name):
        print(f"process weight: {params_name}, param shape is {state_dict[params_name].shape} , var shape is: {weight.shape}, var name : {weight.name}, dtype:{weight.dtype}")
        assert(weight.shape == state_dict[params_name].shape)
        weight.set_value(paddle.to_tensor(state_dict[params_name], dtype=weight.dtype))
        print(f"process weight done")

    def set_state_dict(self, state_dict):
        self.__set_value(self.x_embedder.weight, state_dict, "x_embedder.weight")
        self.__set_value(self.x_embedder.bias, state_dict, "x_embedder.bias")
        self.__set_value(self.t_embedder.mlp[0].weight, state_dict, "t_embedder.mlp.0.weight")
        self.__set_value(self.t_embedder.mlp[0].bias, state_dict, "t_embedder.mlp.0.bias")

        self.__set_value(self.t_embedder.mlp[2].weight, state_dict, "t_embedder.mlp.2.weight")
        self.__set_value(self.t_embedder.mlp[2].bias, state_dict, "t_embedder.mlp.2.bias")

        self.__set_value(self.y_embedder.embedding_table.weight, state_dict, "y_embedder.embedding_table.weight")

        # self.__set_value(self.decoder_pred.weight, state_dict, "decoder_pred.weight")
        # self.__set_value(self.decoder_pred.bias, state_dict, "decoder_pred.bias")
        self.__set_value(self.final_layer.adaLN_modulation[1].weight, state_dict, "final_layer.adaLN_modulation.1.weight")
        self.__set_value(self.final_layer.adaLN_modulation[1].bias, state_dict, "final_layer.adaLN_modulation.1.bias")
        self.__set_value(self.final_layer.linear.weight, state_dict, "final_layer.linear.weight")
        self.__set_value(self.final_layer.linear.bias, state_dict, "final_layer.linear.bias")



        for i in range(self.fmt_layer.num_layers):

            if i==0:
                state_dict["adaLN_modulate_weight"] = state_dict[f"layers.{i}.adaLN_modulation.1.weight"]
                state_dict["adaLN_modulate_bias"] = state_dict[f"layers.{i}.adaLN_modulation.1.bias"]
            else:
                state_dict["adaLN_modulate_weight"] = paddle.concat([state_dict["adaLN_modulate_weight"], state_dict[f"layers.{i}.adaLN_modulation.1.weight"]], axis=-1)
                state_dict["adaLN_modulate_bias"] = paddle.concat([state_dict["adaLN_modulate_bias"], state_dict[f"layers.{i}.adaLN_modulation.1.bias"]], axis=-1)

            self.__set_value(self.fmt_layer.ln_scales[i],
                            state_dict,f"layers.{i}.attention_norm.weight")
                            
            # self.__set_value(self.fmt_layer.ln_biases[i],
            #                 state_dict,f"layers.{i}.attention_norm.bias")

            state_dict[f'layers.{i}.attention.wq.weight'] = state_dict[f'layers.{i}.attention.wq.weight'].transpose([1,0])
            state_dict[f'layers.{i}.attention.wk.weight'] = state_dict[f'layers.{i}.attention.wk.weight'].transpose([1,0])
            state_dict[f'layers.{i}.attention.wv.weight'] = state_dict[f'layers.{i}.attention.wv.weight'].transpose([1,0])

            state_dict[f'layers.{i}.attn.qkv.weight'] = paddle.concat([state_dict[f'layers.{i}.attention.wq.weight'],
                                                                       state_dict[f'layers.{i}.attention.wk.weight'],
                                                                       state_dict[f'layers.{i}.attention.wv.weight']])
            self.__set_value(self.fmt_layer.qkv_weights[i],state_dict, f'layers.{i}.attn.qkv.weight')

            self.__set_value(self.fmt_layer.q_norm_weights[i], state_dict, f'layers.{i}.attention.q_norm.weight')
            self.__set_value(self.fmt_layer.q_norm_biases[i], state_dict, f'layers.{i}.attention.q_norm.bias')

            self.__set_value(self.fmt_layer.k_norm_weights[i], state_dict, f'layers.{i}.attention.k_norm.weight')
            self.__set_value(self.fmt_layer.k_norm_biases[i], state_dict, f'layers.{i}.attention.k_norm.bias')


            self.__set_value(self.fmt_layer.linear_weights[i],state_dict, f'layers.{i}.attention.wo.weight')
            # self.__set_value(self.fmt_layer.linear_biases[i],state_dict, f'layers.{i}.attention.wo.bias')
            #TODO(wangbojun)
            self.__set_value(self.fmt_layer.ffn_ln_scales[i],state_dict, f'layers.{i}.ffn_norm.weight')
            # self.__set_value(self.fmt_layer.ffn_ln_biases[i],state_dict, f'layers.{i}.ffn_norm.bias')
            state_dict[f'layers.{i}.feed_forward.ffn1.weight'] = paddle.concat([
                state_dict[f'layers.{i}.feed_forward.w1.weight'],
                state_dict[f'layers.{i}.feed_forward.w3.weight']
            ], axis=-1)
            self.__set_value(self.fmt_layer.ffn1_weights[i],state_dict, f'layers.{i}.feed_forward.ffn1.weight')
            self.__set_value(self.fmt_layer.ffn2_weights[i],state_dict, f'layers.{i}.feed_forward.w2.weight')
        
        self.__set_value(self.fmt_layer.adaLN_modulate_linear_weight, state_dict, "adaLN_modulate_weight")
        self.__set_value(self.fmt_layer.adaLN_modulate_linear_bias, state_dict, "adaLN_modulate_bias")

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

        # timestep = paddle.to_tensor([timestep],dtype=self.dtype)

        # 1. Input
        hidden_states = self.patchify(hidden_states)
        x = self.x_embedder(hidden_states)
        x_bs, x_seq_len, x_dim = x.shape
        # x_seq_lens_tensor = paddle.to_tensor([x_seq_len]*x_bs, dtype='int32')
        x_seq_lens_tensor =  paddle.full(shape=[x_bs], fill_value=x_seq_len, dtype='int32')
        x_cu_seq_lens_tensor = paddle.concat([paddle.to_tensor([0],dtype='int32'), paddle.cumsum(x_seq_lens_tensor)])
        x_max_seq_lens = x_seq_len
        t = self.t_embedder(timestep)
        y = self.y_embedder(class_labels)
        adaln_input = t + y
        #已对齐
        # 2. Blocks

        freqs_cis_this_time = self.freqs_cis[:x.shape[1]]
        x = self.fmt_layer(x, freqs_cis_this_time, adaln_input,
                            x_cu_seq_lens_tensor,
                            x_seq_lens_tensor,
                            x_max_seq_lens,
                            x_max_seq_lens
                            )
        
        # 3. Output
        hidden_states = self.final_layer(x, adaln_input)

        output = self.unpatchify(hidden_states.reshape([x_bs,x_seq_len,-1]))
        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

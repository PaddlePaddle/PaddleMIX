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

import math
import warnings
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import paddle
from paddle import nn
from paddle.nn.functional import linear, pad


def pad_sequence(sequences, padding_value=0, fix_len=None):
    """Fill sequences(np.ndarray) into a fixed-length matrix."""
    max_size = sequences[0].shape
    trailing_dims = max_size[1:]
    max_len = max([s.shape[0] for s in sequences])
    if fix_len is not None:
        assert fix_len >= max_len, "fix_len is too small."
        max_len = fix_len
    out_dims = (len(sequences), max_len) + tuple(trailing_dims)
    out_tensor = paddle.to_tensor(np.full(out_dims, padding_value)).cast(sequences[0].dtype)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        out_tensor[i, :length, ...] = tensor
    return out_tensor


def get_2d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (image_height, image_width)
    return:
    pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]
    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=-1)
    return emb


def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("hw,d->hwd", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=-1)
    return emb


class Resampler(nn.Layer):
    """
    A 2D perceiver-resampler network with one cross attention layers by
       given learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (batch_size, num_queries, embed_dim)
    """

    def __init__(
        self,
        num_queries,
        embed_dim,
        num_heads,
        kv_dim=None,
        norm_layer=partial(nn.LayerNorm),
        adaptive=False,
        max_size=(70, 70),
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.adaptive = adaptive
        self.max_size = max_size
        self.query = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=[self.num_queries, embed_dim])
        )
        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(in_features=kv_dim, out_features=embed_dim, bias_attr=False)
        else:
            self.kv_proj = nn.Identity()
        self.attn = MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        self.ln_post = norm_layer(embed_dim)
        self.proj = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=embed_dim**-0.5 * paddle.randn(shape=[embed_dim, embed_dim])
        )
        self._set_2d_pos_cache(self.max_size)

    def _set_2d_pos_cache(self, max_size, device="cpu"):
        # if transformers.integrations.is_deepspeed_zero3_enabled():
        device = "gpu"
        pos_embed = (
            paddle.to_tensor(data=get_2d_sincos_pos_embed(self.embed_dim, max_size)).astype(dtype="float32").to(device)
        )
        self.register_buffer(name="pos_embed", tensor=pos_embed, persistable=False)

    def _adjust_pos_cache(self, tgt_sizes, device):
        max_h = paddle.max(x=tgt_sizes[:, 0])
        max_w = paddle.max(x=tgt_sizes[:, 1])
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [max(max_h, self.max_size[0]), max(max_w, self.max_size[1])]
            self._set_2d_pos_cache(self.max_size, device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init_TruncatedNormal = nn.initializer.TruncatedNormal(std=0.02)
            init_TruncatedNormal(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                init_Constant = nn.initializer.Constant(value=0)
                init_Constant(m.bias)
        elif isinstance(m, nn.LayerNorm):
            init_Constant = nn.initializer.Constant(value=0)
            init_Constant(m.bias)
            init_Constant = nn.initializer.Constant(value=1.0)
            init_Constant(m.weight)

    def forward(self, x, tgt_sizes=None):
        assert tuple(x.shape)[0] == tuple(tgt_sizes.shape)[0]
        bs = tuple(x.shape)[0]
        device = x.place
        dtype = x.dtype
        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]
        self._adjust_pos_cache(tgt_sizes, device=device)
        max_patch_len = paddle.max(x=patch_len)
        key_padding_mask = paddle.zeros(shape=(bs, max_patch_len), dtype="bool")
        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype))
            key_padding_mask[i, patch_len[i] :] = True
        pos_embed = pad_sequence(pos_embed, padding_value=0.0).transpose([1, 0, 2])
        x = self.kv_proj(x)
        x = self.ln_kv(x).transpose([1, 0, 2])
        q = self.ln_q(self.query)
        out = self.attn(self._repeat(q, bs), x + pos_embed, x, key_padding_mask=key_padding_mask)[0]
        x = out.transpose([1, 0, 2])
        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def _repeat(self, query, N: int):
        return query.unsqueeze(axis=1).tile(repeat_times=[1, N, 1])


class MultiheadAttention(nn.MultiHeadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ):
        super().__init__(embed_dim, num_heads, dropout, kdim, vdim, bias_attr=bias)
        self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias_attr=bias)
        # 手动 add code
        self.batch_first = batch_first
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = self.create_parameter(
                shape=[embed_dim, embed_dim],
            )
            self.k_proj_weight = self.create_parameter(
                shape=[self.kdim, embed_dim],
            )
            self.v_proj_weight = self.create_parameter(
                shape=[self.vdim, embed_dim],
            )
            self.in_proj_weight = None
        else:
            self.in_proj_weight = self.create_parameter(
                shape=[embed_dim, 3 * embed_dim],
            )
            self.q_proj_weight = None
            self.k_proj_weight = None
            self.v_proj_weight = None

        if bias:
            self.in_proj_bias = self.create_parameter(
                shape=[3 * embed_dim],
            )
        else:
            self.in_proj_bias = None
        # NonDynamicallyQuantizableLinear 用 nn.Linear 替换
        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim,
            bias_attr=bias,
        )

        if add_bias_kv:
            self.bias_k = self.create_parameter(
                shape=[1, 1, embed_dim],
            )
            self.bias_v = self.create_parameter(
                shape=[1, 1, embed_dim],
            )
        else:
            self.bias_k = None
            self.bias_v = None

        self.add_zero_attn = add_zero_attn
        

    def forward(
        self,
        query: paddle.Tensor,
        key: paddle.Tensor,
        value: paddle.Tensor,
        key_padding_mask: Optional[paddle.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[paddle.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
        why_not_fast_path = ""
        if (
            attn_mask is not None
            and paddle.is_floating_point(x=attn_mask)
            or key_padding_mask is not None
            and paddle.is_floating_point(x=key_padding_mask)
        ):
            why_not_fast_path = "floating-point masks are not supported for fast path."
        is_batched = query.dim() == 3
        key_padding_mask = _canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=_none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )
        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = (
                f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
            )
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            why_not_fast_path = (
                f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
            )
        elif self.training:  # False
            why_not_fast_path = "training is enabled"
        elif self.num_heads % 2 != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:  # False
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:  # True
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        # query.is_nested = False paddle is not support nested tensor
        elif False and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time                                  is not supported with NestedTensor input"
        elif paddle.amp.is_auto_cast_enabled():
            why_not_fast_path = "autocast is enabled"
        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )

            def _check_arg_device(x):
                # Paddle中检查设备类型
                if isinstance(x, paddle.Tensor):
                    return x.place.__str__() in ["CPUPlace", "CUDAPlace(0)", "NPUPlace(0)"]  # 根据需要添加其他设备
                return True

            def _arg_requires_grad(x):
                # 检查张量是否需要梯度
                if isinstance(x, paddle.Tensor):
                    return x.stop_gradient is False
                return False

            def _is_make_fx_tracing():
                # Paddle中目前没有完全对应的make_fx tracing
                # 如果需要，可以自定义一个跟踪状态标志
                return False

            # 主要逻辑
            if any(hasattr(x, "__paddle_function__") for x in tensor_args):  # Paddle中检查自定义张量操作
                why_not_fast_path = "some Tensor argument has custom operations"
            elif _is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(_check_arg_device(x) for x in tensor_args):
                why_not_fast_path = (
                    "some Tensor argument's device is neither one of " "cpu, gpu or npu"
                )  # 根据实际支持的设备类型修改
            elif paddle.is_grad_enabled() and any(_arg_requires_grad(x) for x in tensor_args):
                why_not_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )
                if not why_not_fast_path:
                    merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)
                    if self.in_proj_bias is not None and self.in_proj_weight is not None:
                        # 准备输入投影权重和偏置
                        # Paddle中需要将输入投影权重分成Q、K、V三部分
                        q_proj_weight, k_proj_weight, v_proj_weight = self.in_proj_weight.chunk(3, axis=0)
                        q_proj_bias, k_proj_bias, v_proj_bias = self.in_proj_bias.chunk(3, axis=0)

                        # 计算Q、K、V的投影
                        q = paddle.matmul(query, q_proj_weight.t()) + q_proj_bias
                        k = paddle.matmul(key, k_proj_weight.t()) + k_proj_bias
                        v = paddle.matmul(value, v_proj_weight.t()) + v_proj_bias

                        # 重塑张量形状以适应多头注意力
                        batch_size = query.shape[0]
                        q = q.reshape([batch_size, -1, self.num_heads, self.embed_dim // self.num_heads])
                        q = q.transpose([0, 2, 1, 3])  # [batch_size, num_heads, seq_len, head_dim]
                        k = k.reshape([batch_size, -1, self.num_heads, self.embed_dim // self.num_heads])
                        k = k.transpose([0, 2, 1, 3])
                        v = v.reshape([batch_size, -1, self.num_heads, self.embed_dim // self.num_heads])
                        v = v.transpose([0, 2, 1, 3])

                        # 计算注意力分数
                        scale = float(self.embed_dim // self.num_heads) ** -0.5
                        attn_output_weights = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * scale

                        # 应用注意力掩码
                        if merged_mask is not None:
                            attn_output_weights = attn_output_weights + merged_mask

                        attn_output_weights = paddle.nn.functional.softmax(attn_output_weights, axis=-1)

                        # 计算输出
                        attn_output = paddle.matmul(attn_output_weights, v)

                        # 重塑输出
                        attn_output = attn_output.transpose([0, 2, 1, 3])
                        attn_output = attn_output.reshape([batch_size, -1, self.embed_dim])

                        # 应用输出投影
                        output = paddle.matmul(attn_output, self.out_proj.weight.t())
                        if self.out_proj.bias is not None:
                            output = output + self.out_proj.bias

                        if need_weights:
                            if average_attn_weights:
                                attn_output_weights = attn_output_weights.mean(axis=1)
                            return output, attn_output_weights
                        else:
                            return output, None
        # any_nested = query.is_nested or key.is_nested or value.is_nested
        any_nested = False
        assert not any_nested, (
            "MultiheadAttention does not support NestedTensor outside of its fast path. "
            + f"The fast path was not hit because {why_not_fast_path}"
        )
        if self.batch_first and is_batched:
            if paddle.equal_all(key, value):  # 替代 key is value
                if paddle.equal_all(query, key):  # 替代 query is key
                    query = key = value = query.transpose([1, 0, 2])
                else:
                    query = query.transpose([1, 0, 2])
                    key = key.transpose([1, 0, 2])
                    value = key
            else:
                query = query.transpose([1, 0, 2])
                key = key.transpose([1, 0, 2])
                value = value.transpose([1, 0, 2])
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            attn_output, attn_output_weights = self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        if self.batch_first and is_batched:
            return attn_output.transpose([1, 0, 2]), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def multi_head_attention_forward(
        self,
        query: paddle.Tensor,
        key: paddle.Tensor,
        value: paddle.Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Optional[paddle.Tensor],
        in_proj_bias: Optional[paddle.Tensor],
        bias_k: Optional[paddle.Tensor],
        bias_v: Optional[paddle.Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: paddle.Tensor,
        out_proj_bias: Optional[paddle.Tensor],
        training: bool = True,
        key_padding_mask: Optional[paddle.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[paddle.Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[paddle.Tensor] = None,
        k_proj_weight: Optional[paddle.Tensor] = None,
        v_proj_weight: Optional[paddle.Tensor] = None,
        static_k: Optional[paddle.Tensor] = None,
        static_v: Optional[paddle.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:

        is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)
        if not is_batched:
            query = query.unsqueeze(axis=1)
            key = key.unsqueeze(axis=1)
            value = value.unsqueeze(axis=1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(axis=0)
        tgt_len, bsz, embed_dim = tuple(query.shape)
        src_len, _, _ = tuple(key.shape)
        key_padding_mask = _canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=_none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )
        if is_causal and attn_mask is None:
            raise RuntimeError(
                "Need attn_mask if specifying the is_causal hint. You may use the Transformer module method `generate_square_subsequent_mask` to create this mask."
            )
        if is_causal and key_padding_mask is None and not need_weights:
            attn_mask = None
        else:
            attn_mask = _canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=query.dtype,
                check_other=False,
            )
            if key_padding_mask is not None:
                is_causal = False
        assert (
            embed_dim == embed_dim_to_check
        ), f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        if isinstance(embed_dim, paddle.Tensor):
            head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
        else:
            head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        if use_separate_proj_weight:
            assert (
                tuple(key.shape)[:2] == tuple(value.shape)[:2]
            ), f"key's sequence and batch dims {tuple(key.shape)[:2]} do not match value's {tuple(value.shape)[:2]}"
        else:
            assert tuple(key.shape) == tuple(
                value.shape
            ), f"key shape {tuple(key.shape)} does not match value shape {tuple(value.shape)}"
        if not use_separate_proj_weight:
            assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
            q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        else:
            assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
            assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
            assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(chunks=3, axis=-1)
            q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                correct_2d_size = tgt_len, src_len
                if tuple(attn_mask.shape) != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {tuple(attn_mask.shape)}, but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(axis=0)
            elif attn_mask.dim() == 3:
                correct_3d_size = bsz * num_heads, tgt_len, src_len
                if tuple(attn_mask.shape) != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {tuple(attn_mask.shape)}, but should be {correct_3d_size}."
                    )
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")
        if bias_k is not None and bias_v is not None:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
            k = paddle.concat(x=[k, bias_k.tile(repeat_times=[1, bsz, 1])])
            v = paddle.concat(x=[v, bias_v.tile(repeat_times=[1, bsz, 1])])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert bias_k is None
            assert bias_v is None
        q = q.reshape([tgt_len, bsz * num_heads, head_dim]).transpose([1, 0, 2])

        if static_k is None:
            # 如果没有静态 k，重塑并转置 k
            k = k.reshape([k.shape[0], bsz * num_heads, head_dim]).transpose([1, 0, 2])
        else:
            assert (
                static_k.shape[0] == bsz * num_heads
            ), f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.shape[0]}"
            assert (
                static_k.shape[2] == head_dim
            ), f"expecting static_k.size(2) of {head_dim}, but got {static_k.shape[2]}"
            k = static_k
        if static_v is None:
            v = v.reshape([v.shape[0], bsz * num_heads, head_dim]).transpose([1, 0, 2])
        else:
            assert (
                static_v.shape[0] == bsz * num_heads
            ), f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.shape[0]}"
            assert (
                static_v.shape[2] == head_dim
            ), f"expecting static_v.size(2) of {head_dim}, but got {static_v.shape[2]}"
            v = static_v
        if add_zero_attn:
            zero_attn_shape = bsz * num_heads, 1, head_dim
            k = paddle.concat(x=[k, paddle.zeros(shape=zero_attn_shape, dtype=k.dtype)], axis=1)
            v = paddle.concat(x=[v, paddle.zeros(shape=zero_attn_shape, dtype=v.dtype)], axis=1)
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        src_len = k.shape[1]

        if key_padding_mask is not None:
            assert tuple(key_padding_mask.shape) == (
                bsz,
                src_len,
            ), f"expecting key_padding_mask shape of {bsz, src_len}, but got {tuple(key_padding_mask.shape)}"
            key_padding_mask = (
                key_padding_mask.reshape([bsz, 1, 1, src_len])
                .expand(shape=[-1, num_heads, -1, -1])
                .reshape([bsz * num_heads, 1, src_len])
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask
        if not training:
            dropout_p = 0.0

        if need_weights:
            B, Nt, E = q.shape
            q_scaled = q / math.sqrt(E)

            assert not (is_causal and attn_mask is None), "FIXME: is_causal not implemented for need_weights"

            # 计算注意力权重
            if attn_mask is not None:
                # 使用 matmul + add 替代 baddbmm
                q_k = paddle.matmul(q_scaled, k.transpose([0, 2, 1]))
                attn_output_weights = paddle.add(attn_mask, q_k)
            else:
                attn_output_weights = paddle.matmul(q_scaled, k.transpose([0, 2, 1]))

            # 应用 softmax
            attn_output_weights = paddle.nn.functional.softmax(attn_output_weights, axis=-1)

            # 应用 dropout
            if dropout_p > 0.0:
                attn_output_weights = paddle.nn.functional.dropout(
                    attn_output_weights, p=dropout_p, training=self.training, mode="upscale_in_train"
                )

            # 计算注意力输出
            attn_output = paddle.matmul(attn_output_weights, v)

            # 重塑和转置操作
            attn_output = attn_output.transpose([1, 0, 2])  # transpose(0, 1)
            attn_output = attn_output.reshape([tgt_len * bsz, embed_dim])
            attn_output = self.out_proj(attn_output)
            attn_output = attn_output.reshape([tgt_len, bsz, attn_output.shape[1]])

            # 重塑注意力权重并可选地在多个头上取平均
            attn_output_weights = attn_output_weights.reshape([bsz, num_heads, tgt_len, src_len])
            if average_attn_weights:
                attn_output_weights = paddle.mean(attn_output_weights, axis=1)

            # 处理非批次输入的情况
            if not is_batched:
                # 如果输入是非批次的，压缩输出
                attn_output = paddle.squeeze(attn_output, axis=1)
                attn_output_weights = paddle.squeeze(attn_output_weights, axis=0)

            return attn_output, attn_output_weights

        else:
            if attn_mask is not None:
                if attn_mask.shape[0] == 1 and attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(axis=0)
                else:
                    attn_mask = attn_mask.reshape([bsz, num_heads, -1, src_len])
            q = q.reshape([bsz, num_heads, tgt_len, head_dim])
            k = k.reshape([bsz, num_heads, src_len, head_dim])
            v = v.reshape([bsz, num_heads, src_len, head_dim])
            attn_output = nn.functional.scaled_dot_product_attention(
                query=q, key=k, value=v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
            )
            attn_output = attn_output.transpose([2, 0, 1, 3]).contiguous().reshape([bsz * tgt_len, embed_dim])
            attn_output = self.out_proj(attn_output)
            attn_output = attn_output.reshape([tgt_len, bsz, attn_output.shape[1]])
            if not is_batched:
                attn_output = attn_output.squeeze(axis=1)
            return attn_output, None


def _mha_shape_check(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    key_padding_mask: Optional[paddle.Tensor],
    attn_mask: Optional[paddle.Tensor],
    num_heads: int,
):
    if query.dim() == 3:
        is_batched = True
        assert (
            key.dim() == 3 and value.dim() == 3
        ), f"For batched (3-D) `query`, expected `key` and `value` to be 3-D but found {key.dim()}-D and {value.dim()}-D tensors respectively"
        if key_padding_mask is not None:
            assert (
                key_padding_mask.dim() == 2
            ), f"For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D but found {key_padding_mask.dim()}-D tensor instead"
        if attn_mask is not None:
            assert attn_mask.dim() in (
                2,
                3,
            ), f"For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D but found {attn_mask.dim()}-D tensor instead"
    elif query.dim() == 2:
        is_batched = False
        assert (
            key.dim() == 2 and value.dim() == 2
        ), f"For unbatched (2-D) `query`, expected `key` and `value` to be 2-D but found {key.dim()}-D and {value.dim()}-D tensors respectively"
        if key_padding_mask is not None:
            assert (
                key_padding_mask.dim() == 1
            ), f"For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D but found {key_padding_mask.dim()}-D tensor instead"
        if attn_mask is not None:
            assert attn_mask.dim() in (
                2,
                3,
            ), f"For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D but found {attn_mask.dim()}-D tensor instead"
            if attn_mask.dim() == 3:
                expected_shape = num_heads, tuple(query.shape)[0], tuple(key.shape)[0]
                assert (
                    tuple(attn_mask.shape) == expected_shape
                ), f"Expected `attn_mask` shape to be {expected_shape} but got {tuple(attn_mask.shape)}"
    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor"
        )
    return is_batched


def _canonical_mask(
    mask: Optional[paddle.Tensor],
    mask_name: str,
    other_type: Optional[str],
    other_name: str,
    target_type: str,
    check_other: bool = True,
) -> Optional[paddle.Tensor]:
    if mask is not None:
        # 确保mask是Tensor类型
        if not isinstance(mask, paddle.Tensor):
            mask = paddle.to_tensor(mask)

        # 检查数据类型并转换
        _mask_dtype = mask.dtype
        if _mask_dtype not in ["bool", "float32", "float64", "float16"]:
            # 如果不是支持的类型，转换为布尔类型
            mask = paddle.cast(mask, "bool")

        _mask_is_float = paddle.is_floating_point(mask)

        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} is deprecated. Use same type for both instead."
                )

        # 如果不是浮点型，转换为目标类型的浮点tensor
        if not _mask_is_float:
            mask = paddle.zeros_like(mask, dtype=target_type).masked_fill_(mask=mask, value=float("-inf"))

    return mask


def _none_or_dtype(input: Optional[paddle.Tensor]) -> Optional[str]:
    if input is None:
        return None
    elif isinstance(input, paddle.Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or torch.Tensor")


def _in_projection_packed(
    q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor, w: paddle.Tensor, b: Optional[paddle.Tensor] = None
) -> List[paddle.Tensor]:
    """
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.
    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension
        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.shape[-1]
    if k is v:
        if q is k:
            proj = linear(q, w, b)
            # 将proj重塑为(3, E)
            proj = paddle.reshape(proj, [-1, 3, E])
            # 在第一维增加维度
            proj = paddle.unsqueeze(proj, 0)
            # 转置维度
            proj = paddle.transpose(proj, perm=[0, 2, 1])
            # 移除最后一个维度
            proj = paddle.squeeze(proj, axis=-1)
            # 确保内存连续
            proj = paddle.to_tensor(proj, stop_gradient=False)
            return proj[0], proj[1], proj[2]
        else:
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = linear(q, w_q, b_q)
            kv_proj = linear(k, w_kv, b_kv)

            # 将kv_proj重塑为(2, E)
            kv_proj = paddle.reshape(kv_proj, [-1, 2, E])
            # 在第一维增加维度
            kv_proj = paddle.unsqueeze(kv_proj, 0)
            # 转置维度
            kv_proj = paddle.transpose(kv_proj, perm=[0, 2, 1])
            # 移除最后一个维度
            kv_proj = paddle.squeeze(kv_proj, axis=-1)
            # 确保内存连续
            kv_proj = paddle.to_tensor(kv_proj, stop_gradient=False)

            return q_proj, kv_proj[0], kv_proj[1]
    else:
        w_q, w_k, w_v = w.chunk(chunks=3, axis=-1)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(chunks=3, axis=-1)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    w_q: paddle.Tensor,
    w_k: paddle.Tensor,
    w_v: paddle.Tensor,
    b_q: Optional[paddle.Tensor] = None,
    b_k: Optional[paddle.Tensor] = None,
    b_v: Optional[paddle.Tensor] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.
    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`
        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`
    """
    Eq, Ek, Ev = q.shape[-1], k.shape[-1], v.shape[-1]
    assert tuple(w_q.shape) == (Eq, Eq), f"expecting query weights shape of {Eq, Eq}, but got {tuple(w_q.shape)}"
    assert tuple(w_k.shape) == (Eq, Ek), f"expecting key weights shape of {Eq, Ek}, but got {tuple(w_k.shape)}"
    assert tuple(w_v.shape) == (Eq, Ev), f"expecting value weights shape of {Eq, Ev}, but got {tuple(w_v.shape)}"
    assert b_q is None or tuple(b_q.shape) == (Eq,), f"expecting query bias shape of {Eq,}, but got {tuple(b_q.shape)}"
    assert b_k is None or tuple(b_k.shape) == (Eq,), f"expecting key bias shape of {Eq,}, but got {tuple(b_k.shape)}"
    assert b_v is None or tuple(b_v.shape) == (Eq,), f"expecting value bias shape of {Eq,}, but got {tuple(b_v.shape)}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

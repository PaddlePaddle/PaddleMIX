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

import paddle
import triton.language as tl
from paddle import _C_ops
from paddle.base.framework import OpProtoHolder
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode

from .triton_utils import get_dtype_str, paddle_use_triton, rendering_common_template


@paddle_use_triton(
    key=["1"],
)
def ln_partial_rotary_emb_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    q_norm_weight_ptr,
    q_norm_bias_ptr,
    k_norm_weight_ptr,
    k_norm_bias_ptr,
    outq_ptr,
    outk_ptr,
    text_seq_length,
    batch,
    num_heads,
    seq_len,
    n_elements,
    norm_eps,
    HEAD_DIM: tl.constexpr,
):
    # 计算当前线程处理的元素范围
    b_pid = tl.program_id(axis=0)
    h_pid = tl.program_id(axis=1)
    s_pid = tl.program_id(axis=2)

    block_start = b_pid * num_heads * seq_len * HEAD_DIM + h_pid * seq_len * HEAD_DIM + s_pid * HEAD_DIM
    read_offsets = block_start + tl.arange(0, HEAD_DIM)
    mask = read_offsets < n_elements
    q = tl.load(q_ptr + read_offsets, mask=mask)
    k = tl.load(k_ptr + read_offsets, mask=mask)

    # qk layernorm
    offs = tl.arange(0, HEAD_DIM)
    masks = offs < HEAD_DIM
    q_mean = tl.sum(q) / HEAD_DIM
    q_var = tl.sum(q * q) / HEAD_DIM - q_mean * q_mean
    q_rstd = 1 / tl.sqrt(q_var + norm_eps)
    q_resi_hat = (q - q_mean) * q_rstd
    q_weights = tl.load(q_norm_weight_ptr + offs, mask=masks)
    q_resi_hat = q_resi_hat * q_weights
    q_bias = tl.load(q_norm_bias_ptr + offs, mask=masks)
    q_resi_hat = q_resi_hat + q_bias

    k_mean = tl.sum(k, axis=0) / HEAD_DIM
    k_var = tl.sum(k * k, axis=0) / HEAD_DIM - k_mean * k_mean
    k_rstd = 1 / tl.sqrt(k_var + norm_eps)
    k_resi_hat = (k - k_mean) * k_rstd
    k_weights = tl.load(k_norm_weight_ptr + offs, mask=masks)
    k_resi_hat = k_resi_hat * k_weights
    k_bias = tl.load(k_norm_bias_ptr + offs, mask=masks)
    k_resi_hat = k_resi_hat + k_bias

    # qk rotary_emb
    if s_pid > text_seq_length - 1:
        q1, q2 = tl.split(tl.reshape(q_resi_hat, (32, 2)))
        qc = tl.interleave(-q2, q1)

        k1, k2 = tl.split(tl.reshape(k_resi_hat, (32, 2)))
        kc = tl.interleave(-k2, k1)

        block_cs_start = (s_pid - text_seq_length) * HEAD_DIM
        read_cs_offsets = block_cs_start + tl.arange(0, HEAD_DIM)
        cs_mask = read_cs_offsets < ((seq_len - text_seq_length) * HEAD_DIM)
        cos = tl.load(cos_ptr + read_cs_offsets, mask=cs_mask)
        sin = tl.load(sin_ptr + read_cs_offsets, mask=cs_mask)

        oq = tl.cast(tl.cast(q_resi_hat, tl.float32) * cos + tl.cast(qc, tl.float32) * sin, tl.float16)
        ok = tl.cast(tl.cast(k_resi_hat, tl.float32) * cos + tl.cast(kc, tl.float32) * sin, tl.float16)

        tl.store(outq_ptr + read_offsets, oq, mask=mask)
        tl.store(outk_ptr + read_offsets, ok, mask=mask)
    else:
        tl.store(outq_ptr + read_offsets, q_resi_hat, mask=mask)
        tl.store(outk_ptr + read_offsets, k_resi_hat, mask=mask)


def ln_partial_rotary_emb(
    q,
    k,
    text_seq_length_tensor,
    cos,
    sin,
    q_norm_weight,
    q_norm_bias,
    k_norm_weight,
    k_norm_bias,
    norm_eps=1e-5,
):
    batch = q.shape[0]
    num_heads = q.shape[1]
    seq_len = q.shape[2]
    HEAD_DIM = q.shape[3]
    text_seq_length = text_seq_length_tensor.shape[0]
    n_elements = batch * num_heads * seq_len * HEAD_DIM

    prepare_attr_for_triton_kernel = """
    // 这个名字必须保证和kernel形式参数一致！
    int batch = q.dims()[0];
    int num_heads = q.dims()[1];
    int seq_len =  q.dims()[2];
    int HEAD_DIM =  q.dims()[3];
    int text_seq_length = text_seq_length_tensor.dims()[0];
    int n_elements = batch * num_heads * seq_len * HEAD_DIM;
    """

    assert HEAD_DIM == 64, "Now,HEAD_DIM is must is 64"
    op_name = "ln_partial_rotary_emb"
    op_name += get_dtype_str(q.dtype)
    op_name += f"_{HEAD_DIM}"
    # 创建输出张量

    ln_partial_rotary_emb_kernel_config = [
        {"num_warps": 4},
    ]
    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        outq = paddle.empty_like(q)
        outk = paddle.empty_like(k)

        prepare_ptr_for_triton_kernel = """
        // 这个名字必须保证和kernel形式参数一致！
        auto q_ptr = get_tensor_ptr(q);
        auto k_ptr = get_tensor_ptr(k);
        auto cos_ptr = get_tensor_ptr(cos);
        auto sin_ptr = get_tensor_ptr(sin);
        auto q_norm_weight_ptr = get_tensor_ptr(q_norm_weight);
        auto q_norm_bias_ptr = get_tensor_ptr(q_norm_bias);
        auto k_norm_weight_ptr = get_tensor_ptr(k_norm_weight);
        auto k_norm_bias_ptr = get_tensor_ptr(k_norm_bias);

        auto outq = paddle::empty(q.shape(), q.dtype(), q.place());
        auto outk = paddle::empty(k.shape(), k.dtype(), k.place());
        auto outq_ptr = get_tensor_ptr(outq);
        auto outk_ptr = get_tensor_ptr(outk);
        """
        return_tensor_names = "outq, outk"

        template_used = rendering_common_template(
            ln_partial_rotary_emb, prepare_attr_for_triton_kernel, prepare_ptr_for_triton_kernel, return_tensor_names
        )

        grid = ("batch", "num_heads", "seq_len")
        ln_partial_rotary_emb_kernel[(op_name, template_used, grid, ln_partial_rotary_emb_kernel_config)](
            q_ptr=q,
            k_ptr=k,
            cos_ptr=cos,
            sin_ptr=sin,
            q_norm_weight_ptr=q_norm_weight,
            q_norm_bias_ptr=q_norm_bias,
            k_norm_weight_ptr=k_norm_weight,
            k_norm_bias_ptr=k_norm_bias,
            outq_ptr=outq,
            outk_ptr=outk,
            text_seq_length=text_seq_length,
            batch=batch,
            num_heads=num_heads,
            seq_len=seq_len,
            n_elements=n_elements,
            norm_eps=norm_eps,
            HEAD_DIM=HEAD_DIM,
        )
    if in_dynamic_or_pir_mode():
        # print(f"== we are in dynamic mode, op_name: {op_name}")
        outs = _C_ops._run_custom_op(
            op_name,
            q,
            k,
            text_seq_length_tensor,
            cos,
            sin,
            q_norm_weight,
            q_norm_bias,
            k_norm_weight,
            k_norm_bias,
            norm_eps,
        )
        return outs[0], outs[1]
    else:
        # print(f"== we are in dynamic to static mode, op_name: {op_name}")
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "q": q,
            "k": k,
            "text_seq_length_tensor": text_seq_length_tensor,
            "cos": cos,
            "sin": sin,
            "q_norm_weight": q_norm_weight,
            "q_norm_bias": q_norm_bias,
            "k_norm_weight": k_norm_weight,
            "k_norm_bias": k_norm_bias,
        }
        attrs = (
            {
                "norm_eps": norm_eps,
            },
        )
        outq = helper.create_variable_for_type_inference(dtype=q.dtype)
        outk = helper.create_variable_for_type_inference(dtype=q.dtype)

        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs=attrs,
            outputs={"outq": outq, "outk": outk},
        )
        return outq, outk

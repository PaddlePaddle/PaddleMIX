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
def partial_rotary_emb_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    outq_ptr,
    outk_ptr,
    text_seq_length,
    batch,
    num_heads,
    seq_len,
    head_dim,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 计算当前线程处理的元素范围
    b_pid = tl.program_id(axis=0)  # grid内哪个Block
    h_pid = tl.program_id(axis=1)
    s_pid = tl.program_id(axis=2)

    block_start = b_pid * num_heads * seq_len * head_dim + h_pid * seq_len * head_dim + s_pid * head_dim
    read_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = read_offsets < n_elements
    even_mask = tl.arange(0, BLOCK_SIZE) % 2 == 0

    q0 = tl.load(q_ptr + read_offsets, mask=mask & even_mask)
    q1 = tl.load(q_ptr + read_offsets + 1, mask=mask & even_mask)
    k0 = tl.load(k_ptr + read_offsets, mask=mask & even_mask)
    k1 = tl.load(k_ptr + read_offsets + 1, mask=mask & even_mask)

    block_cs_start = tl.where(s_pid >= text_seq_length, (s_pid - text_seq_length) * head_dim, 0)
    read_cs_offsets = block_cs_start + tl.arange(0, BLOCK_SIZE)
    cs_mask = read_cs_offsets < ((seq_len - text_seq_length) * head_dim)
    cos0 = tl.load(cos_ptr + read_cs_offsets, mask=cs_mask & even_mask)
    cos1 = tl.load(cos_ptr + read_cs_offsets + 1, mask=cs_mask & even_mask)
    sin0 = tl.load(sin_ptr + read_cs_offsets, mask=cs_mask & even_mask)
    sin1 = tl.load(sin_ptr + read_cs_offsets + 1, mask=cs_mask & even_mask)

    oq0 = tl.where(s_pid >= text_seq_length, (q0.to(tl.float32) * cos0 - q1.to(tl.float32) * sin0).to(tl.float16), q0)
    oq1 = tl.where(s_pid >= text_seq_length, (q1.to(tl.float32) * cos1 + q0.to(tl.float32) * sin1).to(tl.float16), q1)
    ok0 = tl.where(s_pid >= text_seq_length, (k0.to(tl.float32) * cos0 - k1.to(tl.float32) * sin0).to(tl.float16), k0)
    ok1 = tl.where(s_pid >= text_seq_length, (k1.to(tl.float32) * cos1 + k0.to(tl.float32) * sin1).to(tl.float16), k1)

    tl.store(outq_ptr + read_offsets, oq0, mask=mask & even_mask)
    tl.store(outq_ptr + read_offsets + 1, oq1, mask=mask & even_mask)
    tl.store(outk_ptr + read_offsets, ok0, mask=mask & even_mask)
    tl.store(outk_ptr + read_offsets + 1, ok1, mask=mask & even_mask)


def partial_rotary_emb(
    q,
    k,
    text_seq_length_tensor,
    cos,
    sin,
):
    batch = q.shape[0]
    num_heads = q.shape[1]
    seq_len = q.shape[2]
    head_dim = q.shape[3]
    text_seq_length = text_seq_length_tensor.shape[0]
    n_elements = batch * num_heads * seq_len * head_dim

    prepare_attr_for_triton_kernel = """
    // 这个名字必须保证和kernel形式参数一致！
    int batch = q.dims()[0];
    int num_heads = q.dims()[1];
    int seq_len =  q.dims()[2];
    int head_dim =  q.dims()[3];
    int text_seq_length = text_seq_length_tensor.dims()[0];
    int n_elements = batch * num_heads * seq_len * head_dim;
    """

    assert head_dim == 64, "Now,head_dim is must is 64"
    BLOCK_SIZE = head_dim
    op_name = "partial_rotary_emb"
    op_name += get_dtype_str(q.dtype)
    op_name += f"_{BLOCK_SIZE}"
    # 创建输出张量

    partial_rotary_emb_kernel_config = [
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

        auto outq = paddle::empty(q.shape(), q.dtype(), q.place());
        auto outk = paddle::empty(k.shape(), k.dtype(), k.place());
        auto outq_ptr = get_tensor_ptr(outq);
        auto outk_ptr = get_tensor_ptr(outk);
        """
        return_tensor_names = "outq, outk"

        template_used = rendering_common_template(
            partial_rotary_emb, prepare_attr_for_triton_kernel, prepare_ptr_for_triton_kernel, return_tensor_names
        )

        grid = ("batch", "num_heads", "seq_len")
        partial_rotary_emb_kernel[(op_name, template_used, grid, partial_rotary_emb_kernel_config)](
            q_ptr=q,
            k_ptr=k,
            cos_ptr=cos,
            sin_ptr=sin,
            outq_ptr=outq,
            outk_ptr=outk,
            text_seq_length=text_seq_length,
            batch=batch,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
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
        }
        outq = helper.create_variable_for_type_inference(dtype=q.dtype)
        outk = helper.create_variable_for_type_inference(dtype=q.dtype)

        helper.append_op(
            type=op_name,
            inputs=inputs,
            outputs={"out0_tensor": outq, "out1_tensor": outk},
        )
        return outq, outk

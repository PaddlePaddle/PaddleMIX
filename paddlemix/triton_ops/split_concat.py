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

import os

import paddle
import triton
import triton.language as tl
from paddle import _C_ops
from paddle.base.framework import OpProtoHolder
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode

from .triton_utils import get_dtype_str, paddle_use_triton, rendering_common_template


@paddle_use_triton(
    key=["1"],
)
def split_concat_kernel(
    out0,
    out1,
    out2,
    qkv,
    eqkv,
    batch,
    seq_qkv,
    seq_eqkv,
    output_hidden,
    BLOCK_SIZE: tl.constexpr,
):
    out_id = tl.program_id(axis=0)
    batch = tl.program_id(axis=1)
    out_row = tl.program_id(axis=2)
    if out_row < seq_qkv:
        read_ptr = out_id * output_hidden + out_row * 3 * output_hidden + batch * seq_qkv * output_hidden * 3 + qkv
    else:
        read_ptr = (
            out_id * output_hidden
            + (out_row - seq_qkv) * 3 * output_hidden
            + batch * seq_eqkv * output_hidden * 3
            + eqkv
        )

    read_offsets = tl.arange(0, BLOCK_SIZE)
    mask = read_offsets < output_hidden
    read_data = tl.load(read_ptr + read_offsets, mask=mask)

    real_output = out0
    if out_id == 1:
        real_output = out1
    elif out_id == 2:
        real_output = out2

    write_ptr = batch * (seq_qkv + seq_eqkv) * output_hidden + out_row * output_hidden + real_output + read_offsets

    tl.store(write_ptr, read_data, mask=mask)


# For Paddle-IR, we need to specify the shape of the output tensor.
# This is not required for Paddle-PIR.
d2s_split_concat_infer_shape_dtype = """
std::vector<std::vector<int64_t>> ${op_name}_InferShape(
    const std::vector<int64_t>& A_shape, const std::vector<int64_t>& B_shape) {
        int64_t seq1 = A_shape[1];
        int64_t seq2 = B_shape[1];
        int64_t seq = -1;
        if (seq1 > 0 && seq2 > 0){
            seq = seq1 + seq2;
    }
    std::vector<int64_t> out_shape = {A_shape[0], seq, A_shape[2]/3};
    return {out_shape, out_shape, out_shape};
}

std::vector<paddle::DataType> ${op_name}_InferDtype(const paddle::DataType& A_dtype) {
    return {A_dtype, A_dtype, A_dtype};
}
"""


def split_concat(x, y):
    assert len(x.shape) == 3
    assert len(y.shape) == 3

    assert x.shape[0] == y.shape[0]
    assert x.shape[2] == y.shape[2]

    # baseline.
    if os.getenv("INFERENCE_OPTIMIZE_TRITON") is None:
        q, k, v = paddle.split(x, 3, axis=-1)
        eq, ek, ev = paddle.split(y, 3, axis=-1)
        q = paddle.concat([q, eq], axis=1)
        k = paddle.concat([k, ek], axis=1)
        v = paddle.concat([v, ev], axis=1)
        return q, k, v

    batch = x.shape[0]
    seq_qkv = x.shape[1]
    hidd_x = x.shape[2]
    seq_eqkv = y.shape[1]
    output_hidden = hidd_x // 3

    prepare_attr_for_triton_kernel = """
    int batch = x.dims()[0];
    int seq_qkv = x.dims()[1];
    int hidd_x = x.dims()[2];
    int seq_eqkv = y.dims()[1];
    int output_hidden = hidd_x / 3;
    """

    BLOCK_SIZE = triton.next_power_of_2(output_hidden)
    op_name = "split_concat"
    op_name += get_dtype_str(x.dtype)
    op_name += f"_{BLOCK_SIZE}"

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        out0 = paddle.empty(shape=[batch, seq_qkv + seq_eqkv, output_hidden], dtype=x.dtype)
        out1 = paddle.empty(shape=[batch, seq_qkv + seq_eqkv, output_hidden], dtype=x.dtype)
        out2 = paddle.empty(shape=[batch, seq_qkv + seq_eqkv, output_hidden], dtype=x.dtype)

        prepare_ptr_for_triton_kernel = """
        auto out0_tensor = paddle::empty({batch, seq_qkv+seq_eqkv, output_hidden}, x.dtype(), x.place());
        auto out1_tensor = paddle::empty({batch, seq_qkv+seq_eqkv, output_hidden}, x.dtype(), x.place());
        auto out2_tensor = paddle::empty({batch, seq_qkv+seq_eqkv, output_hidden}, x.dtype(), x.place());
        CUdeviceptr input_ptrs[5] = {
            get_tensor_ptr(out0_tensor),
            get_tensor_ptr(out1_tensor),
            get_tensor_ptr(out2_tensor),
            get_tensor_ptr(x),
            get_tensor_ptr(y),
        };
        """
        return_tensor_names = "out0_tensor,out1_tensor,out2_tensor"

        template_used = rendering_common_template(
            split_concat,
            prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel,
            return_tensor_names,
            d2s_split_concat_infer_shape_dtype,
        )

        grid = ("3", "batch", "seq_qkv + seq_eqkv")
        # -1 means this value does not matter for triton compilation
        split_concat_kernel[(op_name, template_used, grid)](
            out0=out0,
            out1=out1,
            out2=out2,
            qkv=x,
            eqkv=y,
            batch=-1,
            seq_qkv=seq_qkv,
            seq_eqkv=seq_eqkv,
            output_hidden=output_hidden,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    if in_dynamic_or_pir_mode():
        # print(f"== we are in dynamic mode, op_name: {op_name}")
        outs = _C_ops._run_custom_op(
            op_name,
            x,
            y,
        )
        return outs[0], outs[1], outs[2]
    else:
        # print(f"== we are in dynamic to static mode, op_name: {op_name}")
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "x": x,
            "y": y,
        }
        out0 = helper.create_variable_for_type_inference(dtype=x.dtype)
        out1 = helper.create_variable_for_type_inference(dtype=x.dtype)
        out2 = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type=op_name,
            inputs=inputs,
            outputs={"out0_tensor": out0, "out1_tensor": out1, "out2_tensor": out2},
        )
        return out0, out1, out2

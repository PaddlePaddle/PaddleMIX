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
import triton
import triton.language as tl
from paddle import _C_ops
from paddle.base.framework import OpProtoHolder
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode

from .triton_utils import get_dtype_str, paddle_use_triton, rendering_common_template


@paddle_use_triton(
    key=["M"],
)
def rms_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    M,
    N,
    epsilon,
    BLOCK_SIZE_M: tl.constexpr,
    N_npo2: tl.constexpr,
    weight_attr: tl.constexpr,
    bias_attr: tl.constexpr,
):
    row = tl.program_id(axis=0)

    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_an = tl.arange(0, N_npo2)

    # compute var
    all_offs = (row * BLOCK_SIZE_M + offs_am[:, None]) % M * N + offs_an[None, :]

    x_eles = tl.load(x_ptr + all_offs, mask=offs_an[None, :] < N, other=0.0).to(tl.float32)
    var = tl.sum(x_eles * x_eles, axis=1) / N

    resi_hat = x_eles / tl.sqrt(var[:, None] + epsilon)

    if weight_attr:
        weights = tl.load(weight_ptr + offs_an, mask=offs_an < N, other=0.0)
        resi_hat = resi_hat * weights

    if bias_attr:
        bias = tl.load(bias_ptr + offs_an, mask=offs_an < N, other=0.0)
        resi_hat = resi_hat + bias

    tl.store(y_ptr + all_offs, resi_hat, mask=offs_an[None, :] < N)


def rms_norm(x, weight=None, bias=None, epsilon=1e-5):
    """
    Examples:

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    import paddle
    import paddlemix

    batch = 2
    seq = 3600
    num_heads = 1
    head_dim = 64*30
    dtype= "float16"
    x = paddle.rand([batch, seq, num_heads, head_dim], dtype=dtype)
    weight = paddle.rand([head_dim], dtype=dtype)
    bias = paddle.rand([head_dim], dtype=dtype)

    for i in range(100):
        baseline = paddle.incubate.nn.functional.fused_rms_norm(x, weight, bias, 1e-5, begin_norm_axis=3)

    for i in range(100):
        mt_result = paddlemix.triton_ops.rms_norm(x,weight,bias,1e-5)


    baseline = baseline[0]
    print(paddle.max(paddle.abs(baseline-mt_result)))

    """

    assert len(x.shape) == 4, "x should be 4-dim."
    weight_attr = 0
    if weight is not None:
        assert len(weight.shape) == 1, "weight should be 1-dim"
        assert weight.shape[-1] == x.shape[-1], "x and weight should have same shape[-1]"
        weight_attr = 1
    bias_attr = 0
    if bias is not None:
        assert len(bias.shape) == 1, "bias should be 1-dim"
        assert bias.shape[-1] == x.shape[-1], "x and bias should have same shape[-1]"
        bias_attr = 1

    M = x.shape[0] * x.shape[1] * x.shape[2]
    N = x.shape[3]
    N_npo2 = triton.next_power_of_2(N)

    prepare_attr_for_triton_kernel = """
    int M = x.dims()[0] * x.dims()[1] * x.dims()[2];
    int N = x.dims()[3];
    """

    op_name = f"triton_rms_norm_{get_dtype_str(x.dtype)}_{N_npo2}"

    rms_norm_kernel_config = []
    if N_npo2 <= 64:
        rms_norm_kernel_config.append({"BLOCK_SIZE_M": 4, "num_warps": 1})
    else:
        rms_norm_kernel_config.append({"BLOCK_SIZE_M": 1, "num_warps": 4})

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        y = paddle.empty_like(x)
        prepare_ptr_for_triton_kernel = """
        auto y = paddle::empty(x.shape(), x.dtype(), x.place());
        CUdeviceptr input_ptrs[4] = {
            get_tensor_ptr(x),
            get_tensor_ptr(y),
            (CUdeviceptr)(nullptr),
            (CUdeviceptr)(nullptr)
        };
        if (weight) input_ptrs[2] = get_tensor_ptr(*weight);
        if (bias) input_ptrs[3] = get_tensor_ptr(*bias);
        """
        return_tensor_names = "y"

        template_used = rendering_common_template(
            rms_norm, prepare_attr_for_triton_kernel, prepare_ptr_for_triton_kernel, return_tensor_names
        )

        grid = ("((M+BLOCK_SIZE_M-1)/BLOCK_SIZE_M)",)
        rms_norm_kernel[(op_name, template_used, grid, rms_norm_kernel_config)](
            x_ptr=x,
            y_ptr=y,
            weight_ptr=weight,
            bias_ptr=x,
            M=-1,
            N=N,
            epsilon=epsilon,
            N_npo2=N_npo2,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
        )

    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(op_name, x, weight, bias, epsilon)
        return outs[0]
    else:
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "x": x,
            "weight@OPTIONAL": weight,
            "bias@OPTIONAL": bias,
        }
        attrs = {"epsilon": epsilon}
        y = helper.create_variable_for_type_inference(dtype=x.dtype)
        outputs = {"y": y}
        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
        )
        return y

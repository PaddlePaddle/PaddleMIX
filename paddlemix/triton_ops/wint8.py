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

from .triton_utils import (
    get_op_name_with_suffix,
    paddle_use_triton,
    rendering_common_template,
)


def get_wint8_kernel_config():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32, 64, 128]:
            for block_n in [64, 128, 256]:
                for block_k in [64, 128, 256]:
                    for split_k in [1, 2, 4, 8]:
                        num_warps = 4
                        if block_m * block_n >= 128 * 256:
                            num_warps = 8
                        configs.append(
                            {
                                "SPLIT_K": split_k,
                                "BLOCK_SIZE_M": block_m,
                                "BLOCK_SIZE_N": block_n,
                                "BLOCK_SIZE_K": block_k,
                                "GROUP_SIZE_M": 8,
                                "num_stages": num_stages,
                                "num_warps": num_warps,
                            }
                        )
    return configs


d2s_infer_code = """
std::vector<std::vector<int64_t>> ${op_name}_InferShape(const std::vector<int64_t>& a_shape,
                                                        const std::vector<int64_t>& b_shape,
                                                        const std::vector<int64_t>& c_shape,
                                                        const std::vector<int64_t>& d_shape,
                                                        bool bool_trans_w) {
    if (bool_trans_w) {
        return {{a_shape[0], b_shape[0]}};
    } else {
        return {{a_shape[0], b_shape[1]}};
    }
}

"""

wint8_kernel_other_config = {
    "reset_zero_when_tune": "cudaMemset((void*)c_ptr, 0, sizeof(phi::dtype::float16) * M * N);"
}


@paddle_use_triton(
    other_config=wint8_kernel_other_config,
    key=["M", "N", "K"],
)
def wint8_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bs_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """
    assert K % (BLOCK_SIZE_K * SPLIT_K) == 0
    """

    pid = tl.program_id(axis=0)
    pid_sp_k = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # col major mapping
    # pid_m = pid // num_pid_n
    # pid_n = pid % num_pid_n

    # row major mapping
    # pid_m = pid % num_pid_m
    # pid_n = pid // num_pid_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    # offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_SIZE_K), BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    magic_number = 0x00006400
    magic_number = magic_number.to(tl.uint16)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # a = tl.load(a_ptrs, mask=offs_am[:, None] < M, other=0.0)
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        # fp_b = b.to(tl.float16)

        fp_b = b | magic_number
        fp_b = fp_b.to(tl.float16, bitcast=True)
        fp_b = fp_b - 1152

        bs_ptrs = bs_ptr + offs_bn[None, :]
        bs = tl.load(bs_ptrs)
        fp_b = fp_b * bs

        accumulator += tl.dot(a, fp_b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    # only let the first block do epilogue
    if bias_ptr is not None and pid_sp_k == 0:
        bias_ptrs = bias_ptr + offs_bn
        bias = tl.load(bias_ptrs)
        accumulator += bias[None, :]

    c = accumulator.to(tl.float16)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


def weight_only_int8(x, qweight, scales, bias=None, bool_trans_w=True):
    """
    Examples:

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    import paddle
    from paddle.nn.quant import weight_quantize, weight_only_linear

    M = 16
    N = 4096
    K = 4096*4

    activation = paddle.randn((M, K), dtype=paddle.float16)
    original_weight = paddle.randn((K, N), dtype=paddle.float16)
    bias = paddle.rand((N,), dtype=paddle.float16) * 10
    triton_scale = paddle.max(paddle.abs(original_weight), axis=0) / 127

    perm_qweight, scale = weight_quantize(original_weight, algo="weight_only_int8")

    assert paddle.max(triton_scale - scale) == 0

    # 下面是paddle的cutlass代码
    import datetime
    for i in range(100):
        paddle_cutlass_output = weight_only_linear(activation, perm_qweight, bias, scale)

    paddle.device.synchronize()
    starttime = datetime.datetime.now()
    for i in range(100):
        paddle_cutlass_output = weight_only_linear(activation, perm_qweight, bias, scale)
    paddle.device.synchronize()
    endtime = datetime.datetime.now()
    duringtime = endtime - starttime
    time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
    print("paddle cutlass The whoel end to end time : ", time_ms, "ms")

    # 下面是triton的计算代码
    bool_trans_w_triton = False

    triton_qweight = original_weight / triton_scale.reshape([1, N])
    triton_qweight = paddle.round(triton_qweight)
    triton_qweight = paddle.clip(triton_qweight, min=-127, max=127)
    triton_qweight = triton_qweight.astype("int8")

    if bool_trans_w_triton:
        triton_qweight = triton_qweight.transpose([1,0]).contiguous()

    assert activation.is_contiguous()
    assert triton_qweight.is_contiguous()
    assert scale.is_contiguous()
    triton_uint_qweight = (triton_qweight.astype("int32") + 128).astype("uint8")

    for i in range(100):
        triton_output = paddlemix.triton_ops.weight_only_int8(
            activation,
            triton_uint_qweight,
            triton_scale,
            bias, bool_trans_w=bool_trans_w_triton)

    paddle.device.synchronize()

    starttime = datetime.datetime.now()
    for i in range(100):
        triton_output = paddlemix.triton_ops.weight_only_int8(
            activation,
            triton_uint_qweight,
            triton_scale,
            bias,
            bool_trans_w = bool_trans_w_triton)
    paddle.device.synchronize()
    endtime = datetime.datetime.now()
    duringtime = endtime - starttime
    time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
    print("triton The whoel end to end time : ", time_ms, "ms")

    if bool_trans_w_triton:
        triton_qweight = triton_qweight.transpose([1,0]).contiguous()

    for i in range(100):
        dequantized_weight = triton_qweight.astype("float16") * scale.reshape([1, N])
        baseline = paddle.matmul(activation, dequantized_weight)
        baseline += bias

    paddle.device.synchronize()
    starttime = datetime.datetime.now()

    for i in range(100):
        dequantized_weight = triton_qweight.astype("float16") * scale.reshape([1, N])
        baseline = paddle.matmul(activation, dequantized_weight)
        baseline += bias
    paddle.device.synchronize()
    endtime = datetime.datetime.now()
    duringtime = endtime - starttime
    time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
    print("baseline The whoel end to end time : ", time_ms, "ms")

    print("triton and baseline max diff", paddle.max(paddle.abs(triton_output - baseline)))
    print("triton and cutlass max diff", paddle.max(paddle.abs(triton_output - paddle_cutlass_output)))
    """

    M, K = x.shape
    if bool_trans_w:
        N = qweight.shape[0]
        stride_bk = 1
        stride_bn = K
    else:
        N = qweight.shape[1]
        stride_bk = N
        stride_bn = 1
    stride_am = K
    stride_ak = 1  # A always is rowmajor
    stride_cm = N
    stride_cn = 1  # C always is rowmajor

    prepare_attr_for_triton_kernel = """
    int M = x.dims()[0];
    int K = x.dims()[1];
    int N = -1;
    int stride_bk = -1;
    int stride_bn = -1;
    
    if (bool_trans_w) {
        N = qweight.dims()[0];
        stride_bk = 1;
        stride_bn = K;
    } else {
        N = qweight.dims()[1];
        stride_bk = N;
        stride_bn = 1;
    }

    int stride_am = K;
    int stride_ak = 1;
    int stride_cm = N;
    int stride_cn = 1;
    """

    op_name = "triton_wint8"
    if bool_trans_w:
        op_name = "triton_wint8_trans"

    # -1 means this value does not matter for triton compilation
    x_list = [-1, N, K, K, 1, stride_bk, stride_bn, N, 1]

    op_name = get_op_name_with_suffix(op_name, x_list)

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        assert x.is_contiguous(), ""
        assert qweight.is_contiguous(), ""

        # below code is register this kernel, will not run this kernel.
        output = paddle.zeros((M, N), dtype=x.dtype)

        prepare_ptr_for_triton_kernel = """
        auto output = paddle::full({M,N}, 0, x.dtype(), x.place());
        auto c_ptr = get_tensor_ptr(output);
        CUdeviceptr input_ptrs[5] = {
            get_tensor_ptr(x),
            get_tensor_ptr(qweight),
            get_tensor_ptr(output),
            get_tensor_ptr(scales),
            (CUdeviceptr)(nullptr)
        };
        if (bias) input_ptrs[4] = get_tensor_ptr(*bias);
        """

        return_tensor_names = "output"
        template_used = rendering_common_template(
            weight_only_int8,
            prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel,
            return_tensor_names,
            d2s_infer_code,
        )

        grid = (
            "((M+BLOCK_SIZE_M-1)/BLOCK_SIZE_M) * ((N+BLOCK_SIZE_N-1)/BLOCK_SIZE_N)",
            "SPLIT_K",
        )

        wint8_kernel[(op_name, template_used, grid, get_wint8_kernel_config())](
            a_ptr=x,
            b_ptr=qweight,
            c_ptr=output,
            bs_ptr=scales,
            bias_ptr=bias,
            M=M,
            N=N,
            K=K,
            stride_am=stride_am,
            stride_ak=stride_ak,
            stride_bk=stride_bk,
            stride_bn=stride_bn,
            stride_cm=stride_cm,
            stride_cn=stride_cn,
        )

    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(op_name, x, qweight, scales, bias, bool_trans_w)
        return outs[0]
    else:
        helper = LayerHelper(op_name, **locals())

        inputs = {
            "x": x,
            "qweight": qweight,
            "scales": scales,
            "bias@OPTIONAL": bias,
        }
        attrs = {"bool_trans_w": bool_trans_w}
        output = helper.create_variable_for_type_inference(dtype=x.dtype)
        outputs = {"output": output}

        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
        )
        return output

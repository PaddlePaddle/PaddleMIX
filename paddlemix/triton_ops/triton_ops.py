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


import os

import paddle
import triton
import triton.language as tl
from paddle import _C_ops
from paddle.base.framework import OpProtoHolder
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode

from .triton_utils import (
    get_dtype_str,
    paddle_use_triton,
    rendering_common_template,
    tune_and_invoke_part,
)


@paddle_use_triton(
    key=["M"],
)
def fused_adaLN_scale_residual_kernel(
    x_ptr,  # input: residual input of attention
    mha_out_ptr,  # input: attention result
    gate_msa_ptr,
    scale_mlp_ptr,
    shift_mlp_ptr,
    weight_ptr,
    bias_ptr,
    resi_out_ptr,  # output: residual result of attention
    adaLN_out_ptr,  # output: adaptive layer norm result
    M,
    N,
    seq_size,
    epsilon,
    N_npo2: tl.constexpr,
    weight_attr: tl.constexpr,
    bias_attr: tl.constexpr,
):
    row = tl.program_id(axis=0)
    mha_out_ptr += row * N
    x_ptr += row * N
    resi_out_ptr += row * N
    adaLN_out_ptr += row * N
    gate_msa_ptr += (row // seq_size) * N
    scale_mlp_ptr += (row // seq_size) * N
    shift_mlp_ptr += (row // seq_size) * N

    all_offs = tl.arange(0, N_npo2)
    all_mask = all_offs < N
    # compute residual
    mha_eles = tl.load(mha_out_ptr + all_offs, mask=all_mask, other=0.0).to(tl.float32)
    x_eles = tl.load(x_ptr + all_offs, mask=all_mask, other=0.0).to(tl.float32)
    gate_msa_eles = tl.load(gate_msa_ptr + all_offs, mask=all_mask, other=0.0)

    _resi_outs = mha_eles * gate_msa_eles + x_eles
    tl.store(resi_out_ptr + all_offs, _resi_outs, mask=all_mask)

    # compute mean var
    mean = tl.sum(_resi_outs, axis=0) / N
    var = tl.sum(_resi_outs * _resi_outs, axis=0) / N - mean * mean
    rstd = 1 / tl.sqrt(var + epsilon)

    # compute adaLN
    resi_hat = (_resi_outs - mean) * rstd
    if weight_attr:
        weights = tl.load(weight_ptr + all_offs, mask=all_mask, other=0.0)
        resi_hat = resi_hat * weights
    if bias_attr:
        bias = tl.load(bias_ptr + all_offs, mask=all_mask, other=0.0)
        resi_hat = resi_hat + bias
    scales = tl.load(scale_mlp_ptr + all_offs, mask=all_mask, other=0.0)
    shifts = tl.load(shift_mlp_ptr + all_offs, mask=all_mask, other=0.0)
    y = resi_hat * (1 + scales) + shifts
    tl.store(adaLN_out_ptr + all_offs, y, mask=all_mask)


def fused_adaLN_scale_residual(
    x,
    mha_out,
    gate_msa,
    scale_mlp,
    shift_mlp,
    weight=None,
    bias=None,
    epsilon=1e-05,
):
    """
    Examples:

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    import paddle

    batch = 2
    seq = 3600
    hidd = 4096
    dtype= "float16"
    epsilon = 1e-5
    x = paddle.rand([batch, seq, hidd], dtype=dtype)
    mha_out = paddle.rand([batch, seq, hidd], dtype=dtype)
    weight = paddle.rand([hidd], dtype=dtype)
    bias = paddle.rand([hidd], dtype=dtype)

    gate_msa = paddle.rand([batch, hidd], dtype=dtype)
    scale_mlp_x = paddle.rand([batch, hidd], dtype=dtype)
    shift_mlp_x = paddle.rand([batch, hidd], dtype=dtype)


    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(axis=1)) + shift.unsqueeze(axis=1)


    def paddle_fused_adaLN(x, mha_out, gate, hidd, scale, shift, weight, bias, epsilon):
        resi_out_paddle = mha_out * gate.unsqueeze(axis=1) + x
        layer_norm_out_paddle = paddle.nn.functional.layer_norm(resi_out_paddle, [hidd], weight, bias, epsilon)
        adaLN_out_paddle = modulate(layer_norm_out_paddle, shift, scale).to(dtype)
        return resi_out_paddle, adaLN_out_paddle


    for i in range(100):
        resi_out_triton, adaLN_out_triton = paddlemix.triton_ops.fused_adaLN_scale_residual(x, mha_out, gate_msa, scale_mlp_x, shift_mlp_x, weight, bias, epsilon)

    for i in range(100):
        resi_out_paddle, adaLN_out_paddle = paddle_fused_adaLN(x, mha_out, gate_msa, hidd, scale_mlp_x, shift_mlp_x, weight, bias, epsilon)

    print("adaLN_maxdiff: ", paddle.max(paddle.abs(adaLN_out_paddle - adaLN_out_triton)))
    print("resi_maxdiff: ", paddle.max(paddle.abs(resi_out_paddle - resi_out_triton)))
    """

    assert x.shape == mha_out.shape, "x and mha_out should have same shape"
    assert (
        gate_msa.shape == scale_mlp.shape == shift_mlp.shape
    ), "gate_msa, scale_mlp and shift_mlp should have same shape"

    assert len(x.shape) == 3, "x should be 3-dim [batch_size, seq_size, feature_dim]"
    weight_attr = 0
    if weight is not None:
        assert len(weight.shape) == 1, "weight should be 1-dim [feature_dim]"
        assert weight.shape[-1] == x.shape[-1], "x and weight should have same shape[-1] == feature_dim"
        weight_attr = 1
    bias_attr = 0
    if bias is not None:
        assert len(bias.shape) == 1, "bias should be 1-dim [feature_dim]"
        assert bias.shape[-1] == x.shape[-1], "x and bias should have same shape[-1] == feature_dim"
        bias_attr = 1
    assert (
        len(scale_mlp.shape) == 2 and len(shift_mlp.shape) == 2
    ), "scale and shift should be 2-dim [batch_size, feature_dim]"
    assert (
        scale_mlp.shape[0] == shift_mlp.shape[0] == x.shape[0]
    ), "x, scale and shift should have same shape[0] == batch_size"
    assert (
        scale_mlp.shape[1] == shift_mlp.shape[1] == x.shape[-1]
    ), "x, scale and shift should have same shape[-1] == feature_dim"

    M = x.shape[0] * x.shape[1]
    N = x.shape[2]
    seq_size = x.shape[1]
    N_npo2 = triton.next_power_of_2(N)

    prepare_attr_for_triton_kernel = """
    int M = x.dims()[0] * x.dims()[1];
    int N = x.dims()[2];
    int seq_size = x.dims()[1];
    """

    # baseline.
    if os.getenv("INFERENCE_OPTIMIZE_TRITON") is None:
        resi_out_paddle = mha_out * gate_msa.unsqueeze(axis=1) + x
        norm_hidden_states = paddle.nn.functional.layer_norm(resi_out_paddle, [N], weight, bias, epsilon)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return resi_out_paddle, norm_hidden_states

    op_name = "triton_fused_adaLN_scale_residual"
    op_name += get_dtype_str(x.dtype)
    op_name += f"_{N_npo2}_{weight_attr}_{bias_attr}"

    fused_adaLN_scale_residual_kernel_config = [
        {"num_warps": 2},
        {"num_warps": 4},
        {"num_warps": 8},
        {"num_warps": 16},
        {"num_warps": 32},
    ]

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        resi_out = paddle.empty_like(x)
        adaLN_out = paddle.empty_like(x)
        prepare_ptr_for_triton_kernel = """
        auto resi_out = paddle::empty(x.shape(), x.dtype(), x.place());
        auto adaLN_out = paddle::empty(x.shape(), x.dtype(), x.place());

        CUdeviceptr input_ptrs[9] = {
            get_tensor_ptr(x),
            get_tensor_ptr(mha_out),
            get_tensor_ptr(gate_msa),
            get_tensor_ptr(scale_mlp),
            get_tensor_ptr(shift_mlp),
            (CUdeviceptr)(nullptr),
            (CUdeviceptr)(nullptr),
            get_tensor_ptr(resi_out),
            get_tensor_ptr(adaLN_out)
        };
        if (weight) input_ptrs[5] = get_tensor_ptr(*weight);
        if (bias) input_ptrs[6] = get_tensor_ptr(*bias);
        """

        return_tensor_names = "resi_out, adaLN_out"
        template_used = rendering_common_template(
            fused_adaLN_scale_residual,
            prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel,
            return_tensor_names,
        )

        grid = ("M",)
        fused_adaLN_scale_residual_kernel[(op_name, template_used, grid, fused_adaLN_scale_residual_kernel_config)](
            x_ptr=x,
            mha_out_ptr=mha_out,
            gate_msa_ptr=gate_msa,
            scale_mlp_ptr=scale_mlp,
            shift_mlp_ptr=shift_mlp,
            # weight_ptr and bias_ptr may be None, so use shift_mlp.
            weight_ptr=shift_mlp,
            bias_ptr=shift_mlp,
            resi_out_ptr=resi_out,
            adaLN_out_ptr=adaLN_out,
            M=-1,
            N=N,
            seq_size=-1,
            epsilon=epsilon,
            N_npo2=N_npo2,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
        )

    if in_dynamic_or_pir_mode():
        print(f"== we are in dynamic mode, op_name: {op_name}")
        outs = _C_ops._run_custom_op(
            op_name,
            x,
            mha_out,
            gate_msa,
            scale_mlp,
            shift_mlp,
            weight,
            bias,
            epsilon,
        )
        return outs[0], outs[1]
    else:
        print(f"== we are in dynamic to static mode, op_name: {op_name}")
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "x": x,
            "mha_out": mha_out,
            "gate_msa": gate_msa,
            "scale_mlp": scale_mlp,
            "shift_mlp": shift_mlp,
            "weight@OPTIONAL": weight,
            "bias@OPTIONAL": bias,
        }
        resi_out = helper.create_variable_for_type_inference(dtype=x.dtype)
        adaLN_out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs={
                "epsilon": epsilon,
            },
            outputs={"resi_out": resi_out, "adaLN_out": adaLN_out},
        )
        return resi_out, adaLN_out


@paddle_use_triton(
    key=["M"],
)
def adaptive_layer_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    scale_ptr,
    shift_ptr,
    M,
    N,
    seq_size,
    epsilon,
    BLOCK_SIZE: tl.constexpr,
    weight_attr: tl.constexpr,
    bias_attr: tl.constexpr,
):
    row = tl.program_id(axis=0)
    x_ptr += row * N
    y_ptr += row * N
    # Compute mean
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _sum_square = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for col_off in range(0, N, BLOCK_SIZE):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        eles = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        _sum += eles
        _sum_square += eles * eles
    mean = tl.sum(_sum, axis=0) / N
    var = tl.sum(_sum_square, axis=0) / N - mean * mean
    rstd = 1 / tl.sqrt(var + epsilon)
    # Compute output
    scale_ptr += (row // seq_size) * N
    shift_ptr += (row // seq_size) * N
    for col_off in range(0, N, BLOCK_SIZE):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        eles = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (eles - mean) * rstd
        if weight_attr:
            weights = tl.load(weight_ptr + cols, mask=mask, other=0.0)
            x_hat = x_hat * weights
        if bias_attr:
            bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
            x_hat = x_hat + bias
        scales = tl.load(scale_ptr + cols, mask=mask, other=0.0)
        shifts = tl.load(shift_ptr + cols, mask=mask, other=0.0)
        y = x_hat * (1 + scales) + shifts
        tl.store(y_ptr + cols, y, mask=mask)


def adaptive_layer_norm(x, scale, shift, weight=None, bias=None, epsilon=1e-05):
    """
    Examples:

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    import paddle

    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(axis=1)) + shift.unsqueeze(axis=1)

    batch = 2
    seq = 3600
    hidd = 4096
    dtype= "float16"
    x = paddle.rand([batch, seq, hidd], dtype=dtype)
    weight = paddle.rand([hidd], dtype=dtype)
    bias = paddle.rand([hidd], dtype=dtype)

    shift_msa_x = paddle.rand([batch, hidd], dtype=dtype)
    scale_msa_x = paddle.rand([batch, hidd], dtype=dtype)

    for i in range(100):
        mt_result = paddlemix.triton_ops.adaptive_layer_norm(x, scale_msa_x, shift_msa_x, weight, bias)

    for i in range(100):
        baseline = modulate(paddle.nn.functional.layer_norm(x, [hidd], weight, bias, 1e-5), shift_msa_x, scale_msa_x)

    print(paddle.max(paddle.abs(baseline-mt_result)))

    """

    assert len(x.shape) == 3, "x should be 3-dim [batch_size, seq_size, feature_dim]"
    weight_attr = 0
    if weight is not None:
        assert len(weight.shape) == 1, "weight should be 1-dim [feature_dim]"
        assert weight.shape[-1] == x.shape[-1], "x and weight should have same shape[-1] == feature_dim"
        weight_attr = 1
    bias_attr = 0
    if bias is not None:
        assert len(bias.shape) == 1, "bias should be 1-dim [feature_dim]"
        assert bias.shape[-1] == x.shape[-1], "x and bias should have same shape[-1] == feature_dim"
        bias_attr = 1
    assert len(scale.shape) == 2 and len(shift.shape) == 2, "scale and shift should be 2-dim [batch_size, feature_dim]"
    assert scale.shape[0] == shift.shape[0] == x.shape[0], "x, scale and shift should have same shape[0] == batch_size"
    assert (
        scale.shape[1] == shift.shape[1] == x.shape[-1]
    ), "x, scale and shift should have same shape[-1] == feature_dim"

    M = x.shape[0] * x.shape[1]
    N = x.shape[2]
    seq_size = x.shape[1]
    BLOCK_SIZE = triton.next_power_of_2(N)

    prepare_attr_for_triton_kernel = """
    int M = x.dims()[0] * x.dims()[1];
    int N = x.dims()[2];
    int seq_size = x.dims()[1];
    """

    # baseline.
    if os.getenv("INFERENCE_OPTIMIZE_TRITON") is None:
        norm_hidden_states = paddle.nn.functional.layer_norm(x, [N], weight, bias, epsilon)
        norm_hidden_states = norm_hidden_states * (1 + scale[:, None]) + shift[:, None]
        return norm_hidden_states

    op_name = "triton_adaptive_layer_norm"
    op_name += get_dtype_str(x.dtype)
    op_name += f"_{BLOCK_SIZE}_{weight_attr}_{bias_attr}"

    adaptive_layer_norm_kernel_config = [
        {"num_warps": 2},
        {"num_warps": 4},
        {"num_warps": 8},
        {"num_warps": 16},
        {"num_warps": 32},
    ]

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        y = paddle.empty_like(x)
        prepare_ptr_for_triton_kernel = """
        auto y = paddle::empty(x.shape(), x.dtype(), x.place());
        CUdeviceptr input_ptrs[6] = {
            get_tensor_ptr(x),
            get_tensor_ptr(y),
            (CUdeviceptr)(nullptr),
            (CUdeviceptr)(nullptr),
            get_tensor_ptr(scale),
            get_tensor_ptr(shift)
        };
        if (weight) input_ptrs[2] = get_tensor_ptr(*weight);
        if (bias) input_ptrs[3] = get_tensor_ptr(*bias);
        """
        return_tensor_names = "y"
        template_used = rendering_common_template(
            adaptive_layer_norm, prepare_attr_for_triton_kernel, prepare_ptr_for_triton_kernel, return_tensor_names
        )

        grid = ("M",)
        adaptive_layer_norm_kernel[(op_name, template_used, grid, adaptive_layer_norm_kernel_config)](
            x_ptr=x,
            y_ptr=y,
            weight_ptr=y,
            bias_ptr=y,
            scale_ptr=y,
            shift_ptr=y,
            M=-1,
            N=N,
            seq_size=-1,
            epsilon=epsilon,
            BLOCK_SIZE=BLOCK_SIZE,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
        )

    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(op_name, x, scale, shift, weight, bias, epsilon)
        return outs[0]
    else:
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "x": x,
            "scale": scale,
            "shift": shift,
            "weight@OPTIONAL": weight,
            "bias@OPTIONAL": bias,
        }
        y = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs={
                "epsilon": epsilon,
            },
            outputs={"y": y},
        )
        return y


fused_rotary_emb_template = (
    """
std::vector<paddle::Tensor> ${op_name}_func(
  const paddle::Tensor &x,
  const paddle::Tensor &q_norm_weight,
  const paddle::Tensor &q_norm_bias,
  const paddle::Tensor &k_norm_weight,
  const paddle::Tensor &k_norm_bias,
  const paddle::Tensor &freqs_cis,
  float epsilon) {
  int BSZ = x.dims()[0];
  int SEQ_LEN = x.dims()[1];
  int HEAD_DIM = freqs_cis.dims()[2];
  int DIM = q_norm_weight.dims()[0];
  int NUM_HEAD = DIM / HEAD_DIM;
  int M = BSZ * SEQ_LEN;
  int DIM_concat = x.dims()[2];

  auto q_out = paddle::empty({BSZ, SEQ_LEN, NUM_HEAD, HEAD_DIM}, x.dtype(), x.place());
  auto k_out = paddle::empty({BSZ, SEQ_LEN, NUM_HEAD, HEAD_DIM}, x.dtype(), x.place());
  auto v_out = paddle::empty({BSZ, SEQ_LEN, NUM_HEAD, HEAD_DIM}, x.dtype(), x.place());

  auto x_ptr = get_tensor_ptr(x);
  auto q_norm_weight_ptr = get_tensor_ptr(q_norm_weight);
  auto q_norm_bias_ptr = get_tensor_ptr(q_norm_bias);
  auto k_norm_weight_ptr = get_tensor_ptr(k_norm_weight);
  auto k_norm_bias_ptr = get_tensor_ptr(k_norm_bias);
  auto freqs_cis_ptr = get_tensor_ptr(freqs_cis);
  auto q_out_ptr = get_tensor_ptr(q_out);
  auto k_out_ptr = get_tensor_ptr(k_out);
  auto v_out_ptr = get_tensor_ptr(v_out);

  auto run_stream = q_out.stream();
"""
    + tune_and_invoke_part
    + """
    return {q_out, k_out, v_out};
}

std::vector<std::vector<int64_t>> ${op_name}_InferShape(
        const std::vector<int64_t>& A_shape,
        const std::vector<int64_t>& B_shape,
        const std::vector<int64_t>& C_shape,
        const std::vector<int64_t>& D_shape,
        const std::vector<int64_t>& E_shape,
        const std::vector<int64_t>& F_shape) {
  int BSZ = A_shape[0];
  int SEQ_LEN = A_shape[1];
  int HEAD_DIM = F_shape[2];
  int DIM = B_shape[0];
  int NUM_HEAD = DIM / HEAD_DIM;
  std::vector<int64_t> res_shape = {BSZ, SEQ_LEN, NUM_HEAD, HEAD_DIM};
  return {res_shape, res_shape, res_shape};
}

std::vector<paddle::DataType> ${op_name}_InferDtype(const paddle::DataType& A_dtype) {
    return {A_dtype, A_dtype, A_dtype};
}

PD_BUILD_OP(${op_name})
    .Inputs({"x", "q_norm_weight", "q_norm_bias", "k_norm_weight", "k_norm_bias", "freqs_cis"})
    .Outputs({"q_out", "k_out", "v_out"})
    .Attrs({"epsilon: float"})
"""
)


@paddle_use_triton(
    key=["M"],
)
def fused_rotary_emb_kernel(
    x_ptr,  # [BSZ, SEQ_LEN, DIM_concat]
    q_out_ptr,
    k_out_ptr,  # [BSZ, SEQ_LEN, NUM_HEAD, HEAD_DIM, 2]
    v_out_ptr,  # [BSZ, SEQ_LEN, NUM_HEAD, HEAD_DIM]
    q_norm_weight_ptr,
    q_norm_bias_ptr,
    k_norm_weight_ptr,
    k_norm_bias_ptr,  # [DIM]
    freqs_cis_ptr,  # [1, seq_len, 1, head_dim, 2]
    epsilon,
    SEQ_LEN,
    M,
    DIM,
    DIM_concat,
    DIM_npo2: tl.constexpr,
):
    row = tl.program_id(axis=0)
    x_ptr += row * DIM_concat
    offs = tl.arange(0, DIM_npo2)
    masks = offs < DIM
    q_eles = tl.load(x_ptr + offs, mask=masks, other=0.0).to(tl.float32)
    k_eles = tl.load(x_ptr + DIM + offs, mask=masks, other=0.0).to(tl.float32)
    v_eles = tl.load(x_ptr + 2 * DIM + offs, mask=masks, other=0.0)

    # qk layernorm
    q_mean = tl.sum(q_eles, axis=0) / DIM
    q_var = tl.sum(q_eles * q_eles, axis=0) / DIM - q_mean * q_mean
    q_rstd = 1 / tl.sqrt(q_var + epsilon)
    q_resi_hat = (q_eles - q_mean) * q_rstd
    q_weights = tl.load(q_norm_weight_ptr + offs, mask=masks, other=0.0)
    q_resi_hat = q_resi_hat * q_weights
    q_bias = tl.load(q_norm_bias_ptr + offs, mask=masks, other=0.0)
    q_resi_hat = q_resi_hat + q_bias

    k_mean = tl.sum(k_eles, axis=0) / DIM
    k_var = tl.sum(k_eles * k_eles, axis=0) / DIM - k_mean * k_mean
    k_rstd = 1 / tl.sqrt(k_var + epsilon)
    k_resi_hat = (k_eles - k_mean) * k_rstd
    k_weights = tl.load(k_norm_weight_ptr + offs, mask=masks, other=0.0)
    k_resi_hat = k_resi_hat * k_weights
    k_bias = tl.load(k_norm_bias_ptr + offs, mask=masks, other=0.0)
    k_resi_hat = k_resi_hat + k_bias

    # qk rotary_emb
    # freqs_cis = [DIM_npo2, 2]
    freqs_cis_ptr += (row % SEQ_LEN) * DIM * 2
    freqs_offs = tl.arange(0, DIM_npo2 * 2)
    freqs_masks = freqs_offs < DIM * 2
    freqs_cis = tl.load(freqs_cis_ptr + freqs_offs, mask=freqs_masks, other=0.0)
    freqs_cis = tl.reshape(freqs_cis, (DIM_npo2, 2))

    # q_resi_hat = [DIM_npo2] => [DIM_npo2//2, 1, 2]
    q_resi_hat = tl.reshape(q_resi_hat, (DIM_npo2 // 2, 1, 2))
    q_resi_hat = tl.broadcast_to(q_resi_hat, (DIM_npo2 // 2, 2, 2))
    q_resi_hat = tl.reshape(q_resi_hat, (DIM_npo2, 2))
    q_res = tl.sum(q_resi_hat * freqs_cis, axis=1)

    k_resi_hat = tl.reshape(k_resi_hat, (DIM_npo2 // 2, 1, 2))
    k_resi_hat = tl.broadcast_to(k_resi_hat, (DIM_npo2 // 2, 2, 2))
    k_resi_hat = tl.reshape(k_resi_hat, (DIM_npo2, 2))
    k_res = tl.sum(k_resi_hat * freqs_cis, axis=1)

    out_offs = row * DIM + offs
    tl.store(q_out_ptr + out_offs, q_res, mask=masks)
    tl.store(k_out_ptr + out_offs, k_res, mask=masks)
    tl.store(v_out_ptr + out_offs, v_eles, mask=masks)


def fused_rotary_emb(
    x,
    q_norm_weight,
    q_norm_bias,
    k_norm_weight,
    k_norm_bias,
    freqs_cis,
    epsilon=1e-5,
):
    assert x.is_contiguous()
    assert q_norm_weight is not None, "q_norm_weight should not be none"
    assert q_norm_bias is not None, "q_norm_bias should not be none"
    assert k_norm_weight is not None, "k_norm_weight should not be none"
    assert k_norm_bias is not None, "k_norm_bias should not be none"
    DIM = q_norm_weight.shape[0]
    HEAD_DIM = freqs_cis.shape[-2]
    assert (DIM % HEAD_DIM) == 0, "dim should be divisible by head_dim"
    DIM_concat = x.shape[-1]
    assert (DIM * 3) == DIM_concat, "not support GQA, qkv num_head should be equal"

    BSZ = x.shape[0]
    SEQ_LEN = x.shape[1]
    NUM_HEAD = DIM // HEAD_DIM
    M = BSZ * SEQ_LEN
    DIM_npo2 = triton.next_power_of_2(DIM)
    dtype_ = x.dtype

    # q_out_tensor = paddle.empty([BSZ, SEQ_LEN, NUM_HEAD, HEAD_DIM], dtype=dtype_)
    # k_out_tensor = paddle.empty([BSZ, SEQ_LEN, NUM_HEAD, HEAD_DIM], dtype=dtype_)
    # v_out_tensor = paddle.empty([BSZ, SEQ_LEN, NUM_HEAD, HEAD_DIM], dtype=dtype_)
    # fused_rotary_emb_kernel[(M,)](
    #     input_tensor, q_out_tensor, k_out_tensor, v_out_tensor,
    #     q_norm_weight, q_norm_bias, k_norm_weight, k_norm_bias, freqs_cis, epsilon,
    #     SEQ_LEN, M, DIM, DIM_concat,
    #     DIM_npo2, num_warps=4,
    # )
    # return q_out_tensor, k_out_tensor, v_out_tensor

    op_name = "triton_fused_rotary_emb"
    op_name += get_dtype_str(dtype_)
    op_name += f"_{DIM_npo2}"

    fused_rotary_emb_kernel_config = [
        {"num_warps": 2},
        {"num_warps": 4},
        {"num_warps": 8},
        {"num_warps": 16},
        {"num_warps": 32},
    ]

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        empty_dtype = dtype_ if dtype_ != paddle.bfloat16 else paddle.float16
        q_out_tensor = paddle.empty([BSZ, SEQ_LEN, NUM_HEAD, HEAD_DIM], dtype=empty_dtype).astype(dtype_)
        k_out_tensor = paddle.empty([BSZ, SEQ_LEN, NUM_HEAD, HEAD_DIM], dtype=empty_dtype).astype(dtype_)
        v_out_tensor = paddle.empty([BSZ, SEQ_LEN, NUM_HEAD, HEAD_DIM], dtype=empty_dtype).astype(dtype_)
        grid = ("M",)
        fused_rotary_emb_kernel[(op_name, fused_rotary_emb_template, grid, fused_rotary_emb_kernel_config)](
            x,
            q_out_tensor,
            k_out_tensor,
            v_out_tensor,
            q_norm_weight,
            q_norm_bias,
            k_norm_weight,
            k_norm_bias,
            freqs_cis,
            epsilon,
            SEQ_LEN,
            M,
            DIM,
            DIM_concat,
            DIM_npo2,
        )

    if in_dynamic_or_pir_mode():
        print(f"== we are in dynamic mode, op_name: {op_name}")
        outs = _C_ops._run_custom_op(
            op_name,
            x,
            q_norm_weight,
            q_norm_bias,
            k_norm_weight,
            k_norm_bias,
            freqs_cis,
            epsilon,
        )
        return outs[0], outs[1], outs[2]
    else:
        print(f"== we are in dynamic to static mode, op_name: {op_name}")
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "x": x,
            "q_norm_weight": q_norm_weight,
            "q_norm_bias": q_norm_bias,
            "k_norm_weight": k_norm_weight,
            "k_norm_bias": k_norm_bias,
            "freqs_cis": freqs_cis,
        }
        q_out = helper.create_variable_for_type_inference(dtype=dtype_)
        k_out = helper.create_variable_for_type_inference(dtype=dtype_)
        v_out = helper.create_variable_for_type_inference(dtype=dtype_)
        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs={
                "epsilon": epsilon,
            },
            outputs={"q_out": q_out, "k_out": k_out, "v_out": v_out},
        )
        return q_out, k_out, v_out


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


########################### split concat ###############################
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
    ouput_hidden = hidd_x // 3

    prepare_attr_for_triton_kernel = """
    int batch = x.dims()[0];
    int seq_qkv = x.dims()[1];
    int hidd_x = x.dims()[2];
    int seq_eqkv = y.dims()[1];
    int output_hidden = hidd_x / 3;
    """

    BLOCK_SIZE = triton.next_power_of_2(ouput_hidden)
    op_name = "split_concat"
    op_name += get_dtype_str(x.dtype)
    op_name += f"_{BLOCK_SIZE}"

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        out0 = paddle.empty(shape=[batch, seq_qkv + seq_eqkv, ouput_hidden], dtype=x.dtype)
        out1 = paddle.empty(shape=[batch, seq_qkv + seq_eqkv, ouput_hidden], dtype=x.dtype)
        out2 = paddle.empty(shape=[batch, seq_qkv + seq_eqkv, ouput_hidden], dtype=x.dtype)

        prepare_ptr_for_triton_kernel = """
        auto out0_tensor = paddle::empty({batch, seq_qkv+seq_eqkv, output_hidden}, x.dtype(), x.place());
        auto out1_tensor = paddle::empty({batch, seq_qkv+seq_eqkv, output_hidden}, x.dtype(), x.place());
        auto out2_tensor = paddle::empty({batch, seq_qkv+seq_eqkv, output_hidden}, x.dtype(), x.place());
        CUdeviceptr input_ptrs[5] = {
            get_tensor_ptr(x),
            get_tensor_ptr(y),
            get_tensor_ptr(out0_tensor),
            get_tensor_ptr(out1_tensor),
            get_tensor_ptr(out2_tensor)
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
            output_hidden=ouput_hidden,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    if in_dynamic_or_pir_mode():
        print(f"== we are in dynamic mode, op_name: {op_name}")
        outs = _C_ops._run_custom_op(
            op_name,
            x,
            y,
        )
        return outs[0], outs[1], outs[2]
    else:
        print(f"== we are in dynamic to static mode, op_name: {op_name}")
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


########################### triton split ###############################
triton_split_template = (
    """
std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor &x,
    const std::vector<int64_t> num_or_sections,
    const int64_t axis) {

  int output_batch = x.dims()[0];
  int output_seq0 = num_or_sections[0];
  int output_seq1 = num_or_sections[1];
  int output_hidden = x.dims()[2];

  auto out0_tensor = paddle::empty({output_batch, output_seq0, output_hidden}, x.dtype(), x.place());
  auto out1_tensor = paddle::empty({output_batch, output_seq1, output_hidden}, x.dtype(), x.place());
  
  auto out0 = get_tensor_ptr(out0_tensor);
  auto out1 = get_tensor_ptr(out1_tensor);
  
  auto input = get_tensor_ptr(x);
  
  auto  run_stream = out0_tensor.stream();
  
"""
    + tune_and_invoke_part
    + """
    return {out0_tensor, out1_tensor};
}

std::vector<std::vector<int64_t>> ${op_name}_InferShape(
        const std::vector<int64_t>& A_shape) {
  
  std::vector<int64_t> out_shape0 = {A_shape[0], 1024, A_shape[2]};
  std::vector<int64_t> out_shape1 = {A_shape[0], 154, A_shape[2]};
  
  return {out_shape0, out_shape1};
}

std::vector<paddle::DataType> ${op_name}_InferDtype(const paddle::DataType& A_dtype) {
    return {A_dtype, A_dtype};
}

PD_BUILD_OP(${op_name})
    .Inputs({"x"})
    .Outputs({"out0_tensor", "out1_tensor"})
    .Attrs({"num_or_sections: std::vector<int64_t>", "axis: int64_t"})
"""
)


@paddle_use_triton(
    key=["1"],
)
def triton_split_kernel(
    out0,
    out1,
    input,
    output_seq0,
    output_seq1,
    output_batch,
    output_hidden,
    BLOCK_SIZE: tl.constexpr,
):
    batch = tl.program_id(axis=0)
    out_row = tl.program_id(axis=1)
    read_ptr = out_row * output_hidden + batch * (output_seq0 + output_seq1) * output_hidden + input

    read_offsets = tl.arange(0, BLOCK_SIZE)
    mask = read_offsets < output_hidden
    read_data = tl.load(read_ptr + read_offsets, mask=mask)

    if out_row < output_seq0:
        write_ptr = batch * output_seq0 * output_hidden + out_row * output_hidden + out0 + read_offsets
    else:
        write_ptr = batch * output_seq1 * output_hidden + (out_row - output_seq0) * output_hidden + out1 + read_offsets

    tl.store(write_ptr, read_data, mask=mask)


def triton_split(x, num_or_sections=[-1, -1], axis=1):
    assert len(x.shape) == 3
    output_batch = x.shape[0]
    output_seq0 = num_or_sections[0]
    output_seq1 = num_or_sections[1]
    output_hidden = x.shape[2]

    BLOCK_SIZE = triton.next_power_of_2(output_hidden)
    op_name = "triton_split"
    op_name += get_dtype_str(x.dtype)
    op_name += f"_{BLOCK_SIZE}"

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        out0 = paddle.empty(shape=[output_batch, output_seq0, output_hidden], dtype=x.dtype)
        out1 = paddle.empty(shape=[output_batch, output_seq1, output_hidden], dtype=x.dtype)
        grid = ("output_batch", "output_seq0+output_seq1")

        triton_split_kernel[(op_name, triton_split_template, grid)](
            out0, out1, x, output_seq0, output_seq1, output_batch, output_hidden, BLOCK_SIZE=2048
        )

    if in_dynamic_or_pir_mode():
        print(f"== we are in dynamic mode, op_name: {op_name}")
        outs = _C_ops._run_custom_op(
            op_name,
            x,
            num_or_sections,
            axis,
        )
        return outs[0], outs[1]
    else:
        print(f"== we are in dynamic to static mode, op_name: {op_name}")
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "x": x,
        }
        out0 = helper.create_variable_for_type_inference(dtype=x.dtype)
        out1 = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs={
                "num_or_sections": num_or_sections,
                "axis": axis,
            },
            outputs={"out0_tensor": out0, "out1_tensor": out1},
        )
        return out0, out1
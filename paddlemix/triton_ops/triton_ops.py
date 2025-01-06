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
from typing import Any, List, Literal, Optional, Tuple, Union

from .triton_utils import (
    SubstituteTemplate,
    build_package,
    compile_file,
    extract_triton_kernel,
    find_so_path,
    get_dtype_str,
    get_op_name_with_suffix,
    get_pointer_hint,
    get_value_hint,
    link_file,
    multi_process_do,
    paddle_custom_op_head_part,
    python_path,
    rename_c_to_cu,
    tune_and_invoke_part,
)


class KernelInterface:
    def __init__(
        self,
        func,
        custom_op_template,
        other_config,
        key_args=["1"],
    ):
        self.func = func
        self.key_args = key_args

        import inspect

        signature = inspect.signature(func)
        self.arg_names = [v.name for v in signature.parameters.values()]
        for ele in self.arg_names:
            assert self.arg_names.count(ele) == 1
        arg_defaults = [v.default for v in signature.parameters.values()]

        # self.annotations = {
        #     name: ty for name, ty in func.__annotations__.items()
        # }
        self.annotations = dict(func.__annotations__)

        self.constexprs = [
            self.arg_names.index(name)
            for name in self.arg_names
            if self.annotations.get(name) == triton.language.core.constexpr
        ]

        self.arg_exclude_constexpr = [
            self.arg_names[i] for i in range(len(self.arg_names)) if i not in self.constexprs
        ]

        import textwrap

        py_script = textwrap.dedent(inspect.getsource(func))

        import re

        pat = r"def\s" + func.__name__
        func_begin = re.findall(pat, py_script)
        assert len(func_begin) == 1
        func_begin = func_begin[0]
        py_script = py_script[py_script.find(func_begin) :]

        def decorator(*args, **kwargs):
            all_input = []

            for i in range(len(args)):
                all_input.append(args[i])

            position_arguments_num = len(all_input)
            for i in range(position_arguments_num, len(self.arg_names)):
                if self.arg_names[i] in kwargs.keys():
                    all_input.append(kwargs[self.arg_names[i]])
                else:
                    # means this input is not specified, it muse be a tl.constexpr.
                    assert i in self.constexprs
                    all_input.append(None)

            dtypes = []
            x_list = []
            const_args = [self.arg_names[i] for i in self.constexprs]
            # we dont allow there are two strings in const_args, and one is a substring of the other.
            for i in const_args:
                for j in const_args:
                    if i != j and i.find(j) != -1:
                        raise ValueError(
                            f"We find {i}, {j} in tl.constexpr args, and {j} is a substring of {i}, please modify your triton kernel arguments names to avoid this."
                        )

            const_hint_dict = {}
            for i in range(len(all_input)):
                ele = all_input[i]
                if (
                    type(ele) == paddle.Tensor
                    or type(ele) == paddle.base.framework.EagerParamBase
                    or type(ele) == paddle.base.framework.Parameter
                    or type(ele) == paddle.base.framework.Variable
                    or type(ele) == paddle.base.libpaddle.pir.Value
                ):
                    dtypes.append(ele.dtype)
                elif i in self.constexprs:
                    const_hint_dict[self.arg_names[i]] = ele
                else:
                    x_list.append(ele)

            op_name = self.op_name

            python_package_name = f"{op_name}_package"

            generated_dir = os.getenv("TRITON_KERNEL_CACHE_DIR", None)
            print("the kernel cache dir is:", generated_dir)
            assert (
                generated_dir is not None
            ), "TRITON_KERNEL_CACHE_DIR is None, please set it such as export TRITON_KERNEL_CACHE_DIR=/tmp/haha "
            generated_dir = f"{generated_dir}/{op_name}"
            os.makedirs(generated_dir, exist_ok=True)

            py_script_file = f"{generated_dir}/triton_kernels.py"
            extract_triton_kernel(func, py_script_file)

            address_hint = get_pointer_hint(dtypes)
            value_hint = get_value_hint(x_list)
            const_args = [f"{{{ele}}}" for ele in const_args]
            const_args = ",".join(const_args)

            lanuch_grid = list(self.grid)
            for i in range(len(lanuch_grid)):
                ele = lanuch_grid[i]
                if type(ele) == str:
                    for key in const_hint_dict.keys():
                        if key in ele:
                            ele = ele.replace(key, f"{{{key}}}")
                else:
                    ele = str(ele)

                lanuch_grid[i] = ele
            if len(lanuch_grid) < 3:
                lanuch_grid += ["1"] * (3 - len(lanuch_grid))
            lanuch_grid = ",".join(lanuch_grid)

            op_dict = {"op_name": op_name, "reset_zero_when_tune": ""}
            op_dict["triton_kernel_args"] = ",".join(self.arg_exclude_constexpr)
            op_dict["key"] = ",".join(self.key_args)
            # when tunning, we need to reset the out to zero.
            if "reset_zero_when_tune" in other_config.keys():
                op_dict["reset_zero_when_tune"] = other_config["reset_zero_when_tune"]

            paddle_custom_op_file_path = f"{generated_dir}/{op_name}.cu"
            so_path = find_so_path(generated_dir, python_package_name)

            if so_path is None:
                print("== we do not find so_path, we need to compile it")
                with open(paddle_custom_op_file_path, "w") as f:
                    f.write(
                        SubstituteTemplate(
                            custom_op_template,
                            op_dict,
                        )
                    )
                    f.close()

                # ahead of time compile command.
                aot_template = (
                    f"""{python_path}   {compile_file} {py_script_file}   -n {func.__name__} -o {generated_dir}/{op_name}_kernel --out-name {op_name}_kernel  """
                    + """ -w {num_warps} -ns {num_stages} """
                    + f""" -s"{address_hint} {value_hint} {const_args}" """
                    + f"""  -g "{lanuch_grid}" """
                )
                all_tune_config = list(self.tune_config)
                if len(all_tune_config) == 0:
                    # when user do not specify config, we use const_hint_dict as config.
                    all_tune_config = [const_hint_dict]
                    # reset const_hint_dict as empty.
                    const_hint_dict = {}
                codegen_commands = []
                for config in all_tune_config:
                    for key in const_hint_dict.keys():
                        if const_hint_dict[key] is not None:
                            if key not in config.keys():
                                config[key] = const_hint_dict[key]
                            else:
                                raise ValueError(f"you specify {key} both in arguments and config, this is wrong.")
                        else:
                            assert key in config.keys(), f"you must specify {key} in your config."
                    if "num_warps" not in config.keys():
                        config["num_warps"] = 4
                    if "num_stages" not in config.keys():
                        config["num_stages"] = 4

                    for key in config:
                        assert config[key] is not None, f"{key} must be specified."
                    codegen_command = aot_template.format(
                        **config,
                    )
                    print(codegen_command)
                    codegen_commands.append(codegen_command)
                multi_process_do(codegen_commands)

                link_command = f"{python_path}  {link_file}  {generated_dir}/*.h -o {generated_dir}/{op_name}_kernel"
                re = os.system(link_command)
                assert re == 0

                # rename the .c file to .cu
                rename_c_to_cu(generated_dir)
                # build the package to so, not install
                build_package(generated_dir, python_package_name)

            if op_name not in OpProtoHolder.instance().op_proto_map.keys():
                so_path = find_so_path(generated_dir, python_package_name)
                print("== we find so_path: ", so_path)
                assert so_path is not None
                paddle.utils.cpp_extension.load_op_meta_info_and_register_op(so_path)

        self.decorator = decorator

    def __getitem__(self, op_name_and_grid):
        assert len(op_name_and_grid) >= 2, "len(op_name_and_grid) must >= 2."
        self.op_name = op_name_and_grid[0]
        self.grid = op_name_and_grid[1]
        if len(op_name_and_grid) == 2:
            self.tune_config = {}
        else:
            self.tune_config = op_name_and_grid[2]
        return self.decorator


def paddle_use_triton(custom_op_template, other_config={}, key=[]):

    index = custom_op_template.find("PD_BUILD_OP")

    body = custom_op_template[:index]

    if body.find("${op_name}_InferShape") == -1:
        body += "std::vector<std::vector<int64_t>> ${op_name}_InferShape(const std::vector<int64_t>& A_shape) {return {A_shape};}"

    if body.find("${op_name}_InferDtype") == -1:
        body += (
            "std::vector<paddle::DataType> ${op_name}_InferDtype(const paddle::DataType& A_dtype) {return {A_dtype};}"
        )

    tail = custom_op_template[index:]

    tail += """
    .SetKernelFn(PD_KERNEL(${op_name}_func))
    .SetInferDtypeFn(PD_INFER_DTYPE(${op_name}_InferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(${op_name}_InferShape));
    """

    custom_op_template = paddle_custom_op_head_part + body + tail

    def decorator(func):
        return KernelInterface(func, custom_op_template, other_config, key)

    return decorator


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


triton_wint8_template = (
    """
std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor& x,
    const paddle::Tensor& qweight,
    const paddle::Tensor& scales,
    paddle::optional<paddle::Tensor>& bias,
    bool bool_trans_w) {
  int M = x.shape()[0];
  int K = x.shape()[1];
  int N = scales.shape()[0];

  auto c_out = paddle::full({M, N}, 0, x.dtype(), x.place());

  auto a_ptr = get_tensor_ptr(x);
  auto b_ptr = get_tensor_ptr(qweight);
  auto c_ptr = get_tensor_ptr(c_out);
  auto bs_ptr = get_tensor_ptr(scales);
  CUdeviceptr bias_ptr = (CUdeviceptr)(nullptr);
  if (bias) {
    bias_ptr = get_tensor_ptr(*bias);
  }

  int stride_bk = N;
  int stride_bn = 1;

  if (bool_trans_w) {
    stride_bk = 1;
    stride_bn = K;
  }
  int stride_am = K;
  int stride_ak = 1;

  int stride_cm = N;
  int stride_cn = 1;

  auto run_stream = c_out.stream();
"""
    + tune_and_invoke_part
    + """
  return {c_out};
}

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

PD_BUILD_OP(${op_name})
    .Inputs({"x", "qweight", "scales", paddle::Optional("bias")})
    .Outputs({"out"})
    .Attrs({"bool_trans_w: bool"})
"""
)

wint8_kernel_other_config = {
    "reset_zero_when_tune": "cudaMemset((void*)c_ptr, 0, sizeof(phi::dtype::float16) * M * N);"
}


@paddle_use_triton(
    custom_op_template=triton_wint8_template,
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
        triton_output = paddlemix.custom_ops.weight_only_int8(
            activation,
            triton_uint_qweight,
            triton_scale,
            bias, bool_trans_w=bool_trans_w_triton)

    paddle.device.synchronize()

    starttime = datetime.datetime.now()
    for i in range(100):
        triton_output = paddlemix.custom_ops.weight_only_int8(
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
        output = paddle.zeros((1, 1), dtype=x.dtype)

        grid = (
            "((M+BLOCK_SIZE_M-1)/BLOCK_SIZE_M) * ((N+BLOCK_SIZE_N-1)/BLOCK_SIZE_N)",
            "SPLIT_K",
        )

        wint8_kernel[(op_name, grid, get_wint8_kernel_config())](
            x,
            qweight,
            output,
            scales,
            bias,
            M,
            N,
            K,
            K,
            1,  # A always is rowmajor
            stride_bk,
            stride_bn,
            N,
            1,  # C always is rowmajor
        )

    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(op_name, x, qweight, scales, bias, bool_trans_w)
        return outs[0]

    helper = LayerHelper(op_name, **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    inputs = {
        "x": x,
        "qweight": qweight,
        "scales": scales,
        "bias@OPTIONAL": bias,
    }

    helper.append_op(
        type=op_name,
        inputs=inputs,
        attrs={"bool_trans_w": bool_trans_w},
        outputs={"out": out},
    )
    return out


########################### adaptive layer norm ###############################
fused_adaLN_scale_residual_template = (
    """


std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor &x,
    const paddle::Tensor &mha_out,
    const paddle::Tensor &gate_msa,
    const paddle::Tensor &scale_mlp,
    const paddle::Tensor &shift_mlp,
    paddle::optional<paddle::Tensor> &weight,
    paddle::optional<paddle::Tensor> &bias,
    float epsilon) {
  int M = x.dims()[0] * x.dims()[1];
  int N = x.dims()[2];
  int seq_size = x.dims()[1];
  auto resi_out = paddle::empty(x.shape(), x.dtype(), x.place());
  auto adaLN_out = paddle::empty(x.shape(), x.dtype(), x.place());

  auto x_ptr = get_tensor_ptr(x);
  auto mha_out_ptr = get_tensor_ptr(mha_out);
  auto resi_out_ptr = get_tensor_ptr(resi_out);
  auto adaLN_out_ptr = get_tensor_ptr(adaLN_out);
  auto gate_msa_ptr = get_tensor_ptr(gate_msa);
  auto scale_mlp_ptr = get_tensor_ptr(scale_mlp);
  auto shift_mlp_ptr = get_tensor_ptr(shift_mlp);
  CUdeviceptr weight_ptr = (CUdeviceptr)(nullptr);
  if (weight) {
    weight_ptr = get_tensor_ptr(*weight);
  }
  CUdeviceptr bias_ptr = (CUdeviceptr)(nullptr);
  if (bias) {
    bias_ptr = get_tensor_ptr(*bias);
  }
  auto  run_stream = adaLN_out.stream();
"""
    + tune_and_invoke_part
    + """
    return {resi_out, adaLN_out};
}

std::vector<std::vector<int64_t>> ${op_name}_InferShape(
        const std::vector<int64_t>& A_shape) {
  return {A_shape, A_shape};
}

std::vector<paddle::DataType> ${op_name}_InferDtype(const paddle::DataType& A_dtype) {
    return {A_dtype, A_dtype};
}

PD_BUILD_OP(${op_name})
    .Inputs({"x", "mha_out", "gate_msa", "scale_mlp", "shift_mlp", paddle::Optional("weight"), paddle::Optional("bias")})
    .Outputs({"resi_out", "adaLN_out"})
    .Attrs({"epsilon: float"})
"""
)


@paddle_use_triton(
    custom_op_template=fused_adaLN_scale_residual_template,
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
        resi_out_triton, adaLN_out_triton = paddlemix.custom_ops.fused_adaLN_scale_residual(x, mha_out, gate_msa, scale_mlp_x, shift_mlp_x, weight, bias, epsilon)

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
        grid = ("M",)
        fused_adaLN_scale_residual_kernel[(op_name, grid, fused_adaLN_scale_residual_kernel_config)](
            x,
            mha_out,
            gate_msa,
            scale_mlp,
            shift_mlp,
            shift_mlp,
            shift_mlp,
            resi_out,
            adaLN_out,
            -1,
            N,
            -1,
            epsilon,
            N_npo2=N_npo2,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
        )

    if in_dynamic_or_pir_mode():
        #print(f"== we are in dynamic mode, op_name: {op_name}")
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
        #print(f"== we are in dynamic to static mode, op_name: {op_name}")
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


triton_adaptive_layer_norm_template = (
    """

std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor &x,
    const paddle::Tensor &scale,
    const paddle::Tensor &shift,
    paddle::optional<paddle::Tensor> &weight,
    paddle::optional<paddle::Tensor> &bias,
    float epsilon) {
  int M = x.dims()[0] * x.dims()[1];
  int N = x.dims()[2];
  int seq_size = x.dims()[1];
  auto y = paddle::empty(x.shape(), x.dtype(), x.place());

  auto x_ptr = get_tensor_ptr(x);
  auto y_ptr = get_tensor_ptr(y);
  auto scale_ptr = get_tensor_ptr(scale);
  auto shift_ptr = get_tensor_ptr(shift);
  CUdeviceptr weight_ptr = (CUdeviceptr)(nullptr);
  if (weight) {
    weight_ptr = get_tensor_ptr(*weight);
  }
  CUdeviceptr bias_ptr = (CUdeviceptr)(nullptr);
  if (bias) {
    bias_ptr = get_tensor_ptr(*bias);
  }
  auto run_stream = y.stream();
"""
    + tune_and_invoke_part
    + """
  return {y};
}

PD_BUILD_OP(${op_name})
    .Inputs({"x", "scale", "shift", paddle::Optional("weight"), paddle::Optional("bias")})
    .Outputs({"out"})
    .Attrs({"epsilon: float"})
"""
)


@paddle_use_triton(
    custom_op_template=triton_adaptive_layer_norm_template,
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
        mt_result = paddlemix.custom_ops.adaptive_layer_norm(x, scale_msa_x, shift_msa_x, weight, bias)

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
        grid = ("M",)
        adaptive_layer_norm_kernel[(op_name, grid, adaptive_layer_norm_kernel_config)](
            x,
            y,
            y,
            y,
            y,
            y,
            -1,
            N,
            -1,
            epsilon,
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
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs={
                "epsilon": epsilon,
            },
            outputs={"out": out},
        )
        return out


rms_norm_template = (
    """

std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor &x,
    paddle::optional<paddle::Tensor> &weight,
    paddle::optional<paddle::Tensor> &bias,
    float epsilon) {
  int M = x.dims()[0] * x.dims()[1] * x.dims()[2];
  int N = x.dims()[3];
  auto y = paddle::empty(x.shape(), x.dtype(), x.place());

  auto x_ptr = get_tensor_ptr(x);
  auto y_ptr = get_tensor_ptr(y);
  CUdeviceptr weight_ptr = (CUdeviceptr)(nullptr);
  if (weight) {
    weight_ptr = get_tensor_ptr(*weight);
  }
  CUdeviceptr bias_ptr = (CUdeviceptr)(nullptr);
  if (bias) {
    bias_ptr = get_tensor_ptr(*bias);
  }
  auto run_stream = y.stream();
"""
    + tune_and_invoke_part
    + """
    return {y};
}

PD_BUILD_OP(${op_name})
    .Inputs({"x", paddle::Optional("weight"), paddle::Optional("bias")})
    .Outputs({"out"})
    .Attrs({"epsilon: float"})
"""
)


@paddle_use_triton(
    custom_op_template=rms_norm_template,
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


def rms_norm(x, weight=None, bias=None, epsilon=1e-05):
    """
    Examples:

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    import paddle

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
        mt_result = paddlemix.custom_ops.rms_norm(x,weight,bias,1e-5)


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

    op_name = "triton_rms_norm"
    op_name += get_dtype_str(x.dtype)
    op_name += f"_{N_npo2}"

    rms_norm_kernel_config = []
    if N_npo2 <= 64:
        rms_norm_kernel_config.append({"BLOCK_SIZE_M": 4, "num_warps": 1})
    else:
        rms_norm_kernel_config.append({"BLOCK_SIZE_M": 1, "num_warps": 4})

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        y = paddle.empty_like(x)
        grid = ("((M+BLOCK_SIZE_M-1)/BLOCK_SIZE_M)",)
        rms_norm_kernel[(op_name, grid, rms_norm_kernel_config)](
            x,
            y,
            weight,
            x,
            -1,  # M,
            N,
            epsilon,
            N_npo2=N_npo2,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
        )

    if in_dynamic_or_pir_mode():
        #print(f"== we are in dynamic mode, op_name: {op_name}")
        outs = _C_ops._run_custom_op(op_name, x, weight, bias, epsilon)
        return outs[0]
    else:
        #print(f"== we are in dynamic to static mode, op_name: {op_name}")
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "x": x,
            "weight@OPTIONAL": weight,
            "bias@OPTIONAL": bias,
        }
        out = helper.create_variable_for_type_inference(dtype=x.dtype)
        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs={
                "epsilon": epsilon,
            },
            outputs={"out": out},
        )
        return out


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
    custom_op_template=fused_rotary_emb_template,
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
        fused_rotary_emb_kernel[(op_name, grid, fused_rotary_emb_kernel_config)](
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
        #print(f"== we are in dynamic mode, op_name: {op_name}")
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
        #print(f"== we are in dynamic to static mode, op_name: {op_name}")
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


########################### split concat ###############################
split_concat_template = (
    """
std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor &x,
    const paddle::Tensor &y) {

  int batch = x.dims()[0];
  
  int seq_qkv = x.dims()[1];
  int seq_eqkv = y.dims()[1];
  int output_hidden = x.dims()[2] / 3;
  
  auto qkv = get_tensor_ptr(x);
  auto eqkv = get_tensor_ptr(y);
  
  auto out0_tensor = paddle::empty({batch, seq_qkv+seq_eqkv, output_hidden}, x.dtype(), x.place());
  auto out1_tensor = paddle::empty({batch, seq_qkv+seq_eqkv, output_hidden}, x.dtype(), x.place());
  auto out2_tensor = paddle::empty({batch, seq_qkv+seq_eqkv, output_hidden}, x.dtype(), x.place());
  
  auto out0 = get_tensor_ptr(out0_tensor);
  auto out1 = get_tensor_ptr(out1_tensor);
  auto out2 = get_tensor_ptr(out2_tensor);
  
  
  auto  run_stream = out0_tensor.stream();
  
"""
    + tune_and_invoke_part
    + """
    return {out0_tensor, out1_tensor, out2_tensor};
}

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

PD_BUILD_OP(${op_name})
    .Inputs({"x", "y"})
    .Outputs({"out0_tensor", "out1_tensor", "out2_tensor"})
"""
)


@paddle_use_triton(
    custom_op_template=split_concat_template,
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
    BLOCK_SIZE = triton.next_power_of_2(ouput_hidden)
    op_name = "split_concat"
    op_name += get_dtype_str(x.dtype)
    op_name += f"_{BLOCK_SIZE}"

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        out0 = paddle.empty(shape=[batch, seq_qkv + seq_eqkv, ouput_hidden], dtype=x.dtype)
        out1 = paddle.empty(shape=[batch, seq_qkv + seq_eqkv, ouput_hidden], dtype=x.dtype)
        out2 = paddle.empty(shape=[batch, seq_qkv + seq_eqkv, ouput_hidden], dtype=x.dtype)
        grid = ("3", "batch", "seq_qkv + seq_eqkv")
        # -1 means this value does not matter for triton compilation
        split_concat_kernel[(op_name, grid)](
            out0, out1, out2, x, y, -1, seq_qkv, seq_eqkv, ouput_hidden, BLOCK_SIZE=BLOCK_SIZE  # batch,
        )

    if in_dynamic_or_pir_mode():
        #print(f"== we are in dynamic mode, op_name: {op_name}")
        outs = _C_ops._run_custom_op(
            op_name,
            x,
            y,
        )
        return outs[0], outs[1], outs[2]
    else:
        #print(f"== we are in dynamic to static mode, op_name: {op_name}")
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
    custom_op_template=triton_split_template,
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

        triton_split_kernel[(op_name, grid)](
            out0, out1, x, output_seq0, output_seq1, output_batch, output_hidden, BLOCK_SIZE=2048
        )

    if in_dynamic_or_pir_mode():
        #print(f"== we are in dynamic mode, op_name: {op_name}")
        outs = _C_ops._run_custom_op(
            op_name,
            x,
            num_or_sections,
            axis,
        )
        return outs[0], outs[1]
    else:
        #print(f"== we are in dynamic to static mode, op_name: {op_name}")
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


########################### sage attention ###############################
sageattn_per_block_int8_triton_template = (
    """
#include <cmath>
std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor &x,
    paddle::optional<paddle::Tensor> &km,
    int BLK,
    float sm_scale,
    std::string tensor_layout,
    std::string q_or_k
){
    auto output_tensor = paddle::empty(x.shape(), paddle::DataType::INT8, 
                                x.place());
    auto input_tensor = x;
    
    auto input_shape = x.shape();
    
    // define params
    int b, h_attn, seq_len, head_dim;
    int stride_iz, stride_ih, stride_in;
    int stride_oz, stride_oh, stride_on;
    
    // allocate
    if (tensor_layout == std::string("HND")) {
        // tensor layout unpack
        b = input_shape[0];
        h_attn = input_shape[1];
        seq_len = input_shape[2];
        head_dim = input_shape[3];
        
        // stride unpack
        auto tensor_strides = input_tensor.strides();
        stride_iz = tensor_strides[0];
        stride_ih = tensor_strides[1];
        stride_in = tensor_strides[2];
        
        auto tensor_o_strides = output_tensor.strides();
        stride_oz = tensor_o_strides[0];
        stride_oh = tensor_o_strides[1];
        stride_on = tensor_o_strides[2];
        
    } else if (tensor_layout == std::string("NHD")) {
        // tensor layout unpack
        b = input_shape[0];
        h_attn = input_shape[2];    // reverse
        seq_len = input_shape[1];
        head_dim = input_shape[3];
        
        // stride unpack
        auto tensor_strides = input_tensor.strides();
        stride_iz = tensor_strides[0];
        stride_ih = tensor_strides[2];    // reverse
        stride_in = tensor_strides[1];
        
        auto tensor_o_strides = output_tensor.strides();
        stride_oz = tensor_o_strides[0];
        stride_oh = tensor_o_strides[2];  // reverse
        stride_on = tensor_o_strides[1];
    }
    else {
        throw std::runtime_error("Unsupported tensor layout");
    }
    
    int L = seq_len;
    
    auto scale_tensor = paddle::empty({b, h_attn, (seq_len + BLK - 1) / BLK}, 
                                        paddle::DataType::FLOAT32, 
                                        x.place());
    
    int stride_sz = scale_tensor.strides()[0];
    int stride_sh = scale_tensor.strides()[1];
    // sm_scale = sm_scale * 1.44269504;
    if (q_or_k == std::string("k")) {
        sm_scale = 1.0f;
    }
    
    int C = head_dim;
    int Grid = BLK;
    int bsz = b;
    
    // prepare tensor
    auto Input = get_tensor_ptr(x);
    auto Output = get_tensor_ptr(output_tensor);
    auto Scale = get_tensor_ptr(scale_tensor);
    
    auto run_stream = x.stream();
    
""" + tune_and_invoke_part + """
    return {output_tensor, scale_tensor};
}
std::vector<std::vector<int64_t>> ${op_name}_InferShape(
    const std::vector<int64_t>& A_shape) {
    int BSZ = A_shape[0];
    int HEAD_NUM = A_shape[1];
    int SEQ_LEN = A_shape[2];
    int HEAD_DIM = A_shape[3];
    
    std::vector<int64_t> out_shape = {BSZ, HEAD_NUM, SEQ_LEN, HEAD_DIM};
    
    std::string func_name("${op_name}");
    int BLK;
    if (func_name.find("BLK128") != std::string::npos) {
        BLK = 128;
    } else if (func_name.find("BLK64") != std::string::npos) {
        BLK = 64;
    } else {
        throw std::runtime_error("Unsupported BLK");
    }
    std::vector<int64_t> out2_shape = {BSZ, HEAD_NUM, (SEQ_LEN + BLK - 1) / BLK};
    
    return {out_shape, out2_shape};
}
std::vector<paddle::DataType> ${op_name}_InferDtype(const paddle::DataType& A_dtype) {
    return {paddle::DataType::INT8, paddle::DataType::FLOAT32};
}
PD_BUILD_OP(${op_name})
    .Inputs({"x", paddle::Optional("km")})
    .Outputs({"output_tensor", "scale_tensor"})
    .Attrs({"BLK: int", "sm_scale: float", "tensor_layout: std::string", "q_or_k: std::string"})
"""
)

@paddle_use_triton(
    custom_op_template=sageattn_per_block_int8_triton_template,
    key=["1"]
)
def sageattn_quant_per_block_int8_kernel(
    Input,
    Output,
    Scale,
    L,
    stride_iz, stride_ih, stride_in,
    stride_oz, stride_oh, stride_on,
    stride_sz, stride_sh,
    sm_scale,
    Grid,                   # grid num, through compiling
    h_attn,                 # grid num, through compiling
    bsz,                    # grid num, through compiling
    C: tl.constexpr,
    BLK: tl.constexpr
):
    off_blk = tl.program_id(axis=0)
    off_h = tl.program_id(axis=1)
    off_b = tl.program_id(axis=2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk

    x_data = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x_data = x_data.to(tl.float32)
    x_data *= sm_scale
    scale = tl.max(tl.abs(x_data)) / 127.
    x_int8 = x_data / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)

# note: here we need to do one single operation, instead of fused two.
# reference: quant_per_block.py
def sageattn_quant_per_block_int8(x, 
                                km=None, BLKQ=128, BLKK=64,
                                sm_scale=None, 
                                tensor_layout="HND", q_or_k="q"):
    Output = paddle.empty(x.shape, dtype=paddle.int8)

    if km is not None and q_or_k == "k":
        x = x - km

    if tensor_layout == "HND":
        b, h_attn, seq_len, head_dim = x.shape

        # there is no stride in static mode, so we need to compute it manually
        stride_iz, stride_ih, stride_in = head_dim * seq_len * h_attn, head_dim * seq_len, head_dim * 1
        stride_oz, stride_oh, stride_on = head_dim * seq_len * h_attn, head_dim * seq_len, head_dim * 1
    elif tensor_layout == "NHD":
        b, seq_len, h_attn, head_dim = x.shape

        stride_iz, stride_ih, stride_in = head_dim * seq_len * h_attn, head_dim * 1, head_dim * h_attn
        stride_oz, stride_oh, stride_on = head_dim * seq_len * h_attn, head_dim * 1, head_dim * h_attn,
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    BLK = BLKQ if q_or_k == "q" else BLKK
    Scale = paddle.empty((b, h_attn, (seq_len + BLK - 1) // BLK), dtype='float32')

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    L = seq_len
    C = head_dim
    gd = BLK
    sm_scale = sm_scale * 1.44269504 if q_or_k == "q" else 1.0

    stride_sz = h_attn * ((seq_len + BLK - 1) // BLK)
    stride_sh =  (seq_len + BLK - 1) // BLK

    op_name = "triton_sageattn_quant_per_block"
    op_name += get_dtype_str(Output.dtype)
    op_name += f"_BLK{BLK}_seq{seq_len}_h{h_attn}_dim{head_dim}"
    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        grid = ("(L + Grid - 1) / Grid", "h_attn", "bsz")
        sageattn_quant_per_block_int8_kernel[(op_name, grid)](
            x, Output, Scale, L, 
            stride_iz, stride_ih, stride_in,
            stride_oz, stride_oh, stride_on,
            stride_sz, stride_sh,
            sm_scale,
            gd,         # grid num, through compiling
            h_attn,     # grid num, through compiling
            b,          # grid num, through compiling
            C, 
            BLK
        )

    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            op_name,
            x, km, BLK,
            sm_scale, tensor_layout, q_or_k
        )

        return outs[0], outs[1]
    else:
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "x": x,
            "km@OPTIONAL": km,
        }
        out_int8 = helper.create_variable_for_type_inference(dtype=x.dtype)
        out_scale = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs={
                "BLK": BLK,
                "sm_scale": sm_scale,
                "tensor_layout": tensor_layout,
                "q_or_k": q_or_k
            },
            outputs={"output_tensor": out_int8, "scale_tensor": out_scale}
        )
        return out_int8, out_scale


sageattn_sageattn_attn_fwd_casual_false_template = ("""
std::vector<paddle::Tensor> ${op_name}_func(
    const paddle::Tensor &x,        // x stands for q_tensor
    const paddle::Tensor &k_tensor,
    const paddle::Tensor &v_tensor,
    const paddle::Tensor &q_scale,
    const paddle::Tensor &k_scale,
    std::string output_dtype,
    std::string tensor_layout,
    int return_lse
) {
    int BLOCK_M = 128;
    int BLOCK_N = 64;
    int STAGE = 1;
    
    paddle::DataType output_t;
    if (output_dtype == std::string("float16")) {
        output_t = paddle::DataType::FLOAT16;
    } else {
        output_t = paddle::DataType::BFLOAT16;
    }
    
    auto out_tensor = paddle::empty(x.shape(), output_t, x.place());
    auto q_strides = x.strides();
    auto k_strides = k_tensor.strides();
    auto v_strides = v_tensor.strides();
    auto o_strides = out_tensor.strides();
    
    int b, h_qo, qo_len, head_dim;
    int kv_len, h_kv;
    
    int stride_qz, stride_qh, stride_qn;
    int stride_kz, stride_kh, stride_kn;
    int stride_vz, stride_vh, stride_vn;
    int stride_oz, stride_oh, stride_on;
    
    if (tensor_layout == "HND") {
        b = x.shape()[0];
        h_qo = x.shape()[1];
        qo_len = x.shape()[2];
        head_dim = x.shape()[3];
        
        h_kv = k_tensor.shape()[1];
        kv_len = k_tensor.shape()[2];
        
        stride_qz = q_strides[0];
        stride_qh = q_strides[1];
        stride_qn = q_strides[2];
        
        stride_kz = k_strides[0];
        stride_kh = k_strides[1];
        stride_kn = k_strides[2];
        
        stride_vz = v_strides[0];
        stride_vh = v_strides[1];
        stride_vn = v_strides[2];
        
        stride_oz = o_strides[0];
        stride_oh = o_strides[1];
        stride_on = o_strides[2];
    } else if (tensor_layout == "NHD") {
        b = x.shape()[0];
        qo_len = x.shape()[1];   // reverse
        h_qo = x.shape()[2];
        head_dim = x.shape()[3];
        
        kv_len = k_tensor.shape()[1];   // reverse
        h_kv = k_tensor.shape()[2];
        
        stride_qz = q_strides[0];
        stride_qh = q_strides[2];       // reverse
        stride_qn = q_strides[1];
        
        stride_kz = k_strides[0];
        stride_kh = k_strides[2];       // reverse
        stride_kn = k_strides[1];
        
        stride_vz = v_strides[0];
        stride_vh = v_strides[2];       // reverse
        stride_vn = v_strides[1];
        
        stride_oz = o_strides[0];
        stride_oh = o_strides[2];       // reverse
        stride_on = o_strides[1];
    } else {
        throw std::runtime_error("Unsupported tensor layout");
    }
    
    int HEAD_DIM_K = head_dim;
    int num_kv_groups = h_qo / h_kv;
    
    paddle::Tensor lse_tensor;
    
    if (return_lse) {
        lse_tensor = paddle::empty({b, h_qo, qo_len}, x.dtype(), x.place());
    } else {
        lse_tensor = paddle::empty({1,1,1}, paddle::DataType::FLOAT32, paddle::CPUPlace());
    }
    
    bool RETURN_LSE = return_lse;
    int H_ = h_qo;
    int HEAD_DIM = HEAD_DIM_K;
    
    auto Q = get_tensor_ptr(x);
    auto K = get_tensor_ptr(k_tensor);
    auto V = get_tensor_ptr(v_tensor);
    auto Q_scale = get_tensor_ptr(q_scale);
    auto K_scale = get_tensor_ptr(k_scale);
    auto Out = get_tensor_ptr(out_tensor);
    auto Lse = get_tensor_ptr(lse_tensor);
    
    int BSZ = b;
    auto run_stream = x.stream();
""" + tune_and_invoke_part + """
    return {out_tensor, lse_tensor};
}
std::vector<std::vector<int64_t>> ${op_name}_InferShape(
    const std::vector<int64_t>& A_shape,
    const std::vector<int64_t>& B_shape,
    const std::vector<int64_t>& C_shape,
    const std::vector<int64_t>& D_shape,
    const std::vector<int64_t>& E_shape) {
        
    int BSZ = A_shape[0];
    int HEAD_NUM = A_shape[1];
    int SEQ_LEN = A_shape[2];
    int HEAD_DIM = A_shape[3];
    
    std::vector<int64_t> out_shape = {BSZ, HEAD_NUM, SEQ_LEN, HEAD_DIM};
    
    std::vector<int64_t> lse_shape = {BSZ, HEAD_NUM, SEQ_LEN};
    
    return {out_shape, lse_shape};
}
std::vector<paddle::DataType> ${op_name}_InferDtype(const paddle::DataType& A_dtype) {
    return {paddle::DataType::FLOAT16, paddle::DataType::FLOAT32};
}
PD_BUILD_OP(${op_name})
    .Inputs({"x", "k_tensor", "v_tensor", "q_scale", "k_scale"})
    .Outputs({"out_tensor", "lse_tensor"})
    .Attrs({"output_dtype: std::string", "tensor_layout: std::string", "return_lse: int"})
""")


@paddle_use_triton(
    custom_op_template=sageattn_sageattn_attn_fwd_casual_false_template,
    key=["1"]
)
def sageattn_attn_fwd_casual_false_kernel(
            Q, K, V, Q_scale, K_scale, Out, Lse, 
            stride_qz, stride_qh, stride_qn,
            stride_kz, stride_kh, stride_kn,  
            stride_vz, stride_vh, stride_vn,  
            stride_oz, stride_oh, stride_on,  
            qo_len, kv_len, BSZ,
            H_: tl.constexpr, 
            num_kv_groups: tl.constexpr,
            HEAD_DIM: tl.constexpr,  
            BLOCK_M: tl.constexpr,  
            BLOCK_N: tl.constexpr,  
            STAGE: tl.constexpr,
            RETURN_LSE: tl.constexpr,):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    q_scale_offset = (off_z * H_ + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H_ // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)  

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] 
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask = offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)

    # fused zone 
    lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)   
        k = tl.load(K_ptrs, mask = k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale 
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]

        v = tl.load(V_ptrs, mask = offs_n[:, None] < (kv_len - start_n))
        p = p.to(tl.float16)

        acc += tl.dot(p, v, out_dtype=tl.float16)   
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    # zone end

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H_ + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask = (offs_m < qo_len))


def sageattn_forward_casual_false(q, k, v, 
                                  q_scale, k_scale, 
                                  output_dtype="float16",
                                  tensor_layout="HND", 
                                  return_lse=False):
    BLOCK_M = 128
    BLOCK_N = 64
    stage = 1

    assert output_dtype in ["float16", "bfloat16"]

    Out = paddle.empty(q.shape, dtype=output_dtype)
    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_qz, stride_qh, stride_qn = h_qo * qo_len * head_dim, qo_len * head_dim, head_dim
        stride_kz, stride_kh, stride_kn = h_kv * kv_len * head_dim, kv_len * head_dim, head_dim
        stride_vz, stride_vh, stride_vn = h_kv * kv_len * head_dim, kv_len * head_dim, head_dim
        stride_oz, stride_oh, stride_on = h_qo * qo_len * head_dim, qo_len * head_dim, head_dim
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_qz, stride_qh, stride_qn = qo_len * h_qo * head_dim, head_dim, h_qo * head_dim
        stride_kz, stride_kh, stride_kn = kv_len * h_kv * head_dim, head_dim, h_kv * head_dim
        stride_vz, stride_vh, stride_vn = kv_len * h_kv * head_dim, head_dim, h_kv * head_dim
        stride_oz, stride_oh, stride_on = qo_len * h_qo * head_dim, head_dim, h_qo * head_dim
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv
    BSZ = b

    if return_lse:
        Lse = paddle.empty((b, h_qo, qo_len), dtype=paddle.float32)
    else:
        Lse = paddle.empty((0, 0, 0), dtype=paddle.float32)

    op_name = "triton_sageattn_attn_fwd_casual_false"
    op_name += get_dtype_str(q.dtype)
    op_name += f"_{BLOCK_M}_{BLOCK_N}_BSZ{BSZ}_seq{qo_len}_h{h_qo}_head{HEAD_DIM_K}"

    sageattn_attn_fwd_casual_false_config = []
    if head_dim == 64:
        sageattn_attn_fwd_casual_false_config.append({
            "num_warps": 4,
            "num_stages": 3
        })
    else:
        sageattn_attn_fwd_casual_false_config.append({
            "num_warps": 8,
            "num_stages": 4
        })

    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        grid = ("(qo_len+BLOCK_M-1)/BLOCK_M", "H_", "BSZ")
        sageattn_attn_fwd_casual_false_kernel[(op_name, grid, sageattn_attn_fwd_casual_false_config)](
            q, k, v, q_scale, k_scale, Out, Lse, 
            stride_qz, stride_qh, stride_qn,
            stride_kz, stride_kh, stride_kn,
            stride_vz, stride_vh, stride_vn,
            stride_oz, stride_oh, stride_on,
            qo_len, kv_len, BSZ,
            h_qo, 
            num_kv_groups,
            HEAD_DIM_K,
            BLOCK_M, 
            BLOCK_N, 
            stage,
            1 if return_lse else 0
        )

    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            op_name,
            q,k,v, q_scale, k_scale, 
            output_dtype,
            tensor_layout, 
            1 if return_lse else 0
        )

        return outs[0], outs[1]
    else:
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "x": q,
            "k_tensor": k,
            "v_tensor": v,
            "q_scale": q_scale,
            "k_scale": k_scale,
        }
        out_tensor = helper.create_variable_for_type_inference(dtype=Out.dtype)
        out_lse = helper.create_variable_for_type_inference(dtype=Lse.dtype)

        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs={
                "output_type": output_dtype,
                "tensor_layout": tensor_layout,
                "return_lse": 1 if return_lse else 0
            },
            outputs={
                "out_tensor": out_tensor,
                "lse_tensor": out_lse
            }
        )

        return out_tensor, out_lse


# ============== sage attention triton API =================
def per_block_int8(q, k, km=None, BLKQ=128, BLKK=64, sm_scale=None, 
                   tensor_layout="HND"):
    q_int8, q_scale = sageattn_quant_per_block_int8(
        q, km=None, BLKQ=BLKQ, BLKK=BLKK, sm_scale=sm_scale, tensor_layout=tensor_layout, q_or_k='q')
    k_int8, k_scale = sageattn_quant_per_block_int8(
        k, km=km, BLKQ=BLKQ, BLKK=BLKK, sm_scale=sm_scale, tensor_layout=tensor_layout, q_or_k='k')
    return q_int8, q_scale, k_int8, k_scale


def sageattn_qk_int8_pv_fp16_triton(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    tensor_layout: str = "HND",
    is_casual: bool = False,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs
) -> paddle.Tensor:
    dtype = q.dtype
    assert dtype in [paddle.float16, paddle.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert str(q.place) == str(k.place) == str(v.place), f"All tensors must be on the same device. Got q: {q.place}, k: {k.place}, v: {v.place}"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    head_dim_og = q.shape[-1]
    # if not 64 or 128, then fill to 64 or 128
    if head_dim_og < 64:
        q = paddle.nn.functional.pad(q, pad=[0, 64-head_dim_og])
        k = paddle.nn.functional.pad(k, pad=[0, 64-head_dim_og])
        v = paddle.nn.functional.pad(v, pad=[0, 64-head_dim_og])
    elif head_dim_og > 64 and head_dim_og < 128:
        q = paddle.nn.functional.pad(q, pad=[0, 128-head_dim_og])
        k = paddle.nn.functional.pad(k, pad=[0, 128-head_dim_og])
        v = paddle.nn.functional.pad(v, pad=[0, 128-head_dim_og])
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    seq_dim = 1 if tensor_layout == "NHD" else 2

    if smooth_k:
        km = paddle.mean(k, axis=seq_dim, keepdim=True)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = paddle.matmul(paddle.transpose(q, [0, 2, 1, 3]), paddle.squeeze(paddle.transpose(q, [0, 2, 3, 1]), axis=-1)).astype(paddle.float32)
            else:
                lse_correction = paddle.matmul(q, paddle.squeeze(paddle.transpose(km, [0, 1, 3, 2]), axis=-1)).astype(paddle.float32)
    else:
        km = None

    if dtype == paddle.bfloat16 or dtype == paddle.float32:
        v = paddle.cast(v, dtype=paddle.float16)

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)

    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout)

    if is_casual:
        pass
    else:
        o, lse = sageattn_forward_casual_false(q_int8, k_int8, v, q_scale, k_scale, output_dtype="float16", tensor_layout=tensor_layout, return_lse=return_lse)

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o

#include <vector>
#include <map>
#include "triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel.h"
#include "paddle/extension.h"

std::map<std::vector<int>, int> map_problem_triton_fused_adaLN_scale_residual_fp16_2048_0_0;

CUdeviceptr get_tensor_ptr(const paddle::Tensor& input){
  if (input.type() == paddle::DataType::FLOAT16) {
    return (CUdeviceptr)(input.data<phi::dtype::float16>());
  } else if (input.type() == paddle::DataType::BFLOAT16) {
    return (CUdeviceptr)(input.data<phi::dtype::bfloat16>());
  } else if (input.type() == paddle::DataType::INT32) {
    return (CUdeviceptr)(input.data<int>());
  } else if (input.type() == paddle::DataType::FLOAT32) {
    return (CUdeviceptr)(input.data<float>());
  } else if (input.type() == paddle::DataType::UINT8) {
    return (CUdeviceptr)(input.data<uint8_t>());
  } else if (input.type() == paddle::DataType::INT8) {
    return (CUdeviceptr)(input.data<int8_t>());
  } else {
    assert(false);
    return (CUdeviceptr)(nullptr);
  }
} 


std::vector<paddle::Tensor> triton_fused_adaLN_scale_residual_fp16_2048_0_0_func(
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

  std::vector<int> problem_size = {M};
  auto run_triton_kernel = [&](int algo_id) -> CUresult{
      return triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel(run_stream,
                                               x_ptr,mha_out_ptr,gate_msa_ptr,scale_mlp_ptr,shift_mlp_ptr,weight_ptr,bias_ptr,resi_out_ptr,adaLN_out_ptr,M,N,seq_size,epsilon,

                                               algo_id);
  };

  if (!map_problem_triton_fused_adaLN_scale_residual_fp16_2048_0_0.count(problem_size)) {
    std::cout << "we are tuning for triton_fused_adaLN_scale_residual_fp16_2048_0_0 which key is: {";
    for (int i = 0; i < problem_size.size(); i++) {
        std::cout << problem_size[i] << ", ";
    }
    std::cout << "}" << std::endl;

    float min_time = 10000.f;
    int select_id = -1;
    constexpr int WARMUP = 5;
    constexpr int REPEAT = 10;

    for (int algo_id = 0; algo_id < triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_get_num_algos(); ++algo_id) {
        cudaEvent_t beg[REPEAT];
        cudaEvent_t end[REPEAT];
        float elapsed_times[REPEAT];

        auto status = CUDA_SUCCESS;

        for (int ii = 0; ii < WARMUP + REPEAT; ii++) {
            int repeat_id = ii - WARMUP;

            if (repeat_id >= 0) {
                (cudaEventCreate(beg + repeat_id));
                (cudaEventCreate(end + repeat_id));
                (cudaEventRecord(beg[repeat_id]));
            }

            auto flush_l2_cache = paddle::full(
                {10 * 1024 * 1024}, 0, paddle::DataType::INT32, x.place());
            // std::cout << &flush_l2_cache  << std::endl;
            // this is used when out is need to be reset to zero, such as split-k gemm.
            ;

            status = run_triton_kernel(algo_id);
            // assert(status == CUDA_SUCCESS);

            if (repeat_id >= 0) {
                (cudaEventRecord(end[repeat_id]));
                (cudaEventSynchronize(end[repeat_id]));
                (cudaEventElapsedTime(
                    elapsed_times + repeat_id, beg[repeat_id], end[repeat_id]));
            }
        }

        float avg_elapsed_time = 0.f;
        for (int ii = 0; ii < REPEAT; ++ii) {
            avg_elapsed_time += elapsed_times[ii];
        }

        std::cout << "algo id " << algo_id << " costs " << avg_elapsed_time << " ms" << std::endl;

        if (avg_elapsed_time < min_time && status == CUDA_SUCCESS) {
            min_time = avg_elapsed_time;
            select_id = algo_id;
        }
    }

    map_problem_triton_fused_adaLN_scale_residual_fp16_2048_0_0[problem_size] = select_id;
    std::cout << "select algo id: " << select_id << std::endl;
    ;
  }

  if (map_problem_triton_fused_adaLN_scale_residual_fp16_2048_0_0.count(problem_size)) {
    int algo_id = map_problem_triton_fused_adaLN_scale_residual_fp16_2048_0_0[problem_size];
    auto status = run_triton_kernel(algo_id);
    assert(status == CUDA_SUCCESS);
  }

    return {resi_out, adaLN_out};
}

std::vector<std::vector<int64_t>> triton_fused_adaLN_scale_residual_fp16_2048_0_0_InferShape(
        const std::vector<int64_t>& A_shape) {
  return {A_shape, A_shape};
}

std::vector<paddle::DataType> triton_fused_adaLN_scale_residual_fp16_2048_0_0_InferDtype(const paddle::DataType& A_dtype) {
    return {A_dtype, A_dtype};
}

PD_BUILD_OP(triton_fused_adaLN_scale_residual_fp16_2048_0_0)
    .Inputs({"x", "mha_out", "gate_msa", "scale_mlp", "shift_mlp", paddle::Optional("weight"), paddle::Optional("bias")})
    .Outputs({"resi_out", "adaLN_out"})
    .SetKernelFn(PD_KERNEL(triton_fused_adaLN_scale_residual_fp16_2048_0_0_func))
    .Attrs({"epsilon: float"})
    .SetInferDtypeFn(PD_INFER_DTYPE(triton_fused_adaLN_scale_residual_fp16_2048_0_0_InferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(triton_fused_adaLN_scale_residual_fp16_2048_0_0_InferShape));

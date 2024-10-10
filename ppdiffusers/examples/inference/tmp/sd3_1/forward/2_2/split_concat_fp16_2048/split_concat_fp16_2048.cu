#include <vector>
#include <map>
#include "split_concat_fp16_2048_kernel.h"
#include "paddle/extension.h"

std::map<std::vector<int>, int> map_problem_split_concat_fp16_2048;

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
std::vector<paddle::Tensor> split_concat_fp16_2048_func(
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
  

  std::vector<int> problem_size = {1};
  auto run_triton_kernel = [&](int algo_id) -> CUresult{
      return split_concat_fp16_2048_kernel(run_stream,
                                               out0,out1,out2,qkv,eqkv,batch,seq_qkv,seq_eqkv,output_hidden,

                                               algo_id);
  };

  if (!map_problem_split_concat_fp16_2048.count(problem_size)) {
    std::cout << "we are tuning for split_concat_fp16_2048 which key is: {";
    for (int i = 0; i < problem_size.size(); i++) {
        std::cout << problem_size[i] << ", ";
    }
    std::cout << "}" << std::endl;

    float min_time = 10000.f;
    int select_id = -1;
    constexpr int WARMUP = 5;
    constexpr int REPEAT = 10;

    for (int algo_id = 0; algo_id < split_concat_fp16_2048_kernel_get_num_algos(); ++algo_id) {
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

    map_problem_split_concat_fp16_2048[problem_size] = select_id;
    std::cout << "select algo id: " << select_id << std::endl;
    ;
  }

  if (map_problem_split_concat_fp16_2048.count(problem_size)) {
    int algo_id = map_problem_split_concat_fp16_2048[problem_size];
    auto status = run_triton_kernel(algo_id);
    assert(status == CUDA_SUCCESS);
  }

    return {out0_tensor, out1_tensor, out2_tensor};
}

std::vector<std::vector<int64_t>> split_concat_fp16_2048_InferShape(
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

std::vector<paddle::DataType> split_concat_fp16_2048_InferDtype(const paddle::DataType& A_dtype) {
    return {A_dtype, A_dtype, A_dtype};
}

PD_BUILD_OP(split_concat_fp16_2048)
    .Inputs({"x", "y"})
    .Outputs({"out0_tensor", "out1_tensor", "out2_tensor"})
    .SetKernelFn(PD_KERNEL(split_concat_fp16_2048_func))
    .SetInferDtypeFn(PD_INFER_DTYPE(split_concat_fp16_2048_InferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(split_concat_fp16_2048_InferShape));

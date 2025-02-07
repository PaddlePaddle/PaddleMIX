#include <vector>
#include <map>
#include "ln_partial_rotary_emb_fp16_64_kernel.h"
#include "paddle/extension.h"

std::map<std::vector<int>, int> map_problem_ln_partial_rotary_emb_fp16_64;

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

int triton_cdiv(int x, int y) {
    int result = (x + y - 1) / y;
    return (int)(result);
}

std::vector<paddle::Tensor> ln_partial_rotary_emb_fp16_64_func(const paddle::Tensor & q,const paddle::Tensor & k,const paddle::Tensor & text_seq_length_tensor,const paddle::Tensor & cos,const paddle::Tensor & sin,const paddle::Tensor & q_norm_weight,const paddle::Tensor & q_norm_bias,const paddle::Tensor & k_norm_weight,const paddle::Tensor & k_norm_bias,float norm_eps) {
  
    // 这个名字必须保证和kernel形式参数一致！
    int batch = q.dims()[0];
    int num_heads = q.dims()[1];
    int seq_len =  q.dims()[2];
    int HEAD_DIM =  q.dims()[3];
    int text_seq_length = text_seq_length_tensor.dims()[0];
    int n_elements = batch * num_heads * seq_len * HEAD_DIM;
    
  
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
        
  auto  run_stream = outk.stream();
  
  std::vector<int> problem_size = {1};
  auto run_triton_kernel = [&](int algo_id) -> CUresult{
      return ln_partial_rotary_emb_fp16_64_kernel(run_stream,
                                               q_ptr,k_ptr,cos_ptr,sin_ptr,q_norm_weight_ptr,q_norm_bias_ptr,k_norm_weight_ptr,k_norm_bias_ptr,outq_ptr,outk_ptr,text_seq_length,batch,num_heads,seq_len,n_elements,norm_eps,
                                               algo_id);
  };

  if (!map_problem_ln_partial_rotary_emb_fp16_64.count(problem_size)) {
    std::cout << "we are tuning for ln_partial_rotary_emb_fp16_64 which key is: {";
    for (int i = 0; i < problem_size.size(); i++) {
        std::cout << problem_size[i] << ", ";
    }
    std::cout << "}" << std::endl;

    float min_time = 10000.f;
    int select_id = -1;
    constexpr int WARMUP = 5;
    constexpr int REPEAT = 10;

    for (int algo_id = 0; algo_id < ln_partial_rotary_emb_fp16_64_kernel_get_num_algos(); ++algo_id) {
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
                {10 * 1024 * 1024}, 0, paddle::DataType::INT32, outk.place());
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

    map_problem_ln_partial_rotary_emb_fp16_64[problem_size] = select_id;
    std::cout << "select algo id: " << select_id << std::endl;
    ;
  }

  if (map_problem_ln_partial_rotary_emb_fp16_64.count(problem_size)) {
    int algo_id = map_problem_ln_partial_rotary_emb_fp16_64[problem_size];
    auto status = run_triton_kernel(algo_id);
    assert(status == CUDA_SUCCESS);
  }

  return {outq, outk};
}

std::vector<std::vector<int64_t>> ln_partial_rotary_emb_fp16_64_InferShape(const std::vector<int64_t>& A_shape) {return {A_shape,A_shape};}
 std::vector<paddle::DataType> ln_partial_rotary_emb_fp16_64_InferDtype(const paddle::DataType& A_dtype) {return {A_dtype,A_dtype};}
 

PD_BUILD_OP(ln_partial_rotary_emb_fp16_64)
    .Inputs({"q","k","text_seq_length_tensor","cos","sin","q_norm_weight","q_norm_bias","k_norm_weight","k_norm_bias"})
    .Outputs({"outq","outk"})
    .Attrs({"norm_eps: float"})
    .SetKernelFn(PD_KERNEL(ln_partial_rotary_emb_fp16_64_func))
    .SetInferDtypeFn(PD_INFER_DTYPE(ln_partial_rotary_emb_fp16_64_InferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(ln_partial_rotary_emb_fp16_64_InferShape));

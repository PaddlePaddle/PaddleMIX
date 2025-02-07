#include <cuda.h>
#include <stdint.h>
#include <assert.h>

// launcher for: ln_partial_rotary_emb_fp16_64_kernel_64_warps4xstages4
CUresult ln_partial_rotary_emb_fp16_64_kernel_9c1d1c4b_0d1d2d3d4d5d6d7d8d9d101112d13d14d15(CUstream stream, CUdeviceptr q_ptr, CUdeviceptr k_ptr, CUdeviceptr cos_ptr, CUdeviceptr sin_ptr, CUdeviceptr q_norm_weight_ptr, CUdeviceptr q_norm_bias_ptr, CUdeviceptr k_norm_weight_ptr, CUdeviceptr k_norm_bias_ptr, CUdeviceptr outq_ptr, CUdeviceptr outk_ptr, int32_t text_seq_length, int32_t batch, int32_t num_heads, int32_t seq_len, int32_t n_elements, float norm_eps);

CUresult ln_partial_rotary_emb_fp16_64_kernel_64_warps4xstages4(CUstream stream, CUdeviceptr q_ptr, CUdeviceptr k_ptr, CUdeviceptr cos_ptr, CUdeviceptr sin_ptr, CUdeviceptr q_norm_weight_ptr, CUdeviceptr q_norm_bias_ptr, CUdeviceptr k_norm_weight_ptr, CUdeviceptr k_norm_bias_ptr, CUdeviceptr outq_ptr, CUdeviceptr outk_ptr, int32_t text_seq_length, int32_t batch, int32_t num_heads, int32_t seq_len, int32_t n_elements, float norm_eps){
  if ((q_ptr % 16 == 0) && (k_ptr % 16 == 0) && (cos_ptr % 16 == 0) && (sin_ptr % 16 == 0) && (q_norm_weight_ptr % 16 == 0) && (q_norm_bias_ptr % 16 == 0) && (k_norm_weight_ptr % 16 == 0) && (k_norm_bias_ptr % 16 == 0) && (outq_ptr % 16 == 0) && (outk_ptr % 16 == 0) && (num_heads % 16 == 0) && (seq_len % 16 == 0) && (n_elements % 16 == 0))
    return ln_partial_rotary_emb_fp16_64_kernel_9c1d1c4b_0d1d2d3d4d5d6d7d8d9d101112d13d14d15(stream, q_ptr, k_ptr, cos_ptr, sin_ptr, q_norm_weight_ptr, q_norm_bias_ptr, k_norm_weight_ptr, k_norm_bias_ptr, outq_ptr, outk_ptr, text_seq_length, batch, num_heads, seq_len, n_elements, norm_eps);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: ln_partial_rotary_emb_fp16_64_kernel_64_warps4xstages4
void load_ln_partial_rotary_emb_fp16_64_kernel_9c1d1c4b_0d1d2d3d4d5d6d7d8d9d101112d13d14d15();
void load_ln_partial_rotary_emb_fp16_64_kernel_64_warps4xstages4() {
  load_ln_partial_rotary_emb_fp16_64_kernel_9c1d1c4b_0d1d2d3d4d5d6d7d8d9d101112d13d14d15();
}

// unload for: ln_partial_rotary_emb_fp16_64_kernel_64_warps4xstages4
void unload_ln_partial_rotary_emb_fp16_64_kernel_9c1d1c4b_0d1d2d3d4d5d6d7d8d9d101112d13d14d15();
void unload_ln_partial_rotary_emb_fp16_64_kernel_64_warps4xstages4() {
  unload_ln_partial_rotary_emb_fp16_64_kernel_9c1d1c4b_0d1d2d3d4d5d6d7d8d9d101112d13d14d15();
}

typedef CUresult (*kernel_func_t)(CUstream stream, CUdeviceptr q_ptr, CUdeviceptr k_ptr, CUdeviceptr cos_ptr, CUdeviceptr sin_ptr, CUdeviceptr q_norm_weight_ptr, CUdeviceptr q_norm_bias_ptr, CUdeviceptr k_norm_weight_ptr, CUdeviceptr k_norm_bias_ptr, CUdeviceptr outq_ptr, CUdeviceptr outk_ptr, int32_t text_seq_length, int32_t batch, int32_t num_heads, int32_t seq_len, int32_t n_elements, float norm_eps);
kernel_func_t ln_partial_rotary_emb_fp16_64_kernel_kernels[] = {
  ln_partial_rotary_emb_fp16_64_kernel_64_warps4xstages4,
};

int ln_partial_rotary_emb_fp16_64_kernel_get_num_algos(void){
  return (int)(sizeof(ln_partial_rotary_emb_fp16_64_kernel_kernels) / sizeof(ln_partial_rotary_emb_fp16_64_kernel_kernels[0]));
}

CUresult ln_partial_rotary_emb_fp16_64_kernel(CUstream stream, CUdeviceptr q_ptr, CUdeviceptr k_ptr, CUdeviceptr cos_ptr, CUdeviceptr sin_ptr, CUdeviceptr q_norm_weight_ptr, CUdeviceptr q_norm_bias_ptr, CUdeviceptr k_norm_weight_ptr, CUdeviceptr k_norm_bias_ptr, CUdeviceptr outq_ptr, CUdeviceptr outk_ptr, int32_t text_seq_length, int32_t batch, int32_t num_heads, int32_t seq_len, int32_t n_elements, float norm_eps, int algo_id){
  assert (algo_id < (int)sizeof(ln_partial_rotary_emb_fp16_64_kernel_kernels));
  return ln_partial_rotary_emb_fp16_64_kernel_kernels[algo_id](stream, q_ptr, k_ptr, cos_ptr, sin_ptr, q_norm_weight_ptr, q_norm_bias_ptr, k_norm_weight_ptr, k_norm_bias_ptr, outq_ptr, outk_ptr, text_seq_length, batch, num_heads, seq_len, n_elements, norm_eps);
}

void load_ln_partial_rotary_emb_fp16_64_kernel(void){
  load_ln_partial_rotary_emb_fp16_64_kernel_64_warps4xstages4();
}

void unload_ln_partial_rotary_emb_fp16_64_kernel(void){
  unload_ln_partial_rotary_emb_fp16_64_kernel_64_warps4xstages4();
}


CUresult ln_partial_rotary_emb_fp16_64_kernel_default(CUstream stream, CUdeviceptr q_ptr, CUdeviceptr k_ptr, CUdeviceptr cos_ptr, CUdeviceptr sin_ptr, CUdeviceptr q_norm_weight_ptr, CUdeviceptr q_norm_bias_ptr, CUdeviceptr k_norm_weight_ptr, CUdeviceptr k_norm_bias_ptr, CUdeviceptr outq_ptr, CUdeviceptr outk_ptr, int32_t text_seq_length, int32_t batch, int32_t num_heads, int32_t seq_len, int32_t n_elements, float norm_eps){
  return ln_partial_rotary_emb_fp16_64_kernel(stream, q_ptr, k_ptr, cos_ptr, sin_ptr, q_norm_weight_ptr, q_norm_bias_ptr, k_norm_weight_ptr, k_norm_bias_ptr, outq_ptr, outk_ptr, text_seq_length, batch, num_heads, seq_len, n_elements, norm_eps, 0);
}

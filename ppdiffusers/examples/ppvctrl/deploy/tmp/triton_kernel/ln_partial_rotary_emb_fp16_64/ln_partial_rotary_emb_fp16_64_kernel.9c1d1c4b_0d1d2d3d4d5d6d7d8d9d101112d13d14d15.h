#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#endif

void unload_ln_partial_rotary_emb_fp16_64_kernel_9c1d1c4b_0d1d2d3d4d5d6d7d8d9d101112d13d14d15(void);
void load_ln_partial_rotary_emb_fp16_64_kernel_9c1d1c4b_0d1d2d3d4d5d6d7d8d9d101112d13d14d15(void);
// tt-linker: ln_partial_rotary_emb_fp16_64_kernel_9c1d1c4b_0d1d2d3d4d5d6d7d8d9d101112d13d14d15:CUdeviceptr q_ptr, CUdeviceptr k_ptr, CUdeviceptr cos_ptr, CUdeviceptr sin_ptr, CUdeviceptr q_norm_weight_ptr, CUdeviceptr q_norm_bias_ptr, CUdeviceptr k_norm_weight_ptr, CUdeviceptr k_norm_bias_ptr, CUdeviceptr outq_ptr, CUdeviceptr outk_ptr, int32_t text_seq_length, int32_t batch, int32_t num_heads, int32_t seq_len, int32_t n_elements, float norm_eps:64_warps4xstages4
CUresult ln_partial_rotary_emb_fp16_64_kernel_9c1d1c4b_0d1d2d3d4d5d6d7d8d9d101112d13d14d15(CUstream stream, CUdeviceptr q_ptr, CUdeviceptr k_ptr, CUdeviceptr cos_ptr, CUdeviceptr sin_ptr, CUdeviceptr q_norm_weight_ptr, CUdeviceptr q_norm_bias_ptr, CUdeviceptr k_norm_weight_ptr, CUdeviceptr k_norm_bias_ptr, CUdeviceptr outq_ptr, CUdeviceptr outk_ptr, int32_t text_seq_length, int32_t batch, int32_t num_heads, int32_t seq_len, int32_t n_elements, float norm_eps);

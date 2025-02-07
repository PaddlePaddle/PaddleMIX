#include <cuda.h>

CUresult ln_partial_rotary_emb_fp16_64_kernel_64_warps4xstages4(CUstream stream, CUdeviceptr q_ptr, CUdeviceptr k_ptr, CUdeviceptr cos_ptr, CUdeviceptr sin_ptr, CUdeviceptr q_norm_weight_ptr, CUdeviceptr q_norm_bias_ptr, CUdeviceptr k_norm_weight_ptr, CUdeviceptr k_norm_bias_ptr, CUdeviceptr outq_ptr, CUdeviceptr outk_ptr, int32_t text_seq_length, int32_t batch, int32_t num_heads, int32_t seq_len, int32_t n_elements, float norm_eps);
void load_ln_partial_rotary_emb_fp16_64_kernel_64_warps4xstages4();
void unload_ln_partial_rotary_emb_fp16_64_kernel_64_warps4xstages4();
    
int ln_partial_rotary_emb_fp16_64_kernel_get_num_algos(void);

CUresult ln_partial_rotary_emb_fp16_64_kernel_default(CUstream stream, CUdeviceptr q_ptr, CUdeviceptr k_ptr, CUdeviceptr cos_ptr, CUdeviceptr sin_ptr, CUdeviceptr q_norm_weight_ptr, CUdeviceptr q_norm_bias_ptr, CUdeviceptr k_norm_weight_ptr, CUdeviceptr k_norm_bias_ptr, CUdeviceptr outq_ptr, CUdeviceptr outk_ptr, int32_t text_seq_length, int32_t batch, int32_t num_heads, int32_t seq_len, int32_t n_elements, float norm_eps);
CUresult ln_partial_rotary_emb_fp16_64_kernel(CUstream stream, CUdeviceptr q_ptr, CUdeviceptr k_ptr, CUdeviceptr cos_ptr, CUdeviceptr sin_ptr, CUdeviceptr q_norm_weight_ptr, CUdeviceptr q_norm_bias_ptr, CUdeviceptr k_norm_weight_ptr, CUdeviceptr k_norm_bias_ptr, CUdeviceptr outq_ptr, CUdeviceptr outk_ptr, int32_t text_seq_length, int32_t batch, int32_t num_heads, int32_t seq_len, int32_t n_elements, float norm_eps, int algo_id);
void load_ln_partial_rotary_emb_fp16_64_kernel();
void unload_ln_partial_rotary_emb_fp16_64_kernel();
    
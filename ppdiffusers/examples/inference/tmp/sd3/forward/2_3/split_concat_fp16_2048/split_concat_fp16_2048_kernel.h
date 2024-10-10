#include <cuda.h>

CUresult split_concat_fp16_2048_kernel_2048_warps4xstages4(CUstream stream, CUdeviceptr out0, CUdeviceptr out1, CUdeviceptr out2, CUdeviceptr qkv, CUdeviceptr eqkv, int32_t batch, int32_t seq_qkv, int32_t seq_eqkv, int32_t output_hidden);
void load_split_concat_fp16_2048_kernel_2048_warps4xstages4();
void unload_split_concat_fp16_2048_kernel_2048_warps4xstages4();
    
int split_concat_fp16_2048_kernel_get_num_algos(void);

CUresult split_concat_fp16_2048_kernel_default(CUstream stream, CUdeviceptr out0, CUdeviceptr out1, CUdeviceptr out2, CUdeviceptr qkv, CUdeviceptr eqkv, int32_t batch, int32_t seq_qkv, int32_t seq_eqkv, int32_t output_hidden);
CUresult split_concat_fp16_2048_kernel(CUstream stream, CUdeviceptr out0, CUdeviceptr out1, CUdeviceptr out2, CUdeviceptr qkv, CUdeviceptr eqkv, int32_t batch, int32_t seq_qkv, int32_t seq_eqkv, int32_t output_hidden, int algo_id);
void load_split_concat_fp16_2048_kernel();
void unload_split_concat_fp16_2048_kernel();
    
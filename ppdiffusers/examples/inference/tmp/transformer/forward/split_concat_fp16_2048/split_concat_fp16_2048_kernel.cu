#include <cuda.h>
#include <stdint.h>
#include <assert.h>

// launcher for: split_concat_fp16_2048_kernel_2048_warps4xstages4
CUresult split_concat_fp16_2048_kernel_9dc56806_0d1d2d3d4d56d78d(CUstream stream, CUdeviceptr out0, CUdeviceptr out1, CUdeviceptr out2, CUdeviceptr qkv, CUdeviceptr eqkv, int32_t batch, int32_t seq_qkv, int32_t seq_eqkv, int32_t output_hidden);

CUresult split_concat_fp16_2048_kernel_2048_warps4xstages4(CUstream stream, CUdeviceptr out0, CUdeviceptr out1, CUdeviceptr out2, CUdeviceptr qkv, CUdeviceptr eqkv, int32_t batch, int32_t seq_qkv, int32_t seq_eqkv, int32_t output_hidden){
  if ((out0 % 16 == 0) && (out1 % 16 == 0) && (out2 % 16 == 0) && (qkv % 16 == 0) && (eqkv % 16 == 0) && (seq_qkv % 16 == 0) && (output_hidden % 16 == 0))
    return split_concat_fp16_2048_kernel_9dc56806_0d1d2d3d4d56d78d(stream, out0, out1, out2, qkv, eqkv, batch, seq_qkv, seq_eqkv, output_hidden);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: split_concat_fp16_2048_kernel_2048_warps4xstages4
void load_split_concat_fp16_2048_kernel_9dc56806_0d1d2d3d4d56d78d();
void load_split_concat_fp16_2048_kernel_2048_warps4xstages4() {
  load_split_concat_fp16_2048_kernel_9dc56806_0d1d2d3d4d56d78d();
}

// unload for: split_concat_fp16_2048_kernel_2048_warps4xstages4
void unload_split_concat_fp16_2048_kernel_9dc56806_0d1d2d3d4d56d78d();
void unload_split_concat_fp16_2048_kernel_2048_warps4xstages4() {
  unload_split_concat_fp16_2048_kernel_9dc56806_0d1d2d3d4d56d78d();
}

typedef CUresult (*kernel_func_t)(CUstream stream, CUdeviceptr out0, CUdeviceptr out1, CUdeviceptr out2, CUdeviceptr qkv, CUdeviceptr eqkv, int32_t batch, int32_t seq_qkv, int32_t seq_eqkv, int32_t output_hidden);
kernel_func_t split_concat_fp16_2048_kernel_kernels[] = {
  split_concat_fp16_2048_kernel_2048_warps4xstages4,
};

int split_concat_fp16_2048_kernel_get_num_algos(void){
  return (int)(sizeof(split_concat_fp16_2048_kernel_kernels) / sizeof(split_concat_fp16_2048_kernel_kernels[0]));
}

CUresult split_concat_fp16_2048_kernel(CUstream stream, CUdeviceptr out0, CUdeviceptr out1, CUdeviceptr out2, CUdeviceptr qkv, CUdeviceptr eqkv, int32_t batch, int32_t seq_qkv, int32_t seq_eqkv, int32_t output_hidden, int algo_id){
  assert (algo_id < (int)sizeof(split_concat_fp16_2048_kernel_kernels));
  return split_concat_fp16_2048_kernel_kernels[algo_id](stream, out0, out1, out2, qkv, eqkv, batch, seq_qkv, seq_eqkv, output_hidden);
}

void load_split_concat_fp16_2048_kernel(void){
  load_split_concat_fp16_2048_kernel_2048_warps4xstages4();
}

void unload_split_concat_fp16_2048_kernel(void){
  unload_split_concat_fp16_2048_kernel_2048_warps4xstages4();
}


CUresult split_concat_fp16_2048_kernel_default(CUstream stream, CUdeviceptr out0, CUdeviceptr out1, CUdeviceptr out2, CUdeviceptr qkv, CUdeviceptr eqkv, int32_t batch, int32_t seq_qkv, int32_t seq_eqkv, int32_t output_hidden){
  return split_concat_fp16_2048_kernel(stream, out0, out1, out2, qkv, eqkv, batch, seq_qkv, seq_eqkv, output_hidden, 0);
}

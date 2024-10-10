#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#endif

void unload_split_concat_fp16_2048_kernel_9dc56806_0d1d2d3d4d56d78d(void);
void load_split_concat_fp16_2048_kernel_9dc56806_0d1d2d3d4d56d78d(void);
// tt-linker: split_concat_fp16_2048_kernel_9dc56806_0d1d2d3d4d56d78d:CUdeviceptr out0, CUdeviceptr out1, CUdeviceptr out2, CUdeviceptr qkv, CUdeviceptr eqkv, int32_t batch, int32_t seq_qkv, int32_t seq_eqkv, int32_t output_hidden:2048_warps4xstages4
CUresult split_concat_fp16_2048_kernel_9dc56806_0d1d2d3d4d56d78d(CUstream stream, CUdeviceptr out0, CUdeviceptr out1, CUdeviceptr out2, CUdeviceptr qkv, CUdeviceptr eqkv, int32_t batch, int32_t seq_qkv, int32_t seq_eqkv, int32_t output_hidden);

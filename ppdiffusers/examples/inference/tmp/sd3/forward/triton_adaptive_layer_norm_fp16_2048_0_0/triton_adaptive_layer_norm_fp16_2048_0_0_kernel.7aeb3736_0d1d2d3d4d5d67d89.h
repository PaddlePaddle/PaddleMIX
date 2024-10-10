#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#endif

void unload_triton_adaptive_layer_norm_fp16_2048_0_0_kernel_7aeb3736_0d1d2d3d4d5d67d89(void);
void load_triton_adaptive_layer_norm_fp16_2048_0_0_kernel_7aeb3736_0d1d2d3d4d5d67d89(void);
// tt-linker: triton_adaptive_layer_norm_fp16_2048_0_0_kernel_7aeb3736_0d1d2d3d4d5d67d89:CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr scale_ptr, CUdeviceptr shift_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon:2048x0x0_warps4xstages4
CUresult triton_adaptive_layer_norm_fp16_2048_0_0_kernel_7aeb3736_0d1d2d3d4d5d67d89(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr scale_ptr, CUdeviceptr shift_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);

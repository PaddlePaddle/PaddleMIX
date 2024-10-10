#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#endif

void unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_b7aadab9_0d1d2d3d4d5d6d7d8d910d1112(void);
void load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_b7aadab9_0d1d2d3d4d5d6d7d8d910d1112(void);
// tt-linker: triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_b7aadab9_0d1d2d3d4d5d6d7d8d910d1112:CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon:2048x0x0_warps16xstages4
CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_b7aadab9_0d1d2d3d4d5d6d7d8d910d1112(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);

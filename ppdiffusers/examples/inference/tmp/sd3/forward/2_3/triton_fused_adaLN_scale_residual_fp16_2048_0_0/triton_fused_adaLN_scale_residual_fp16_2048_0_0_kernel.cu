#include <cuda.h>
#include <stdint.h>
#include <assert.h>

// launcher for: triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps16xstages4
CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_b7aadab9_0d1d2d3d4d5d6d7d8d910d1112(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);

CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps16xstages4(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon){
  if ((x_ptr % 16 == 0) && (mha_out_ptr % 16 == 0) && (gate_msa_ptr % 16 == 0) && (scale_mlp_ptr % 16 == 0) && (shift_mlp_ptr % 16 == 0) && (weight_ptr % 16 == 0) && (bias_ptr % 16 == 0) && (resi_out_ptr % 16 == 0) && (adaLN_out_ptr % 16 == 0) && (N % 16 == 0))
    return triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_b7aadab9_0d1d2d3d4d5d6d7d8d910d1112(stream, x_ptr, mha_out_ptr, gate_msa_ptr, scale_mlp_ptr, shift_mlp_ptr, weight_ptr, bias_ptr, resi_out_ptr, adaLN_out_ptr, M, N, seq_size, epsilon);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps16xstages4
void load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_b7aadab9_0d1d2d3d4d5d6d7d8d910d1112();
void load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps16xstages4() {
  load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_b7aadab9_0d1d2d3d4d5d6d7d8d910d1112();
}

// unload for: triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps16xstages4
void unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_b7aadab9_0d1d2d3d4d5d6d7d8d910d1112();
void unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps16xstages4() {
  unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_b7aadab9_0d1d2d3d4d5d6d7d8d910d1112();
}

// launcher for: triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps8xstages4
CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_bc21a394_0d1d2d3d4d5d6d7d8d910d1112(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);

CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps8xstages4(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon){
  if ((x_ptr % 16 == 0) && (mha_out_ptr % 16 == 0) && (gate_msa_ptr % 16 == 0) && (scale_mlp_ptr % 16 == 0) && (shift_mlp_ptr % 16 == 0) && (weight_ptr % 16 == 0) && (bias_ptr % 16 == 0) && (resi_out_ptr % 16 == 0) && (adaLN_out_ptr % 16 == 0) && (N % 16 == 0))
    return triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_bc21a394_0d1d2d3d4d5d6d7d8d910d1112(stream, x_ptr, mha_out_ptr, gate_msa_ptr, scale_mlp_ptr, shift_mlp_ptr, weight_ptr, bias_ptr, resi_out_ptr, adaLN_out_ptr, M, N, seq_size, epsilon);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps8xstages4
void load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_bc21a394_0d1d2d3d4d5d6d7d8d910d1112();
void load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps8xstages4() {
  load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_bc21a394_0d1d2d3d4d5d6d7d8d910d1112();
}

// unload for: triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps8xstages4
void unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_bc21a394_0d1d2d3d4d5d6d7d8d910d1112();
void unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps8xstages4() {
  unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_bc21a394_0d1d2d3d4d5d6d7d8d910d1112();
}

// launcher for: triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps32xstages4
CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_c8a66c04_0d1d2d3d4d5d6d7d8d910d1112(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);

CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps32xstages4(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon){
  if ((x_ptr % 16 == 0) && (mha_out_ptr % 16 == 0) && (gate_msa_ptr % 16 == 0) && (scale_mlp_ptr % 16 == 0) && (shift_mlp_ptr % 16 == 0) && (weight_ptr % 16 == 0) && (bias_ptr % 16 == 0) && (resi_out_ptr % 16 == 0) && (adaLN_out_ptr % 16 == 0) && (N % 16 == 0))
    return triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_c8a66c04_0d1d2d3d4d5d6d7d8d910d1112(stream, x_ptr, mha_out_ptr, gate_msa_ptr, scale_mlp_ptr, shift_mlp_ptr, weight_ptr, bias_ptr, resi_out_ptr, adaLN_out_ptr, M, N, seq_size, epsilon);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps32xstages4
void load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_c8a66c04_0d1d2d3d4d5d6d7d8d910d1112();
void load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps32xstages4() {
  load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_c8a66c04_0d1d2d3d4d5d6d7d8d910d1112();
}

// unload for: triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps32xstages4
void unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_c8a66c04_0d1d2d3d4d5d6d7d8d910d1112();
void unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps32xstages4() {
  unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_c8a66c04_0d1d2d3d4d5d6d7d8d910d1112();
}

// launcher for: triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps4xstages4
CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_f41eb178_0d1d2d3d4d5d6d7d8d910d1112(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);

CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps4xstages4(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon){
  if ((x_ptr % 16 == 0) && (mha_out_ptr % 16 == 0) && (gate_msa_ptr % 16 == 0) && (scale_mlp_ptr % 16 == 0) && (shift_mlp_ptr % 16 == 0) && (weight_ptr % 16 == 0) && (bias_ptr % 16 == 0) && (resi_out_ptr % 16 == 0) && (adaLN_out_ptr % 16 == 0) && (N % 16 == 0))
    return triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_f41eb178_0d1d2d3d4d5d6d7d8d910d1112(stream, x_ptr, mha_out_ptr, gate_msa_ptr, scale_mlp_ptr, shift_mlp_ptr, weight_ptr, bias_ptr, resi_out_ptr, adaLN_out_ptr, M, N, seq_size, epsilon);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps4xstages4
void load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_f41eb178_0d1d2d3d4d5d6d7d8d910d1112();
void load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps4xstages4() {
  load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_f41eb178_0d1d2d3d4d5d6d7d8d910d1112();
}

// unload for: triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps4xstages4
void unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_f41eb178_0d1d2d3d4d5d6d7d8d910d1112();
void unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps4xstages4() {
  unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_f41eb178_0d1d2d3d4d5d6d7d8d910d1112();
}

// launcher for: triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps2xstages4
CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_f4315a71_0d1d2d3d4d5d6d7d8d910d1112(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);

CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps2xstages4(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon){
  if ((x_ptr % 16 == 0) && (mha_out_ptr % 16 == 0) && (gate_msa_ptr % 16 == 0) && (scale_mlp_ptr % 16 == 0) && (shift_mlp_ptr % 16 == 0) && (weight_ptr % 16 == 0) && (bias_ptr % 16 == 0) && (resi_out_ptr % 16 == 0) && (adaLN_out_ptr % 16 == 0) && (N % 16 == 0))
    return triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_f4315a71_0d1d2d3d4d5d6d7d8d910d1112(stream, x_ptr, mha_out_ptr, gate_msa_ptr, scale_mlp_ptr, shift_mlp_ptr, weight_ptr, bias_ptr, resi_out_ptr, adaLN_out_ptr, M, N, seq_size, epsilon);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps2xstages4
void load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_f4315a71_0d1d2d3d4d5d6d7d8d910d1112();
void load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps2xstages4() {
  load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_f4315a71_0d1d2d3d4d5d6d7d8d910d1112();
}

// unload for: triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps2xstages4
void unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_f4315a71_0d1d2d3d4d5d6d7d8d910d1112();
void unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps2xstages4() {
  unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_f4315a71_0d1d2d3d4d5d6d7d8d910d1112();
}

typedef CUresult (*kernel_func_t)(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);
kernel_func_t triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_kernels[] = {
  triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps16xstages4,
  triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps8xstages4,
  triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps32xstages4,
  triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps4xstages4,
  triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps2xstages4,
};

int triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_get_num_algos(void){
  return (int)(sizeof(triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_kernels) / sizeof(triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_kernels[0]));
}

CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon, int algo_id){
  assert (algo_id < (int)sizeof(triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_kernels));
  return triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_kernels[algo_id](stream, x_ptr, mha_out_ptr, gate_msa_ptr, scale_mlp_ptr, shift_mlp_ptr, weight_ptr, bias_ptr, resi_out_ptr, adaLN_out_ptr, M, N, seq_size, epsilon);
}

void load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel(void){
  load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps16xstages4();
  load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps8xstages4();
  load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps32xstages4();
  load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps4xstages4();
  load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps2xstages4();
}

void unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel(void){
  unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps16xstages4();
  unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps8xstages4();
  unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps32xstages4();
  unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps4xstages4();
  unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps2xstages4();
}


CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_default(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon){
  return triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel(stream, x_ptr, mha_out_ptr, gate_msa_ptr, scale_mlp_ptr, shift_mlp_ptr, weight_ptr, bias_ptr, resi_out_ptr, adaLN_out_ptr, M, N, seq_size, epsilon, 0);
}

#include <cuda.h>
#include <stdint.h>
#include <assert.h>

// launcher for: triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps2xstages4
CUresult triton_adaptive_layer_norm_fp16_2048_0_1_kernel_023df1ab_0d1d2d3d4d5d67d89(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr scale_ptr, CUdeviceptr shift_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);

CUresult triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps2xstages4(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr scale_ptr, CUdeviceptr shift_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon){
  if ((x_ptr % 16 == 0) && (y_ptr % 16 == 0) && (weight_ptr % 16 == 0) && (bias_ptr % 16 == 0) && (scale_ptr % 16 == 0) && (shift_ptr % 16 == 0) && (N % 16 == 0))
    return triton_adaptive_layer_norm_fp16_2048_0_1_kernel_023df1ab_0d1d2d3d4d5d67d89(stream, x_ptr, y_ptr, weight_ptr, bias_ptr, scale_ptr, shift_ptr, M, N, seq_size, epsilon);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps2xstages4
void load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_023df1ab_0d1d2d3d4d5d67d89();
void load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps2xstages4() {
  load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_023df1ab_0d1d2d3d4d5d67d89();
}

// unload for: triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps2xstages4
void unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_023df1ab_0d1d2d3d4d5d67d89();
void unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps2xstages4() {
  unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_023df1ab_0d1d2d3d4d5d67d89();
}

// launcher for: triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps4xstages4
CUresult triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2a03fae1_0d1d2d3d4d5d67d89(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr scale_ptr, CUdeviceptr shift_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);

CUresult triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps4xstages4(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr scale_ptr, CUdeviceptr shift_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon){
  if ((x_ptr % 16 == 0) && (y_ptr % 16 == 0) && (weight_ptr % 16 == 0) && (bias_ptr % 16 == 0) && (scale_ptr % 16 == 0) && (shift_ptr % 16 == 0) && (N % 16 == 0))
    return triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2a03fae1_0d1d2d3d4d5d67d89(stream, x_ptr, y_ptr, weight_ptr, bias_ptr, scale_ptr, shift_ptr, M, N, seq_size, epsilon);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps4xstages4
void load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2a03fae1_0d1d2d3d4d5d67d89();
void load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps4xstages4() {
  load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2a03fae1_0d1d2d3d4d5d67d89();
}

// unload for: triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps4xstages4
void unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2a03fae1_0d1d2d3d4d5d67d89();
void unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps4xstages4() {
  unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2a03fae1_0d1d2d3d4d5d67d89();
}

// launcher for: triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps8xstages4
CUresult triton_adaptive_layer_norm_fp16_2048_0_1_kernel_30f7541a_0d1d2d3d4d5d67d89(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr scale_ptr, CUdeviceptr shift_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);

CUresult triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps8xstages4(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr scale_ptr, CUdeviceptr shift_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon){
  if ((x_ptr % 16 == 0) && (y_ptr % 16 == 0) && (weight_ptr % 16 == 0) && (bias_ptr % 16 == 0) && (scale_ptr % 16 == 0) && (shift_ptr % 16 == 0) && (N % 16 == 0))
    return triton_adaptive_layer_norm_fp16_2048_0_1_kernel_30f7541a_0d1d2d3d4d5d67d89(stream, x_ptr, y_ptr, weight_ptr, bias_ptr, scale_ptr, shift_ptr, M, N, seq_size, epsilon);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps8xstages4
void load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_30f7541a_0d1d2d3d4d5d67d89();
void load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps8xstages4() {
  load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_30f7541a_0d1d2d3d4d5d67d89();
}

// unload for: triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps8xstages4
void unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_30f7541a_0d1d2d3d4d5d67d89();
void unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps8xstages4() {
  unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_30f7541a_0d1d2d3d4d5d67d89();
}

// launcher for: triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps16xstages4
CUresult triton_adaptive_layer_norm_fp16_2048_0_1_kernel_cc9a52db_0d1d2d3d4d5d67d89(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr scale_ptr, CUdeviceptr shift_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);

CUresult triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps16xstages4(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr scale_ptr, CUdeviceptr shift_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon){
  if ((x_ptr % 16 == 0) && (y_ptr % 16 == 0) && (weight_ptr % 16 == 0) && (bias_ptr % 16 == 0) && (scale_ptr % 16 == 0) && (shift_ptr % 16 == 0) && (N % 16 == 0))
    return triton_adaptive_layer_norm_fp16_2048_0_1_kernel_cc9a52db_0d1d2d3d4d5d67d89(stream, x_ptr, y_ptr, weight_ptr, bias_ptr, scale_ptr, shift_ptr, M, N, seq_size, epsilon);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps16xstages4
void load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_cc9a52db_0d1d2d3d4d5d67d89();
void load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps16xstages4() {
  load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_cc9a52db_0d1d2d3d4d5d67d89();
}

// unload for: triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps16xstages4
void unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_cc9a52db_0d1d2d3d4d5d67d89();
void unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps16xstages4() {
  unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_cc9a52db_0d1d2d3d4d5d67d89();
}

// launcher for: triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps32xstages4
CUresult triton_adaptive_layer_norm_fp16_2048_0_1_kernel_fdea6b42_0d1d2d3d4d5d67d89(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr scale_ptr, CUdeviceptr shift_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);

CUresult triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps32xstages4(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr scale_ptr, CUdeviceptr shift_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon){
  if ((x_ptr % 16 == 0) && (y_ptr % 16 == 0) && (weight_ptr % 16 == 0) && (bias_ptr % 16 == 0) && (scale_ptr % 16 == 0) && (shift_ptr % 16 == 0) && (N % 16 == 0))
    return triton_adaptive_layer_norm_fp16_2048_0_1_kernel_fdea6b42_0d1d2d3d4d5d67d89(stream, x_ptr, y_ptr, weight_ptr, bias_ptr, scale_ptr, shift_ptr, M, N, seq_size, epsilon);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps32xstages4
void load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_fdea6b42_0d1d2d3d4d5d67d89();
void load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps32xstages4() {
  load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_fdea6b42_0d1d2d3d4d5d67d89();
}

// unload for: triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps32xstages4
void unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_fdea6b42_0d1d2d3d4d5d67d89();
void unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps32xstages4() {
  unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_fdea6b42_0d1d2d3d4d5d67d89();
}

typedef CUresult (*kernel_func_t)(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr scale_ptr, CUdeviceptr shift_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);
kernel_func_t triton_adaptive_layer_norm_fp16_2048_0_1_kernel_kernels[] = {
  triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps2xstages4,
  triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps4xstages4,
  triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps8xstages4,
  triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps16xstages4,
  triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps32xstages4,
};

int triton_adaptive_layer_norm_fp16_2048_0_1_kernel_get_num_algos(void){
  return (int)(sizeof(triton_adaptive_layer_norm_fp16_2048_0_1_kernel_kernels) / sizeof(triton_adaptive_layer_norm_fp16_2048_0_1_kernel_kernels[0]));
}

CUresult triton_adaptive_layer_norm_fp16_2048_0_1_kernel(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr scale_ptr, CUdeviceptr shift_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon, int algo_id){
  assert (algo_id < (int)sizeof(triton_adaptive_layer_norm_fp16_2048_0_1_kernel_kernels));
  return triton_adaptive_layer_norm_fp16_2048_0_1_kernel_kernels[algo_id](stream, x_ptr, y_ptr, weight_ptr, bias_ptr, scale_ptr, shift_ptr, M, N, seq_size, epsilon);
}

void load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel(void){
  load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps2xstages4();
  load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps4xstages4();
  load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps8xstages4();
  load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps16xstages4();
  load_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps32xstages4();
}

void unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel(void){
  unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps2xstages4();
  unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps4xstages4();
  unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps8xstages4();
  unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps16xstages4();
  unload_triton_adaptive_layer_norm_fp16_2048_0_1_kernel_2048x0x1_warps32xstages4();
}


CUresult triton_adaptive_layer_norm_fp16_2048_0_1_kernel_default(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr scale_ptr, CUdeviceptr shift_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon){
  return triton_adaptive_layer_norm_fp16_2048_0_1_kernel(stream, x_ptr, y_ptr, weight_ptr, bias_ptr, scale_ptr, shift_ptr, M, N, seq_size, epsilon, 0);
}

#include <cuda.h>

CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps16xstages4(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);
void load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps16xstages4();
void unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps16xstages4();
    

CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps8xstages4(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);
void load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps8xstages4();
void unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps8xstages4();
    

CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps32xstages4(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);
void load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps32xstages4();
void unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps32xstages4();
    

CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps4xstages4(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);
void load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps4xstages4();
void unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps4xstages4();
    

CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps2xstages4(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);
void load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps2xstages4();
void unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_2048x0x0_warps2xstages4();
    
int triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_get_num_algos(void);

CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel_default(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon);
CUresult triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr mha_out_ptr, CUdeviceptr gate_msa_ptr, CUdeviceptr scale_mlp_ptr, CUdeviceptr shift_mlp_ptr, CUdeviceptr weight_ptr, CUdeviceptr bias_ptr, CUdeviceptr resi_out_ptr, CUdeviceptr adaLN_out_ptr, int32_t M, int32_t N, int32_t seq_size, float epsilon, int algo_id);
void load_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel();
void unload_triton_fused_adaLN_scale_residual_fp16_2048_0_0_kernel();
    
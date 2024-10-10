
# source /root/paddlejob/workspace/env_run/output/changwenbin/miniconda3/bin/activate  /root/paddlejob/workspace/env_run/output/changwenbin/miniconda3/envs/cwb_env_310

# export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/output/changwenbin/Research/TensorRT-10.3.0.26/lib/:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/output/changwenbin/Research/TensorRT-10.3.0.26/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/output/changwenbin/Paddle/paddle/phi/kernels/fusion/cutlass/conv2d/build:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/output/changwenbin/Paddle/paddle/phi/kernels/fusion/cutlass/gemm_epilogue/build:$LD_LIBRARY_PATH
# export GLOG_vmodule=stats=8
# export FLAGS_log_memory_stats=1
# export CUDA_VISIBLE_DEVICES=3
# process=$1
# pid=$(ps x | grep $process | grep -v grep | awk '{print $1}')
# echo $pid
# # PDEXEC $pd_args --shape_file=${RESULT_DIR}/${MODEL_NAME}.pbtxt --warmup=10 --repeats=100 > pd_log.txt 2>&1 & pid=$!
# bash /root/paddlejob/workspace/env_run/output/changwenbin/PaddleMIX/ppdiffusers/examples/inference/nvidia-smi.sh $pid "/root/paddlejob/workspace/env_run/output/changwenbin/PaddleMIX/ppdiffusers/examples/inference/nvidia-smi_mem.log"

# nsys profile -o inference_SD3 \
# python  text_to_image_generation-stable_diffusion_3.py \
python -m paddle.distributed.launch --gpus "2,3" text_to_image_generation-stable_diffusion_3.py \
--dtype float16 \
--height 512 \
--width 512 \
--num-inference-steps 50 \
--inference_optimize 1 \
--inference_optimize_bp 1 \
--benchmark 1


# distributed_qwen2_vl_infer_72B.sh

# 使用 Paddle 分布式启动推理
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" \
    paddlemix/examples/qwen2_5_vl/distributed_qwen2_5_vl_infer.py \
    --model_path Qwen/Qwen2.5-VL-72B-Instruct \
    --mp_degree 8


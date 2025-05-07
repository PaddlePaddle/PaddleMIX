# distributed_ppdocbee2_infer.sh

# 使用 Paddle 分布式启动推理
python -m paddle.distributed.launch --gpus="0,1" \
    paddlemix/examples/ppdocbee2/distributed_ppdocbee2_infer.py \
    --model_path PaddleMIX/PPDocBeeV2-3B \
    --mp_degree 2

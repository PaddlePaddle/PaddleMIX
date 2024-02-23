
export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export NVIDIA_TF32_OVERRIDE=0
export NCCL_ALGO=Tree

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m paddle.distributed.launch \
    --nnodes=1 --nproc_per_node=8 --use_env \
    train.py \
    --model DiT-XL/2 \
    --feature-path ./data/fastdit_imagenet256 \
    --global-batch-size 16

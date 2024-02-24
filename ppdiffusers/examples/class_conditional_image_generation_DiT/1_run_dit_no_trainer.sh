export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export NVIDIA_TF32_OVERRIDE=0
export NCCL_ALGO=Tree

#CUDA_VISIBLE_DEVICES=0 python -m paddle.distributed.launch --nnodes=1 --nproc_per_node=1 --use_env \
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" \
    train_image_generation_dit_notrainer.py \
    --dit_config_file config/DiT_XL_patch2.json \
    --feature_path ./data/fastdit_imagenet256 \
    --global_batch_size 16

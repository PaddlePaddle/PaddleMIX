python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" \
    train_image_generation_notrainer.py \
    --config_file config/DiT_XL_patch2.json \
    --feature_path ./data/fastdit_imagenet256 \
    --global_batch_size 256

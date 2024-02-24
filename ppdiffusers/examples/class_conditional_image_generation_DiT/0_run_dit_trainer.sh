export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export NVIDIA_TF32_OVERRIDE=0
export NCCL_ALGO=Tree

#python -u train_image_generation_dit_trainer.py \
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_image_generation_dit_trainer.py \
    --do_train \
    --feature_path data/fastdit_imagenet256 \
    --output_dir ./output_trainer \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --max_steps 7000000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 1000 \
    --logging_steps 20 \
    --save_steps 10000 \
    --save_total_limit 50 \
    --seed 23 \
    --dataloader_num_workers 1 \
    --vae_name_or_path stabilityai/sd-vae-ft-mse \
    --dit_config_file config/DiT_XL_patch2.json \
    --num_inference_steps 25 \
    --use_ema True \
    --recompute True \
    --max_grad_norm -1 \
    --overwrite_output_dir True

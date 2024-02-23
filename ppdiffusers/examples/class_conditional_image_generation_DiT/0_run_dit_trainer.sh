
export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export NVIDIA_TF32_OVERRIDE=0
export NCCL_ALGO=Tree

python -u train_txt2img_dit_trainer.py \
    --do_train \
    --output_dir ./dit_output_trainer \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --max_steps 1000000000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 1000 \
    --logging_steps 50 \
    --save_steps 5000 \
    --save_total_limit 50 \
    --seed 23 \
    --dataloader_num_workers 1 \
    --vae_name_or_path stabilityai/sd-vae-ft-mse \
    --dit_config_file config/dit_s_4.json \
    --num_inference_steps 50 \
    --model_max_length 77 \
    --max_grad_norm -1


ppdiffusers_lvdm_path=/root/project/paddlenlp/lvdm/paddle/PaddleMIX/ppdiffusers
export PYTHONPATH=$ppdiffusers_lvdm_path:$PYTHONPATH
set -eux

# export CUDA_VISIBLE_DEVICES=1
# export NVIDIA_TF32_OVERRIDE=0

# ppdiffusers train
cd $ppdiffusers_lvdm_path/examples/text_to_video_lvdm
python -u train_lvdm_short.py \
    --do_train \
    --do_eval \
    --label_names pixel_values \
    --eval_steps 5 \
    --vae_type 3d \
    --output_dir temp/checkpoints_short \
    --unet_config_file unet_configs/lvdm_short_sky_no_ema/unet/config.json \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 6e-5 \
    --max_steps 1000000000 \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --image_logging_steps 10 \
    --logging_steps 1 \
    --save_steps 5000 \
    --seed 23 \
    --dataloader_num_workers 0 \
    --weight_decay 0.01 \
    --max_grad_norm 0 \
    --overwrite_output_dir False \
    --pretrained_model_name_or_path westfish/lvdm_short_sky_no_ema
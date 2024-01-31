# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -uex

# test dreambooth
cd ../../examples/dreambooth

export FLAGS_conv_workspace_size_limit=4096
export INSTANCE_DIR="./dogs"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="dreambooth_danka"
export FLAG_FUSED_LINEAR=0
export FLAG_XFORMERS_ATTENTION_OP=auto

python -u train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_checkpointing \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50 \
  --noise_offset 1 \
  --enable_xformers_memory_efficient_attention 
  
  
# export OUTPUT_DIR="dreambooth_duoka"
python -u -m paddle.distributed.launch --gpus "0,1" train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_checkpointing \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50 \
  --noise_offset 1 \
  --enable_xformers_memory_efficient_attention 


# export OUTPUT_DIR="dreambooth_lora_danka"
python train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=50 \
  --learning_rate=1e-4 \
  --report_to="visualdl" \
  --gradient_checkpointing \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=5 \
  --lora_rank=4 \
  --checkpointing_steps 50 \
  --seed=0 \
  --noise_offset=1 \
  --train_text_encoder \
  --enable_xformers_memory_efficient_attention 

  
# export OUTPUT_DIR="dreambooth_lora_duoka"
python -u -m paddle.distributed.launch --gpus "0,1" train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --gradient_checkpointing \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=50 \
  --learning_rate=1e-4 \
  --report_to="visualdl" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=5 \
  --checkpointing_steps 50 \
  --lora_rank=4 \
  --seed=0 \
  --noise_offset=1 \
  --train_text_encoder \
  --enable_xformers_memory_efficient_attention 

export OUTPUT_DIR="dreambooth_lora_sdxl_danka"
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

python -u -m paddle.distributed.launch --gpus "0" train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --height=512 \
  --width=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --report_to="visualdl" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" \
  --checkpointing_steps=100 \
  --enable_xformers_memory_efficient_attention

cd -

# test text_to_image
cd ../../examples/text_to_image
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"
export OUTPUT_DIR="sd-pokemon-model"

python -u train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --debug \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --max_train_steps=50 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --noise_offset=1 \
  --output_dir=${OUTPUT_DIR}


export OUTPUT_DIR="sd-pokemon-model-duoka"
python -u -m paddle.distributed.launch --gpus "0,1" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --debug \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --max_train_steps=50 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --noise_offset=1 \
  --output_dir=${OUTPUT_DIR}
  

export OUTPUT_DIR="sd-pokemon-model-lora"
python train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=4 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=50 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --debug \
  --gradient_checkpointing \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=visualdl \
  --checkpointing_steps=50 \
  --validation_prompt="Totoro" \
  --lora_rank=4 \
  --seed=1337 \
  --noise_offset=1 \
  --validation_epochs 1 \
  --enable_xformers_memory_efficient_attention

export OUTPUT_DIR="sd-pokemon-model-lora-duoka"
python -u -m paddle.distributed.launch --gpus "0,1" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=4 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=50 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --debug \
  --gradient_checkpointing \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=visualdl \
  --checkpointing_steps=50 \
  --validation_prompt="Totoro" \
  --lora_rank=4 \
  --seed=1337 \
  --noise_offset=1 \
  --validation_epochs 1 \
  --enable_xformers_memory_efficient_attention
  
cd -

# test textual_inversion
cd ../../examples/textual_inversion
export DATA_DIR="cat-toy"


export OUTPUT_DIR="textual_inversion_cat"
python -u train_textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=100 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed 42 \
  --save_steps 50 \
  --gradient_checkpointing \
  --validation_prompt "A <cat-toy> backpack" \
  --validation_epochs 1 \
  --noise_offset 1 \
  --output_dir=${OUTPUT_DIR}


export OUTPUT_DIR="textual_inversion_cat_duoka"
python -u -m paddle.distributed.launch --gpus "0,1" train_textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=100 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed 42 \
  --save_steps 50 \
  --gradient_checkpointing \
  --validation_prompt "A <cat-toy> backpack" \
  --validation_epochs 1 \
  --noise_offset 1 \
  --output_dir=${OUTPUT_DIR} 
  
  
cd -

# test text_to_image_laion400m
cd ../../examples/text_to_image_laion400m
python -u train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ./laion400m_pretrain_output_trainer \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.02 \
    --max_steps 100 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 20 \
    --logging_steps 10 \
    --save_steps 50 \
    --save_total_limit 5 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --vae_name_or_path CompVis/stable-diffusion-v1-4/vae \
    --text_encoder_config_file config/ldmbert.json \
    --unet_config_file config/unet.json \
    --file_list ./data/filelist/train.filelist.list \
    --num_inference_steps 10 \
    --model_max_length 77 \
    --tokenizer_name bert-base-uncased \
    --max_grad_norm -1 \
    --recompute True \
    --overwrite_output_dir \
    --benchmark True


python -u -m paddle.distributed.launch --gpus "0,1" train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ./laion400m_pretrain_output_trainer \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.02 \
    --max_steps 100 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 20 \
    --logging_steps 10 \
    --save_steps 50 \
    --save_total_limit 5 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --vae_name_or_path CompVis/stable-diffusion-v1-4/vae \
    --text_encoder_config_file config/ldmbert.json \
    --unet_config_file config/unet.json \
    --file_list ./data/filelist/train.filelist.list \
    --num_inference_steps 10 \
    --model_max_length 77 \
    --tokenizer_name bert-base-uncased \
    --max_grad_norm -1 \
    --recompute True \
    --overwrite_output_dir \
    --benchmark True


cd -

# test stable_diffusion
cd ../../examples/stable_diffusion
# CUDA version needs to be greater than 11.7.
python -u train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ./laion400m_pretrain_output_trainer \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.02 \
    --max_steps 100 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 20 \
    --logging_steps 10 \
    --save_steps 50 \
    --save_total_limit 5 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --file_list ./data/filelist/train.filelist.list \
    --num_inference_steps 10 \
    --model_max_length 77 \
    --max_grad_norm -1 \
    --recompute True \
    --overwrite_output_dir \
    --benchmark True \
    --fp16_opt_level O2

# CUDA version needs to be greater than 11.7.
python -u -m paddle.distributed.launch --gpus "0,1" train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ./laion400m_pretrain_output_trainer \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.02 \
    --max_steps 100 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 20 \
    --logging_steps 10 \
    --save_steps 50 \
    --save_total_limit 5 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --file_list ./data/filelist/train.filelist.list \
    --num_inference_steps 10 \
    --model_max_length 77 \
    --max_grad_norm -1 \
    --recompute True \
    --overwrite_output_dir \
    --benchmark True \
    --fp16_opt_level O2

cd -


# test autoencoder
cd ../../examples/autoencoder/vae

python train_vae.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --ignore_keys decoder. \
    --vae_config_file config/vae.json \
    --freeze_encoder \
    --enable_xformers_memory_efficient_attention \
    --input_size 256 256 \
    --max_train_steps 100 \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --num_workers 4 \
    --logging_steps 25 \
    --save_steps 4000 \
    --image_logging_steps 25 \
    --disc_start 10 \
    --kl_weight 0.000001 \
    --disc_weight 0.5 \
    --resolution 512


python -u -m paddle.distributed.launch --gpus "0,1" train_vae.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --ignore_keys decoder. \
    --vae_config_file config/vae.json \
    --freeze_encoder \
    --enable_xformers_memory_efficient_attention \
    --input_size 256 256 \
    --max_train_steps 100 \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --num_workers 4 \
    --logging_steps 25 \
    --save_steps 4000 \
    --image_logging_steps 25 \
    --disc_start 10 \
    --kl_weight 0.000001 \
    --disc_weight 0.5 \
    --resolution 512


cd -


# test controlnet
cd ../../examples/controlnet


python -u train_txt2img_control_trainer.py \
    --do_train \
    --output_dir ./sd15_control_danka \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.02 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --sd_locked True \
    --max_steps 100 \
    --logging_steps 50 \
    --image_logging_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --max_grad_norm -1 \
    --file_path ./fill50k \
    --recompute True \
    --overwrite_output_dir


python -u -m paddle.distributed.launch --gpus "0,1" train_txt2img_control_trainer.py \
    --do_train \
    --output_dir ./sd15_control_duoka \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.02 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --sd_locked True \
    --max_steps 100 \
    --logging_steps 50 \
    --image_logging_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --max_grad_norm -1 \
    --file_path ./fill50k \
    --recompute True \
    --overwrite_output_dir

cd -


# test t2i-adapter
cd ../../examples/t2i-adapter

python -u train_t2i_adapter_trainer.py \
    --do_train \
    --output_dir ./sd15_openpose_danka \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.02 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --max_steps 100 \
    --logging_steps 1 \
    --image_logging_steps 50 \
    --save_steps 50 \
    --save_total_limit 1 \
    --seed 4096 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --max_grad_norm -1 \
    --file_list ./data_demo/train.openpose.filelist \
    --recompute True --use_ema False \
    --control_type raw \
    --data_format img2img \
    --use_paddle_conv_init False \
    --overwrite_output_dir \
    --timestep_sample_schedule cubic


python -u -m paddle.distributed.launch --gpus "0,1" train_t2i_adapter_trainer.py \
    --do_train \
    --output_dir ./sd15_openpose_duoka \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.02 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --max_steps 100 \
    --logging_steps 1 \
    --image_logging_steps 50 \
    --save_steps 50 \
    --save_total_limit 1 \
    --seed 4096 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --max_grad_norm -1 \
    --file_list ./data_demo/train.openpose.filelist \
    --recompute True --use_ema False \
    --control_type raw \
    --data_format img2img \
    --use_paddle_conv_init False \
    --overwrite_output_dir \
    --timestep_sample_schedule cubic
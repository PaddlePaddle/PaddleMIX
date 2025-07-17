# DMD2 模型蒸馏
DMD（Distribution Matching Distillation）是一种将昂贵的扩散模型推理过程蒸馏成单步生成器的一种技术。DMD2在DMD的基础上提供了一系列的技巧，简化了训练流程，提升了效果。

## 快速开始

### 安装DMD2需要的依赖
```shell
pip install -r requirements.txt
```
### 推理示例

```shell
python -m edm.imagenet_example  --checkpoint_path YOUR_TRAINED_MODEL_PATH
```

我们提供了一个预训练好的[模型](https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/dmd2/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch.pdparams)

### 训练示例


* 硬件要求：Nvidia A100 80G，如果显存不足可以对应较少batch size

#### 数据准备
```
wget https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz -O $CHECKPOINT_PATH/imagenet_fid_refs_edm.npz

###### download the imagenet-64x64 lmdb
wget https://huggingface.co/tianweiy/DMD2/resolve/main/data/imagenet/imagenet-64x64_lmdb.zip?download=true -O $CHECKPOINT_PATH/imagenet-64x64_lmdb.zip
unzip $CHECKPOINT_PATH/imagenet-64x64_lmdb.zip -d $CHECKPOINT_PATH

###### 下载edm模型的预训练权重
wget https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/dmd2/edm-imagenet-64x64-cond-adm.pdparams
```

#### 安装算子
```
cd ops
python setup.py install
cd ..
```

#### 训练

```bash
#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m paddle.distributed.launch edm/train_edm.py \
    --generator_lr 2e-6 \
    --guidance_lr 2e-6 \
    --train_iters 200000 \
    --output_path output/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch \
    --batch_size 24 \
    --initialie_generator \
    --log_iters 500 \
    --resolution 64 \
    --label_dim 1000 \
    --dataset_name "imagenet" \
    --seed 1 \
    --model_id datas/edm-imagenet-64x64-cond-adm.pdparams \
    --wandb_iters 100 \
    --wandb_entity YOUR_ENTITY \
    --wandb_project dmd2_imagenet \
    --wandb_name "imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch" \
    --real_image_path datas/imagenet-64x64_lmdb \
    --dfake_gen_update_ratio 5 \
    --cls_loss_weight 1e-2 \
    --gan_classifier \
    --gen_cls_loss_weight 3e-3 \
    --diffusion_gan \
    --diffusion_gan_max_timestep 1000 \
    --delete_ckpts \
    --max_checkpoint 500 \
    --use_fp16

```

#### 评估

训练完后，可用以下脚本进行评估，获得模型的fid.

```bash
python -u edm/test_folder_edm.py \
    --folder output/imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch/time_1741760210_seed1/ \
    --wandb_name test_imagenet_gan_classifier_genloss3e-3_diffusion1000_lr2e-6_scratch \
    --wandb_entity jll-none \
    --wandb_project dmd2 \
    --resolution 64 \
    --label_dim 1000 \
    --ref_path datas/imagenet_fid_refs_edm.npz
```

### SDXL-Lora 训练

#### 数据准备
```
# training prompts
wget  https://huggingface.co/tianweiy/DMD2/resolve/main/data/laion/captions_laion_score6.25.pkl?download=true -O $CHECKPOINT_PATH/captions_laion_score6.25.pkl

# evaluation prompts
wget  https://huggingface.co/tianweiy/DMD2/resolve/main/data/coco/captions_coco14_test.pkl?download=true -O $CHECKPOINT_PATH/captions_coco14_test.pkl


mkdir $CHECKPOINT_PATH/sdxl_vae_latents_laion_500k
# real dataset
for INDEX in {0..59}
do
    # Format the index to be zero-padded to three digits
    INDEX_PADDED=$(printf "%03d" $INDEX)

    # Download the file
    wget "https://huggingface.co/tianweiy/DMD2/resolve/main/data/laion_vae_latents/sdxl_vae_latents_laion_500k/vae_latents_${INDEX_PADDED}.pt?download=true" -O "${CHECKPOINT_PATH}/sdxl_vae_latents_laion_500k/vae_latents_${INDEX_PADDED}.pt"
done

# generate the lmdb database from the downloaded files
python main/data/create_lmdb_iterative.py   --data_path $CHECKPOINT_PATH/sdxl_vae_latents_laion_500k/  --lmdb_path $CHECKPOINT_PATH/sdxl_vae_latents_laion_500k_lmdb

# evaluation images
wget https://huggingface.co/tianweiy/DMD2/resolve/main/data/coco/coco10k.zip?download=true -O $CHECKPOINT_PATH/coco10k.zip
unzip $CHECKPOINT_PATH/coco10k.zip -d $CHECKPOINT_PATH
```

#### 训练命令
```bash
USE_PEFT_BACKEND=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m paddle.distributed.launch train_sd.py \
    --generator_lr 5e-5 \
    --guidance_lr 5e-5 \
    --train_iters 200000 \
    --output_path  output/sdxl_cond999_8node_lr5e-7_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch \
    --batch_size 1 \
    --grid_size 1 \
    --initialie_generator \
    --log_iters 1000 \
    --resolution 1024 \
    --latent_resolution 128 \
    --seed 10 \
    --real_guidance_scale 8 \
    --fake_guidance_scale 1.0 \
    --max_grad_norm 10.0 \
    --model_id "stabilityai/stable-diffusion-xl-base-1.0" \
    --wandb_iters 100 \
    --wandb_entity dmd2 \
    --wandb_project sdxl \
    --wandb_name "sdxl_cond999_8node_lr5e-7_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch" \
    --dfake_gen_update_ratio 5 \
    --sdxl \
    --gsp \
    --max_step_percent 0.98 \
    --cls_on_clean_image \
    --gen_cls_loss \
    --gen_cls_loss_weight 5e-3 \
    --guidance_cls_loss_weight 1e-2 \
    --diffusion_gan \
    --diffusion_gan_max_timestep 1000 \
    --denoising \
    --num_denoising_step 4 \
    --denoising_timestep 1000 \
    --backward_simulation \
    --train_prompt_path ckpts/captions_laion_score6.25.pkl \
    --real_image_path ckpts/sdxl_vae_latents_laion_500k_lmdb \
    --generator_lora
```

#### 评估

训练完后，可用以下脚本进行评估，获得模型的fid.

```bash
export PYTHONPATH=./:$PWD/../../scripts/fid_clip_score/:$PYTHONPATH USE_PEFT_BACKEND=1
python -u sdxl/test_sdxl_single_ckpt.py  \
    --checkpoint_path YOUR-TRAINED-WEIGHT \
    --conditioning_timestep 999 \
    --num_step 4 \
    --wandb_entity YOUR-ENTITY \
    --wandb_project dmd2 \
    --num_train_timesteps 1000 \
    --seed 10 \
    --eval_res 512 \
    --ref_dir ckpts/coco10k/subset \
    --anno_path  ckpts/coco10k/all_prompts.pkl \
    --total_eval_samples 10000 \
    --wandb_name YOUR_WANDB_NAME \
    --generator_lora
```

这里提供了一个预训练好的[模型](https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/dmd2/sdxl_cond999_8node_lr5e-5_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch_lora.pdparams)

也可以将生成的文件保存成一个文件夹的形式，使用[该目录下的脚本](../../../../scripts/fid_clip_score/)获得fid.

## 参考
- https://github.com/tianweiy/DMD2

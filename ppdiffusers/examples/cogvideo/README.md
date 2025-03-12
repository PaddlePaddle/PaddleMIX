# CogVideoX视频生成

CogVideoX是的开源视频生成模型，支持文本到视频（Text-to-Video）和图像到视频（Image-to-Video）两种模式。本仓库提供了CogVideoX的paddle实现，支持推理和lora训练


## 快速开始

### 推理示例

```shell
python scripts/infer.py \
  --prompt "a bear is walking in a zoon" \
  --model_path THUDM/CogVideoX-2b \
  --generate_type "t2v" \
  --dtype "float16" \
  --seed 42
```

model_path 当前支持: THUDM/CogVideoX-2b、THUDM/CogVideoX-5b


### 训练示例


* 硬件要求：Nvidia A100 80G
* 目前，LoRA微调仅针对 [CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b).

#### 数据准备


提供两个txt文件，一个是包含文本提示的`prompts.txt`文件，另一个是包含视频路径的`videos.txt` 文件（视频文件应该在给定的`-instance_data_root`根目录下）。

例如,
`--instance_data_root`=`/dataset`, 则 `/dataset`下应包含 `prompts.txt` and `videos.txt`.

`prompts.txt` 里的prompt 以行分隔:

```
A black and white animated sequence featuring a rabbit, named Rabbity Ribfried, and an anthropomorphic goat in a musical, playful environment, showcasing their evolving interaction.
A black and white animated sequence on a ship's deck features a bulldog character, named Bully Bulldoger, showcasing exaggerated facial expressions and body language. The character progresses from confident to focused, then to strained and distressed, displaying a range of emotions as it navigates challenges. The ship's interior remains static in the background, with minimalistic details such as a bell and open door. The character's dynamic movements and changing expressions drive the narrative, with no camera movement to distract from its evolving reactions and physical gestures.
...
```

`videos.txt` 文件应包含视频文件以行分隔，并且以`prompts.txt`里一一对应.

```
videos/00000.mp4
videos/00001.mp4
...
```

目录结构如下:

```
/dataset
├── prompts.txt
├── videos.txt
├── videos
    ├── videos/00000.mp4
    ├── videos/00001.mp4
    ├── ...
```

启动命令时，`--caption_column` 指定为 `prompts.txt`； `--video_column` 指定为 `videos.txt`.

本仓库提供示例数据集
```bash
wget https://bj.bcebos.com/v1/dataset/PaddleMIX/davis_validation_for_cogvideox.tar
tar -xvf davis_validation_for_cogvideox.tar
```

#### Lora微调

```bash
#!/bin/bash

export USE_PEFT_BACKEND=True

python examples/cogvideo/scripts/train_cogvideox_lora.py \
  --pretrained_model_name_or_path THUDM/CogVideoX-2b \
  --instance_data_root <PATH_TO_WHERE_VIDEO_FILES_ARE_STORED> \
  --caption_column prompts.txt \
  --video_column videos.txt \
  --id_token DISNEY \
  --validation_prompt "a bear is walking in a zoon" \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_epochs 1 \
  --seed 42 \
  --rank 64 \
  --lora_alpha 64 \
  --mixed_precision bf16 \
  --fp16_opt_level O2 \
  --output_dir ./cogvideox-lora \
  --height 480 --width 720 --fps 8 --max_num_frames 49 --skip_frames_start 0 --skip_frames_end 0 \
  --train_batch_size 1 \
  --num_train_epochs 10 \
  --checkpointing_steps 1000 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-3 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --optimizer Adam \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --report_to wandb

```

**参数说明**
```
pretrained_model_name_or_path：预训练模型路径
instance_data_root：数据集路径
caption_column：文本提示文件
video_column：视频文件
id_token：自定义的标识符，非必要，保持默认即可
validation_prompt：验证提示词
validation_prompt_separator：验证提示词的分隔符
num_validation_videos：验证视频数量
validation_epochs：验证epoch周期
seed：随机种子
rank：LoRA秩
lora_alpha：LoRA缩放因子
mixed_precision：混合精度, 可选bf16, fp16，no; no表示不使用混合精度,用float32
fp16_opt_level：fp16优化级别，可选O0, O1, O2; O0表示不使用混合精度,用float32
output_dir：输出目录
height：视频高度
width：视频宽度
fps：视频帧率
max_num_frames：视频最大帧数,当前最大支持49帧
skip_frames_start：起始帧跳过数
skip_frames_end：结束帧跳过数
train_batch_size：训练批次大小
num_train_epochs：训练周期
checkpointing_steps：检查点保存间隔
gradient_accumulation_steps：梯度累积步数
learning_rate：学习率,paddle默认会*0.01
lr_scheduler：学习率调度器
lr_warmup_steps：学习率预热步数
lr_num_cycles：学习率周期数
enable_slicing：vae是否启用切片
enable_tiling：vae是否启用平铺
optimizer：优化器
adam_beta1：Adam优化器β1
adam_beta2：Adam优化器β2
max_grad_norm：最大梯度范数
report_to：报告工具，可选wandb
```

## Lora推理

Lora训练完后，可用以下脚本进行推理.

```bash
python scripts/lora_infer.py \
      --model_path THUDM/CogVideoX-2b \
      --prompt "a bear is walking in a zoon" \
      --lora_path path-lora \
      --output_path output.mp4
```

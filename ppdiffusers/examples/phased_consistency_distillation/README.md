# Phased Consistency Models

## 1. 模型简介

[Phased Consistency Models](https://arxiv.org/abs/2405.18407) 能够在极少步数（1～4step）下生成高分辨率的高质量图像。

<p align="center">
  <img src="assets/teaser.png" align="middle" width = "600" />
</p>

## 2. 数据准备

PCM的LoRA训练需要使用`CC3M`数据集或者`自定义数据集`，[CC3M数据集下载地址](https://huggingface.co/datasets/pixparse/cc3m-wds)

## 3. PCM-LoRA训练

### 3.1 硬件要求

示例脚本配置在显存 >=50GB 的显卡上可正常训练，如显存不满足要求，可通过修改参数的方式运行脚本：
- 修改`--train_batch_size`减少batch size
- 修改`--resolution` 降低分辨率

### 3.2 训练脚本

```bash
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_pcm_lora_sd3_adv.py \
    --data_path "./cc3m" \
    --pretrained_teacher_model stabilityai/stable-diffusion-3-medium-diffusers \
    --output_dir "outputs/lora_64_fuyun_PCM_sd3_202503191011" \
    --tracker_project_nam "lora_64_formal_fuyun_PCM_sd3_202503191011" \
    --mixed_precision "fp16" \
    --resolution "1024" \
    --lora_rank "32" \
    --learning_rate "5e-6" \
    --loss_type "huber" \
    --adam_weight_decay "1e-3" \
    --max_train_steps "20000" \
    --dataloader_num_workers "16" \
    --w_min "4" \
    --w_max "5" \
    --validation_steps "1000" \
    --checkpointing_steps "2000" \
    --checkpoints_total_limit "10" \
    --train_batch_size "2" \
    --gradient_accumulation_steps "1" \
    --resume_from_checkpoint "latest" \
    --seed "453645634" \
    --num_euler_timesteps "100" \
    --multiphase "4" \
    --adv_weight "0.1" \
    --adv_lr "1e-5" \
    --report_to wandb \
```
参数说明
* `--data_path`: 训练数据集路径
* `--pretrained_teacher_model`: 预训练的Stable Diffusion模型路径
* `--output_dir`：输出文件夹路径
* `--tracker_project_nam`：wandb项目名称
* `--mixed_precision`：是否开启混合精度训练
* `--resolution`：训练分辨率
* `--lora_rank`：lora rank
* `--learning_rate`：学习率
* `--loss_type`：损失类型
* `--adam_weight_decay`：adam优化器的权重衰减
* `--max_train_steps`：训练step数量
* `--dataloader_num_workers`：读取数据的线程数
* `--w_min`：guidance scale最小值
* `--w_max`：guidance scale最大值
* `--validation_steps`：验证间隔step数
* `--checkpointing_steps`：保存checkpoint间隔step数
* `--checkpoints_total_limit`：保存的checkpoint数量上限
* `--train_batch_size`：训练batch size
* `--gradient_accumulation_steps`：梯度累积步数
* `--resume_from_checkpoint`: 恢复训练的checkpoint路径
* `--seed`: 随机种子
* `--num_euler_timesteps`：euler solver的timestep数量
* `--multiphase`: 多阶段的数量
* `--adv_weight`：adversarial loss的权重
* `--adv_lr`：Discriminator的学习率
* `--report_to`：报告工具，支持wandb

## 4. 模型推理

下载模型权重

| params | download  |
|------------|:-------:|
| `pcm_deterministic_2step_shift1.pdparams` | [下载地址](https://paddlenlp.bj.bcebos.com/models/community/pcm_paddle/pcm_deterministic_2step_shift1.pdparams) |
| `pcm_deterministic_4step_shift1.pdparams` | [下载地址](https://paddlenlp.bj.bcebos.com/models/community/pcm_paddle/pcm_deterministic_4step_shift1.pdparams) |
| `pcm_deterministic_4step_shift3.pdparams` | [下载地址](https://paddlenlp.bj.bcebos.com/models/community/pcm_paddle/pcm_deterministic_4step_shift3.pdparams) |

```python
import os
os.environ["USE_PEFT_BACKEND"] = "True"
import paddle
import numpy as np
from PIL import Image
from pcm_fm_deterministic_scheduler import PCMFMDeterministicScheduler
from ppdiffusers import StableDiffusion3Pipeline

path_to_lora = "./pcm_deterministic_4step_shift3.pdparams"
step = 4
shift = 3
num_pcm_timesteps = 50

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    scheduler=PCMFMDeterministicScheduler(1000, shift, num_pcm_timesteps),
    map_location="cpu",
    paddle_dtype=paddle.float16
)
pipe.load_lora_weights(path_to_lora)
prompt = "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography"

with paddle.no_grad():
    result_image = pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=step,
        guidance_scale=1.2,
        generator=paddle.Generator().manual_seed(42),
        joint_attention_kwargs={"scale": 0.25}  # for lora scaling
    ).images[0]
result_image.save(prompt[:5] + prompt[-5:] + ".png")

```
生成的图片如下所示：
<center><img src="example-1.png" width=100%></center>
<center><img src="example-2.png" width=100%></center>


## 5. 参考资料
- https://github.com/G-U-N/Phased-Consistency-Model
# Latent Consistency Models

## 1. 模型简介

[Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/pdf/2310.04378.pdf) 是清华大学交叉信息科学研究院研发的一款生成模型。它的特点是可以通过少量步骤推理合成高分辨率图像，使图像生成速度提升 2-5 倍，需要的算力也更少。官方称 LCMs 是继 LDMs（Latent Diffusion Models 潜在扩散模型）之后的新一代生成模型。

<p align="center">
  <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/50394665/5bfed3e3-8c00-4188-8b54-7e97bae1a908" align="middle" width = "600" />
</p>

注：该图引自[LCM Project Page](https://latent-consistency-models.github.io/)。


### Latent Consistency Models Model zoo

<div align="center">

| model name | params | weight |
|------------|:-------:|:------:|
| `latent-consistency/lcm-sdxl` | TODO |TODO |
| `latent-consistency/lcm-lora-sdv1-5` | TODO |TODO |
| `latent-consistency/lcm-lora-sdxl` | TODO |TODO |

</div>

- 后续将陆续支持更多的 LCM 模型。
- 模型下载地址：TODO，后续将提供 AI Studio 上预训练模型下载地址。



## 2. 环境准备
通过 `git clone` 命令拉取 PaddleMIX 源码，并安装必要的依赖库。请确保你的 PaddlePaddle 框架版本在 2.5.2 之后，PaddlePaddle 框架安装可参考 [飞桨官网-安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

```bash
# 克隆 PaddleMIX 仓库
git clone https://github.com/PaddlePaddle/PaddleMIX

# 安装2.5.2版本的paddlepaddle-gpu，当前我们选择了cuda11.7的版本，可以查看 https://www.paddlepaddle.org.cn/ 寻找自己适合的版本
python -m pip install paddlepaddle-gpu==2.5.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# 进入consistency_distillation目录
cd PaddleMIX/ppdiffusers/examples/consistency_distillation/lcm_trainer

# 安装所需的依赖, 如果提示权限不够，请在最后增加 --user 选项
pip install -r requirements.txt
```

> 注：本模型训练与推理需要依赖 CUDA 11.2 及以上版本，如果本地机器不符合要求，建议前往 [AI Studio](https://aistudio.baidu.com/index) 进行模型训练、推理任务。

## 3. 数据准备

LCM蒸馏训练或LCM-LoRA训练需要使用 `Laion400M数据集`或者`自定义数据集`，请先按照[这里的数据准备](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/stable_diffusion#3-%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87)准备数据。

## 4. LCM-LoRA训练

> Tips: 非LoRA训练仅仅需要设置`--is_lora=False`, 当前非lora的训练还处于精度验证阶段，请优先使用`LCM-LoRA`方式进行训练。

### 4.1 硬件要求

示例脚本配置在显存 >=32GB 的显卡上可正常训练，如显存不满足要求，可通过修改参数的方式运行脚本：
- 如果本地环境显存不够，请使用 AIStudio 上 32G 显存的 GPU 环境，并修改 `--per_device_train_batch_size` 为 12.
- bf16 混合精度训练模式支持 A100、3090、3080 等硬件，不支持使用 V100 进行训练，如果你的硬件满足要求，修改 `--bf16` 为 `True` 可启动混合精度训练模式，体验更快速的训练。

### 4.2 单机单卡训练

> 注意，我们当前训练的分辨率是 `512x512` ，如果需要训练 `1024x1024` 分辨率，请修改 `--resolution` 为 1024 并且降低`--per_device_train_batch_size` 参数，否则会报显存不足的错误。

单机单卡训练启动脚本如下，建议保存为 `train.sh` 后执行命令 `bash train.sh`：

```bash
export FLAGS_conv_workspace_size_limit=4096

# 2.6.0的时候会有很多类型提升的warning，GLOG_minloglevel=2将会关闭这些warning
export GLOG_minloglevel=2
export OUTPUT_DIR="lcm_lora_outputs"
export BATCH_SIZE=12
export MAX_ITER=10000

# 如果使用自定义数据
FILE_LIST=./processed_data/filelist/custom_dataset.filelist.list
# 如果使用laion400m_demo数据集，需要把下面的注释取消
# FILE_LIST=./data/filelist/train.filelist.list

# 如果使用sd15，
# BF16 O2 需要16G显存
MODEL_NAME_OR_PATH="runwayml/stable-diffusion-v1-5"
IS_SDXL=False
RESOLUTION=512

# 如果使用sdxl
# BF16 O2 需要46G显存
# MODEL_NAME_OR_PATH="stabilityai/stable-diffusion-xl-base-1.0"
# IS_SDXL=True
# RESOLUTION=1024

python train_lcm.py \
    --do_train \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --max_steps ${MAX_ITER} \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 100 \
    --logging_steps 10 \
    --resolution ${RESOLUTION} \
    --save_steps 2000 \
    --save_total_limit 20 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path ${MODEL_NAME_OR_PATH} \
    --file_list ${FILE_LIST} \
    --model_max_length 77 \
    --max_grad_norm 1 \
    --disable_tqdm True \
    --overwrite_output_dir \
    --recompute True \
    --loss_type "huber" \
    --lora_rank 64 \
    --is_sdxl ${IS_SDXL} \
    --is_lora True
```

[train_lcm.py](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/examples/consistency_distillation/lcm_trainer/train_lcm.py) 可传入的参数解释如下：
* `--pretrained_model_name_or_path`: 加载预训练模型的名称或本地路径，如 `runwayml/stable-diffusion-v1-5`或`stabilityai/stable-diffusion-xl-base-1.0`。
* `--per_device_train_batch_size`: 训练时每张显卡所使用的 `batch_size批量`，当我们的显存较小的时候，需要将这个值设置的小一点。
* `--gradient_accumulation_steps`: 梯度累积的步数，用户可以指定梯度累积的步数，在梯度累积的 step 中。减少多卡之间梯度的通信，减少更新的次数，扩大训练的 batch_size。
* `--learning_rate`: 学习率, 这里我们使用了`1e-6`，推荐训练lora时采用更大的学习率`1e-4`。
* `--weight_decay`: AdamW 优化器的 `weight_decay`，默认值为`1e-2`。
* `--max_steps`: 最大的训练步数。
* `--save_steps`: 每间隔多少步 `（global step步数）`，保存模型。
* `--save_total_limit`: 最多保存多少个模型。
* `--lr_scheduler_type`: 要使用的学习率调度策略。默认为 `constant`。
* `--warmup_steps`: 用于从 0 到 `learning_rate` 的线性 warmup 的步数。
* `--resolution`: 预训练阶段将训练的图像的分辨率，默认为 `512`。
* `--vae_encode_batch_size`: vae在编码图片时使用的batch_size，sd15-512x512分辨率的情况下默认为`32`、sdxl-1024x1024分辨率的情况下默认为`8`，如果值太大会爆出`cublas error`的错误。
* `--image_logging_steps`: 每隔多少步，log 训练过程中的图片，默认为 `1000` 步，注意 `image_logging_steps` 需要是 `logging_steps` 的整数倍。
* `--logging_steps`: logging 日志的步数，默认为 `500` 步。
* `--output_dir`: 模型保存路径。
* `--seed`: 随机种子，为了可以复现训练结果，Tips：当前 paddle 设置该随机种子后仍无法完美复现。
* `--dataloader_num_workers`: Dataloader 所使用的 `num_workers` 参数，请确保处理后的`part文件`数量要大于等于`dataloader_num_workers` * `num_gpus`，否则程序会卡住，例如：`dataloader_num_workers=4`、`num_gpus=2`时候，请确保切分后的`part文件`数量要大于等于`8`。
* `--file_list`: file_list 文件地址。
* `--num_inference_steps`: 推理预测时候使用的步数，默认值为`4`。
* `--model_max_length`: `tokenizer` 中的 `model_max_length` 参数，超过该长度将会被截断。
* `--max_grad_norm`: 梯度剪裁的最大 norm 值，`-1` 表示不使用梯度裁剪策略。
* `--recompute`: 是否开启重计算，(`bool`，可选，默认为 `False`)，在开启后我们可以增大 batch_size，注意在小 batch_size 的条件下，开启 recompute 后显存变化不明显，只有当开大 batch_size 后才能明显感受到区别。
* `--bf16`: 是否使用 bf16 混合精度模式训练，默认是 fp32 训练。(`bool`，可选，默认为 `False`)
* `--fp16`: 是否使用 fp16 混合精度模式训练，默认是 fp32 训练。(`bool`，可选，默认为 `False`)
* `--fp16_opt_level`: 混合精度训练模式，可为 ``O1`` 或 ``O2`` 模式，默认 ``O1`` 模式，默认 ``O1`` 只在 fp16 选项开启时候生效。
* `--w_min`: 训练过程中CFG scale采样的最小值，sd15时默认值为`5.0`，sdxl时默认值为`3.0`。
* `--w_max`: 训练过程中CFG scale采样的最大值，默认为`15.0`。
* `--num_ddim_timesteps`: DDIM Sover采样的步数，默认为`50`。
* `--loss_type`: 损失类型，可从 `["l2", "huber"]` 选择，默认为`l2`。
* `--huber_c`: `huber loss`的参数，只有当`--loss_type=huber`时候才有作用，默认为`0.001`。
* `--is_lora`: 是否使用`lora`训练，默认为`True`。
* `--timestep_scaling_factor`: 在计算LCM的边界缩放时所使用的乘法时间步长缩放因子。缩放因子越高，近似误差就越低，默认值为`10.0`。
* `--lora_rank`: lora的rank大小，只有当`--is_lora=True`时候才有作用，默认为`64`。
* `--unet_time_cond_proj_dim`: `Unet`中`CFG embedding`嵌入的维度，只有当`--is_lora=False`时候才有作用，默认为`256`。
* `--ema_decay`: `EMA`更新`target_unet`所使用的`decay`参数，只有当`--is_lora=False`时候才有作用，默认为`0.95`。
* `--is_sdxl`: 是否使用`sdxl`模型训练，默认为`False`。
* `--use_fix_crop_and_size`: 是否为教师模型使用固定的裁剪和尺寸，只有当`--is_sdxl=True`时候才有作用，默认值为`True`。
* `--center_crop`: 是否对图片进行中心裁剪，只有当`--is_sdxl=True`时候才有作用，默认为`False`。
* `--random_flip`: 是否对图片进行随机旋转，只有当`--is_sdxl=True`时候才有作用，默认为`False`。

训练过程中，我们可以使用 `visualdl` 工具来查看训练过程中的`loss`变化及生成的图片。
```bash
visualdl --logdir . --host 0.0.0.0 --port 8765
```
![image](https://github.com/PaddlePaddle/PaddleMIX/assets/50394665/1c824085-3810-46c2-9b1b-42cd7cf71215)


### 4.3 单机多卡训练
```bash
export FLAGS_conv_workspace_size_limit=4096

# 2.6.0的时候会有很多类型提升的warning，GLOG_minloglevel=2将会关闭这些warning
export GLOG_minloglevel=2
export OUTPUT_DIR="lcm_lora_8gpus_outputs"
export BATCH_SIZE=12
export MAX_ITER=10000

# 如果使用自定义数据
FILE_LIST=./processed_data/filelist/custom_dataset.filelist.list
# 如果使用laion400m_demo数据集，需要把下面的注释取消
# FILE_LIST=./data/filelist/train.filelist.list

# 如果使用sd15
# BF16 O2 需要16G显存
MODEL_NAME_OR_PATH="runwayml/stable-diffusion-v1-5"
IS_SDXL=False
RESOLUTION=512

# 如果使用sdxl
# BF16 O2 需要46G显存
# MODEL_NAME_OR_PATH="stabilityai/stable-diffusion-xl-base-1.0"
# IS_SDXL=True
# RESOLUTION=1024

python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_lcm.py \
    --do_train \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --max_steps ${MAX_ITER} \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 100 \
    --logging_steps 10 \
    --resolution ${RESOLUTION} \
    --save_steps 2000 \
    --save_total_limit 20 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path ${MODEL_NAME_OR_PATH} \
    --file_list ${FILE_LIST} \
    --model_max_length 77 \
    --max_grad_norm 1 \
    --disable_tqdm True \
    --overwrite_output_dir \
    --recompute True \
    --loss_type "huber" \
    --lora_rank 64 \
    --is_sdxl ${IS_SDXL} \
    --is_lora True
```

### 4.4 多机多卡训练

需在 `paddle.distributed.launch` 后增加参数 `--ips IP1,IP2,IP3,IP4`，分别对应多台机器的 IP，更多信息可参考 [飞桨官网-分布式训练](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/cluster_quick_start_collective_cn.html)。

## 5. LCM-LoRA模型推理

```python
# 关闭类型提升出现的warning
import os
os.environ["GLOG_minloglevel"] = "2"
import paddle
from ppdiffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from ppdiffusers.utils import image_grid
from lcm import LCMScheduler, merge_weights

# 如果是sdxl模型
# 需要将 pretrained_model 替换为 "stabilityai/stable-diffusion-xl-base-1.0"
# 需要将 pipe_cls 替换为 StableDiffusionXLPipeline
pretrained_model = "runwayml/stable-diffusion-v1-5"
pipe_cls = StableDiffusionPipeline

pipe = pipe_cls.from_pretrained(
    pretrained_model,
    scheduler=LCMScheduler.from_pretrained(pretrained_model,
                                           subfolder="scheduler"),
    paddle_dtype=paddle.float16,
    safety_checker=None,
    requires_safety_checker=False,
)

# 合并我们训练好后的lora权重，注意这里的lora权重只支持kohya格式的safetensors权重。
lora_path = "./lcm_lora_outputs/checkpoint-2000/lora/lcm_lora.safetensors"
merge_weights(pipe.unet, lora_path)

# 合并完毕后，我们可以保存一份合并后的unet的权重，方便我们后续直接加载无需重复合并。
pipe.unet.save_pretrained("./merged_lcm_lora_unet")

generator = paddle.Generator().manual_seed(42)
validation_prompts = [
    "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
]

image_logs = []

for index, prompt in enumerate(validation_prompts):
    images = pipe(
        prompt=prompt,
        num_inference_steps=4,
        num_images_per_prompt=4,
        generator=generator,
        guidance_scale=1.0,
    ).images
    image_grid(images, 1, 4).save(f"image_{index}.png")
```
生成的图片如下所示：
<center><img src="https://github.com/PaddlePaddle/PaddleMIX/assets/50394665/9b849504-5c03-4f93-8723-6a7ea8da0bdf" width=100%></center>
<center><img src="https://github.com/PaddlePaddle/PaddleMIX/assets/50394665/9068dd09-a632-4824-81a7-6aa7200ae920" width=100%></center>
<center><img src="https://github.com/PaddlePaddle/PaddleMIX/assets/50394665/142d2623-2fb3-4ce7-9bbc-87f4eb3e36a5" width=100%></center>
<center><img src="https://github.com/PaddlePaddle/PaddleMIX/assets/50394665/b3d07354-4aaa-46be-b2f3-77bb5dc77d0f" width=100%></center>


## 6. 参考资料
- https://github.com/luosiallen/latent-consistency-model
- https://github.com/huggingface/diffusers/tree/main/examples/consistency_distillation

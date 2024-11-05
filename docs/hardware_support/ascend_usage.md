# PaddleMIX昇腾使用说明

为了满足用户对AI芯片多样化的需求， PaddleMIX 团队基于飞桨框架在硬件兼容性和灵活性方面的优势，深度适配了昇腾910芯片，为用户提供了国产计算芯片上的训推能力。只需安装说明安装多硬件版本的飞桨框架后，在模型配置文件中添加一个配置设备的参数，即可在相关硬件上使用PaddleMIX。当前PaddleMIX昇腾版适配涵盖了多模态理解模型InternVL2、LLaVA和多模态生成模型SD3、SDXL。未来我们将继续在用户使用的多种算力平台上适配 PaddleMIX 更多的模型，敬请期待。

## 1. 模型列表
<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>多模态理解</b>
      </td>
      <td>
        <b>多模态生成</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        </ul>
          <li><b>图文预训练</b></li>
        <ul>
            <li><a href="../../paddlemix/examples/llava">LLaVA-1.6</a></li>
            <li><a href="../../paddlemix/examples/internvl2">InternVL2</a></li>
      </ul>
      </td>
      <td>
        <ul>
        </ul>
          <li><b>文生图</b></li>
        <ul>
           <li><a href="../../ppdiffusers/examples/stable_diffusion">Stable Diffusion</a></li>
           <li><a href="../../ppdiffusers/examples/dreambooth/README_sd3.md">Stable Diffusion 3 (SD3)</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

## 2. 安装说明

### 2.1 创建标准化环境

当前 PaddleMIX 支持昇腾 910B 芯片，昇腾驱动版本为 23.0.3。考虑到环境差异性，我们推荐使用飞桨官方提供的标准镜像（支持x86服务器与Arm服务器）完成环境准备。

参考如下命令启动容器，ASCEND_RT_VISIBLE_DEVICES 指定可见的 NPU 卡号

```shell
docker run -it --name paddle-npu-dev -v $(pwd):/work \
    --privileged --network=host --shm-size=128G -w=/work \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    registry.baidubce.com/device/paddle-npu:cann80T13-ubuntu20-$(uname -m)-gcc84-py39 /bin/bash
```

### 2.2 安装飞桨

在容器内安装飞桨

```shell
# 注意需要先安装飞桨 cpu 版本，目前仅支持python3.9版本
python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python -m pip install --pre paddle-custom-npu -i https://www.paddlepaddle.org.cn/packages/nightly/npu/
```

### 2.3 安装PaddleMIX

克隆PaddleMIX仓库

```shell
# 使用最新发布的release/2.1分支
git clone https://github.com/PaddlePaddle/PaddleMIX -b release/2.1
cd PaddleMIX
```

### 2.4 安装依赖

```shell
sh build_env.sh
python -m pip install -U librosa
```

## 3. 多模态理解

多模态大模型（Multimodal LLM）是当前研究的热点，在 2024 年迎来了井喷式的发展，它将多模态输入经由特定的多模态 encoder 转化为与文本对齐的 token ，随后被输入到大语言模型中来执行多模态任务。PaddleMIX 2.1 新增了两大系列多模态大模型：InternVL2 系列和 Qwen2-VL 系列，同时支持指令微调训练和推理部署，模型能力覆盖了图片问答、文档图表理解、关键信息提取、场景文本理解、 OCR 识别、科学数学问答、视频理解、多图联合理解等。

InternVL2系列模型支持昇腾 910B 芯片上训练和推理，使用昇腾 910B 芯片训练推理时请先参考本文安装说明章节中的内容安装相应版本的飞桨框架。InternVL2模型训练推理使用方法参考如下:

### 3.1 微调训练

#### 3.1.1 数据准备

参照[文档](../../paddlemix/examples/internvl2)进行数据准备

#### 3.1.2 环境设置

设置NPU相关环境变量

```shell
export FLAGS_use_stride_kernel=0
export FLAGS_npu_storage_format=0 # 关闭私有格式
export FLAGS_npu_jit_compile=0 # 关闭即时编译
export FLAGS_npu_scale_aclnn=True # aclnn加速
export FLAGS_npu_split_aclnn=True # aclnn加速
export CUSTOM_DEVICE_BLACK_LIST=set_value,set_value_with_tensor # set_value加入黑名单

# 将ppdiffusers加入到PYTHONPATH中
export PYTHONPATH=`pwd`/ppdiffusers:$PYTHONPATH
```
#### 3.1.3 微调训练

执行微调训练，可以从[PaddleMIX工具箱介绍](../..//paddlemix/tools/README.md)查看详细的参数说明

```shell
# 以2B权重为例子
sh paddlemix/examples/internvl2/shell/internvl2.0/2nd_finetune/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_full.sh
```

### 3.2 推理

#### 3.2.1 环境设置

参考上述步骤设置环境变量

#### 3.2.2 执行推理

```shell
python paddlemix/examples/internvl2/chat_demo.py \
    --model_name_or_path "OpenGVLab/InternVL2-2B" \
    --image_path 'paddlemix/demo_images/examples_image1.jpg' \
    --text "Please describe this image in detail."
```

## 4. 多模态生成

PPDiffusers 提供了 SD3 的的个性化微调训练样例，只需要少量主题图像即可定制个性化 SD3 模型，支持 DreamBooth LoRA 微调及 DreamBooth 全参数微调。在推理上，提供 SD3 模型高性能推理实现。

多模态生成Stable Diffusion系列模型支持昇腾 910B 芯片上训练和推理，使用昇腾 910B 芯片训练推理时请先参考本文安装说明章节中的内容安装相应版本的飞桨框架。SDXL模型训练推理使用方法参考如下:

### 4.1 训练

#### 4.1.1 环境设置

昇腾 910B 芯片上进行SDXL训练时设置相应的环境变量

```shell
export FLAGS_npu_storage_format=0
export FLAGS_use_stride_kernel=0
export FLAGS_npu_scale_aclnn=True
export FLAGS_allocator_strategy=auto_growth

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/naruto-blip-captions"
export HF_ENDPOINT=https://hf-mirror.com
export FLAGS_conv_workspace_size_limit=4096

# 将ppdiffusers加入到PYTHONPATH中
export PYTHONPATH=`pwd`/ppdiffusers:$PYTHONPATH
```

#### 4.1.2 启动SDXL微调训练

```shell
python -u ppdiffusers/examples/text_to_image/train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=512 --center_crop --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=10000 \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --validation_prompt="a cute Sundar Pichai creature" --validation_epochs 5 \
  --checkpointing_steps=5000 \
  --output_dir="sdxl-pokemon-model"
```

#### 4.1.3 启动SDXL LoRA训练

```shell
python -u ppdiffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=1024 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=2 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="sd-pokemon-model-lora-sdxl" \
  --validation_prompt="cute dragon creature" \
  --report_to="wandb"
```

### 4.2 推理

推理脚本参考如下

```python
from ppdiffusers import StableDiffusionXLPipeline
from ppdiffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
import paddle

unet_path = "your-checkpoint/unet"
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", paddle_dtype=paddle.float16)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
unet = UNet2DConditionModel.from_pretrained(unet_path)

prompt = "A pokemon with green eyes and red legs."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("pokemon.png")
```

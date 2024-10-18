# Stable Diffusion XL 微调

`train_text_to_image_sdxl.py` 脚本展示了如何在你自己的数据集上微调 Stable Diffusion XL (SDXL) 模型。

🚨 这个脚本是实验性的。脚本会微调整个模型，而且很多时候模型会过拟合，并遇到像灾难性遗忘这样的问题。建议尝试不同的超参数以获得最佳结果。🚨

## 本地运行

### 安装依赖项

在运行脚本之前，确保安装了库的训练依赖项：

**重要**

为了确保你能成功运行最新版本的示例脚本，我们强烈推荐 **从源代码安装** 并保持安装是最新的，因为我们经常更新示例脚本并安装一些特定于示例的要求。为此，执行以下步骤在一个新的虚拟环境中：

```bash
git clone https://github.com/PaddlePaddle/PaddleMIX.git
cd PaddleMIX/ppdiffusers
pip install -e .
```

然后进入 `examples/text_to_image` 文件夹并运行
```bash
pip install -r requirements_sdxl.txt
```


### 训练

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/naruto-blip-captions"

export HF_ENDPOINT=https://hf-mirror.com
export FLAGS_conv_workspace_size_limit=4096


python -u train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  --enable_xformers_memory_efficient_attention \
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

**注释**：

* `train_text_to_image_sdxl.py` 脚本会预计算文本嵌入和VAE编码，并将它们保存在内存中。对于像 [`lambdalabs/naruto-blip-captions`](https://hf.co/datasets/lambdalabs/naruto-blip-captions) 这样的小数据集来说，这可能不是问题，但当脚本用于更大的数据集时，肯定会导致内存问题。对于这些情况，你可能会希望将这些预计算的表示序列化到磁盘上，并在微调过程中加载它们。有关更深入的讨论，请参阅 [这个 PR](https://github.com/huggingface/diffusers/pull/4505)。
* 训练脚本是计算密集型的，可能无法在消费级GPU上运行，比如 Tesla T4。
* 上面显示的训练命令在训练周期之间执行中间质量验证，并将结果记录到 Weights and Biases。`--report_to`、`--validation_prompt` 和 `--validation_epochs` 是这里相关的 CLI 参数。
* 众所周知，SDXL的VAE存在数值不稳定性问题。这就是为什么我们还暴露了一个 CLI 参数，即 `--pretrained_vae_model_name_or_path`，让你指定更好的VAE的位置（例如[这个](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)）。
* 不支持`--use_8bit_adam`

### 推理

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

可以通过以下代码进行多个checkpoint的推理：
```python
from ppdiffusers import StableDiffusionXLPipeline
from ppdiffusers import (
    AutoencoderKL,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
import paddle
import os

dir_name = "your-checkpoints-dir"
for file_name in sorted(os.listdir(dir_name)):
    print(file_name)
    unet_path = os.path.join(dir_name, file_name)

    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", paddle_dtype=paddle.float16)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
    unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder="unet")

    prompt = "A pokemon with green eyes and red legs."
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save("sdxl_train_pokemon_" + file_name + ".png")
```

## NPU硬件训练推理

1. 请先参照[PaddleCustomDevice](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/README_cn.md)安装NPU硬件Paddle
2. 使用NPU进行sdxl微调训练和推理时参考如下命令设置相应的环境变量，训练和推理运行命令可直接参照上述微调训练和推理命令。
```bash
export FLAGS_npu_storage_format=0
export FLAGS_use_stride_kernel=0
```

注意NPU训练暂不支持enable_xformers_memory_efficient_attention选项，启动命令如下:
```bash
python -u train_text_to_image_sdxl.py \
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


## Stable Diffusion XL (SDXL) LoRA 训练示例

Low-Rank Adaptation of Large Language Models 最初由 Microsoft 在 [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen* 提出。

简而言之，LoRA 允许通过向现有权重添加一对秩分解矩阵来调整预训练模型，并**仅**训练这些新添加的权重。这有几个优点：

- 以前的预训练权重被保持冻结，因此模型不容易遭受 [灾难性遗忘](https://www.pnas.org/doi/10.1073/pnas.1611835114)。
- 秩分解矩阵的参数远少于原始模型，这意味着训练后的 LoRA 权重很轻便。
- LoRA 注意力层允许通过 `scale` 参数控制模型适应新训练图像的程度。

[cloneofsimo](https://github.com/cloneofsimo) 是第一个尝试为 Stable Diffusion 在一个流行的 [lora](https://github.com/cloneofsimo/lora) GitHub 仓库中进行 LoRA 训练的人。

使用 LoRA，可以在消费级 GPU 上微调 Stable Diffusion 自定义图像-标题对数据集，比如 Tesla T4, Tesla V100。

### 训练

首先，你需要按照[安装部分](#安装依赖项)中解释的设置开发环境。确保设置了 `MODEL_NAME` 和 `DATASET_NAME` 环境变量，以及可选的 `VAE_NAME` 变量。这里，我们将使用 [Stable Diffusion XL 1.0-base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 和 [Pokemons 数据集](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions)。

**___注：通过在训练过程中定期生成样本图像来监控训练进度非常有用。[Weights and Biases](https://docs.wandb.ai/quickstart) 是一个很好的解决方案，可以轻松地在训练过程中查看生成的图像。你需要做的就是在训练前运行 `pip install wandb`，以自动记录图像。___**

```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/naruto-blip-captions"

export HF_ENDPOINT=https://hf-mirror.com
export FLAGS_conv_workspace_size_limit=4096
```


现在我们可以开始训练了！

```bash
python -u train_text_to_image_lora_sdxl.py \
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
  --validation_prompt="cute dragon creature" --report_to="wandb"
```

上述命令还将在微调过程中执行推理，并将结果记录到 Weights and Biases。

**注释**：

* 众所周知，SDXL的VAE存在数值不稳定性问题。这就是为什么我们还暴露了一个 CLI 参数，即 `--pretrained_vae_model_name_or_path`，让你指定更好的VAE的位置（例如[这个](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)）。
* 不支持`--use_8bit_adam`



### 微调文本编码器和 UNet

脚本还允许你微调 `text_encoder` 以及 `unet`。

🚨 训练文本编码器需要额外的内存。

将 `--train_text_encoder` 参数传递给训练脚本以启用微调 `text_encoder` 和 `unet`：

```bash
python -u train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=1024 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=2 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="sd-pokemon-model-lora-sdxl-txt" \
  --train_text_encoder \
  --validation_prompt="cute dragon creature" --report_to="wandb"
```

### 推理

一旦你使用上面的命令训练了一个模型，推理可以简单地使用 `StableDiffusionXLPipeline` 在加载训练好的 LoRA 权重后进行。通过修改推理脚本中的model_path变量，可以传递需要加载的 LoRA 训练权重，在这个案例中，是 `sd-pokemon-model-lora-sdxl`。

```python
from ppdiffusers import StableDiffusionXLPipeline
import paddle

model_path = "takuoko/sd-pokemon-model-lora-sdxl"
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", paddle_dtype=paddle.float16)
pipe.load_lora_weights(model_path)

prompt = "A pokemon with green eyes and red legs."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("pokemon.png")
```

如果想进行多个checkpoint的推理，你可以使用下面的代码。
```python
# multi image
from ppdiffusers import StableDiffusionXLPipeline
import paddle
import os

dir_name = "your-checkpoints-path/sd-pokemon-model-lora-sdxl/"
for file_name in sorted(os.listdir(dir_name)):
    if 'checkpoint' not in file_name:
        continue
    print(file_name)
    model_path = os.path.join(dir_name, file_name)
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", paddle_dtype=paddle.float16)
    pipe.load_lora_weights(model_path)

    prompt = "A pokemon with green eyes and red legs."
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save("pokemon_" + file_name + ".png")
```

## NPU硬件训练
1. 请先参照[PaddleCustomDevice](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/README_cn.md)安装NPU硬件Paddle
2. 使用NPU进行LoRA训练和推理时参考如下命令设置相应的环境变量，训练和推理运行命令可直接参照上述LoRA训练和推理命令。
```bash
export FLAGS_npu_storage_format=0
export FLAGS_use_stride_kernel=0
export FLAGS_npu_scale_aclnn=True
export FLAGS_allocator_strategy=auto_growth
```

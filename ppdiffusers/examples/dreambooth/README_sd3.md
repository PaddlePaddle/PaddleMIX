# DreamBooth训练示例：Stable Diffusion 3 (SD3)

[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) 是一种用于个性化文本到图像模型的方法，只需要主题的少量图像（3~5张）即可。

`train_dreambooth_sd3.py` 脚本展示了如何进行DreamBooth全参数微调[Stable Diffusion 3](https://huggingface.co/papers/2403.03206)， `train_dreambooth_lora_sd3.py` 脚本中展示了如何进行DreamBooth LoRA微调。


> [!NOTE]
> Stable Diffusion 3遵循 [Stability Community 开源协议](https://stability.ai/license)。
> Community License: Free for research, non-commercial, and commercial use for organisations or individuals with less than $1M annual revenue. You only need a paid Enterprise license if your yearly revenues exceed USD$1M and you use Stability AI models in commercial products or services. Read more: https://stability.ai/license


## DreamBooth微调

### 安装依赖

在运行脚本之前，请确保安装了库的训练依赖项：

```bash
pip install -r requirements_sd3.txt
```



### 示例
首先需要获取示例数据集。在这个示例中，我们将使用一些狗的图像：https://paddlenlp.bj.bcebos.com/models/community/westfish/develop-sdxl/dog.zip 。

解压数据集``unzip dog.zip``后，使用以下命令启动训练：

```bash
export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-sd3"
wandb offline
```

```bash
python train_dreambooth_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" \
  --checkpointing_steps=250
```

fp16训练需要显存67000MiB，为了更好地跟踪我们的训练实验，我们在上面的命令中使用了以下标志：

* `report_to="wandb"` 将确保在 Weights and Biases 上跟踪训练运行。要使用它，请确保安装 `wandb`，使用 `pip install wandb`。
* `validation_prompt` 和 `validation_epochs` 允许脚本进行几次验证推理运行。这可以让我们定性地检查训练是否按预期进行。


### 推理
训练完成后，我们可以通过以下python脚本执行推理：
```python
from ppdiffusers import StableDiffusion3Pipeline
from ppdiffusers import (
    AutoencoderKL,
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
)
import paddle

transformer_path = "your-checkpoint/transformer"

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", paddle_dtype=paddle.float16
)
transformer = SD3Transformer2DModel.from_pretrained(transformer_path)

image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
image.save("sks_dog_dreambooth_finetune.png")
```



## LoRA + DreamBooth

[LoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) 是一种流行的节省参数的微调技术，允许您以极少的可学习参数实现全微调的性能。

要使用 LoRA 进行 DreamBooth，运行：

```bash
export MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-sd3-lora"
export USE_PEFT_BACKEND=True
wandb offline

python train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=5e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" \
  --checkpointing_steps=250
```

fp16训练需要显存47000MiB，。训练完成后，我们可以通过以下python脚本执行推理：
```python
from ppdiffusers import StableDiffusion3Pipeline
from ppdiffusers import (
    AutoencoderKL,
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
)
import paddle

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", paddle_dtype=paddle.float16
)
pipe.load_lora_weights('your-lora-checkpoint')

image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
image.save("sks_dog_dreambooth_lora.png")
```

## NPU硬件训练
1. 请先参照[PaddleCustomDevice](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/README_cn.md)安装NPU硬件Paddle
2. 使用NPU进行LoRA训练和推理时参考如下命令设置相应的环境变量，训练和推理运行命令可直接参照上述LoRA训练和推理命令。

使用NPU进行LoRA训练和推理时参考如下命令设置相应的环境变量，训练和推理运行命令可直接参照上述LoRA训练和推理命令。
```bash
export FLAGS_npu_storage_format=0
export FLAGS_use_stride_kernel=0
export FLAGS_npu_scale_aclnn=True
export FLAGS_allocator_strategy=auto_growth
```
训练(DreamBooth微调)时如果显存不够，可以尝试添加参数(训练完成后不进行评测)`not_validation_final`, 并去除`validation_prompt`，具体命令如下所示
```
python train_dreambooth_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50 \
  --validation_epochs=25 \
  --seed="0" \
  --checkpointing_steps=250 \
  --not_validation_final
```

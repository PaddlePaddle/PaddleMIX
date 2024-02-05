# Stable Diffusion XL (SDXL) 的 DreamBooth 训练示例
[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) 是一种用于个性化文本到图像模型的方法，只需要主题的少量图像（3~5张）即可。

`train_dreambooth_lora_sdxl.py` 脚本展示了如何实施训练过程，并将其适应于 Stable Diffusion XL。

💡 注意：目前，我们仅支持通过 LoRA 对 SDXL UNet 进行 DreamBooth 微调。LoRA 是一种参数高效的微调技术，具体参考[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)。

## 本地训练
### 安装依赖项
在运行脚本之前，请确保安装了库的训练依赖项：

```bash
pip install -r requirements.txt
```

### 示例
首先需要获取示例数据集。在这个示例中，我们将使用一些狗的图像：https://paddlenlp.bj.bcebos.com/models/community/westfish/develop-sdxl/dog.zip 。

解压数据集``unzip dog.zip``后，使用以下命令启动训练：
```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="lora-trained-xl"
```

```
python train_dreambooth_lora_sdxl.py \
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
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed="0" \
  --checkpointing_steps=100
```


#### 推理
训练完成后，我们可以执行推理，如下所示：

```python
from ppdiffusers import DiffusionPipeline
from ppdiffusers import DDIMScheduler

import paddle

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", paddle_dtype=paddle.float16)
pipe.load_lora_weights("paddle_lora_weights.safetensors")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
image.save("sks_dog.png")
```
<p align="center">
    <img src="https://user-images.githubusercontent.com/20476674/267534284-4c203609-4e9a-449c-82f3-4592a564a1fc.png">
</p>

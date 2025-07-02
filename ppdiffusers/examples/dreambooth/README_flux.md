# DreamBooth训练示例：FLUX

[DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) 是一种用于个性化文本到图像模型的方法，只需要主题的少量图像（3~5张）即可。

`train_dreambooth_lora_flux.py` 脚本中展示了如何进行DreamBooth LoRA微调。


> [!NOTE]
> FLUX LoRA 微调需要40GB以上的显存。


## DreamBooth LoRA微调

### 安装依赖

在运行脚本之前，请确保安装了库的训练依赖项：

```bash
pip install -r requirements_flux.txt
```


### 示例
首先需要获取示例数据集。在这个示例中，我们将使用一些狗的图像：https://paddlenlp.bj.bcebos.com/models/community/westfish/develop-sdxl/dog.zip 。

解压数据集``unzip dog.zip``后，使用以下命令启动训练：

```bash
export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-flux-lora"
export USE_PEFT_BACKEND=True
wandb offline

python train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=20 \
  --seed="0" \
  --not_validation_final \
  --checkpointing_steps=250
```

为了更好地跟踪我们的训练实验，我们在上面的命令中使用了以下标志：
* `report_to="wandb"` 将确保在 Weights and Biases 上跟踪训练运行。要使用它，请确保安装 `wandb`，使用 `pip install wandb`。
* `validation_prompt` 和 `validation_epochs` 允许脚本进行几次验证推理运行。这可以让我们定性地检查训练是否按预期进行。

在H100等显卡训练时，需要加上以下环境变量：
```bash
export FLAGS_sdpa_select_math="yes"
export FLAGS_use_fused_rmsnorm="yes"
```



### 推理
训练完成后，我们可以通过以下python脚本执行推理：
```python
from ppdiffusers import FluxPipeline
import paddle

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16
)
pipe.load_lora_weights('your-lora-checkpoint')

image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
image.save("sks_dog_dreambooth_lora.png")
```

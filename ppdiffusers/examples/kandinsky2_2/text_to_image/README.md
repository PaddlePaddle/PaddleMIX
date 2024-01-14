# 微调Kandinsky2.2 text-to-image

Kandinsky 2.2 包含一个根据文本提示生成图像嵌入表示的prior pipeline和一个根据图像嵌入表示生成输出图像的decoder pipeline。我们提供了 'train_text_to_image_prior.py '和 train_text_to_image_decoder.py '脚本，向您展示如何根据自己的数据集分别微调 Kandinsky prior模型和decoder模型。为了达到最佳效果，您应该对prior模型和decoder模型均进行微调。

___注意___:

___这个脚本是试验性的。该脚本会对整个'decoder'或'prior'模型进行微调，有时模型会过度拟合，并遇到像`"catastrophic forgetting"`等问题。建议尝试不同的超参数，以便在自定义``数据集上获得最佳结果。___

## 1 安装依赖

在运行这个训练代码前，我们需要安装下面的训练依赖:

```bash
# 安装2.6.0版本的paddlepaddle-gpu，当前我们选择了cuda12.0的版本，可以查看 https://www.paddlepaddle.org.cn/ 寻找自己适合的版本
python -m pip install paddlepaddle-gpu==2.6.0.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# 安装所需的依赖, 如果提示权限不够，请在最后增加 --user 选项
pip install -r requirements.txt
```

### 2 Pokemon训练示例

#### 2.1 微调 decoder

```bash
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

python -u train_text_to_image_decoder.py \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="kandi2-decoder-pokemon-model" 
```

<!-- accelerate_snippet_end -->

如果用户想要在自己的数据集上进行训练，那么需要根据`huggingface的 datasets 库`所需的格式准备数据集，有关数据集的介绍可以查看 [HF dataset的文档](https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder-with-metadata).

如果用户想要修改代码中的部分训练逻辑，那么需要修改训练代码。

```bash
export TRAIN_DIR="path_to_your_dataset"

python -u train_text_to_image_decoder.py \
  --train_data_dir=$TRAIN_DIR \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="kandi2-decoder-pokemon-model" 
```

训练完成后，模型将保存在命令中指定的 `output_dir` 目录中。在本例中是 `kandi22-decoder-pokemon-model`。要加载微调后的模型进行推理，后将该路径传递给 `KandinskyV22CombinedPipeline` :

```python
from ppdiffusers import KandinskyV22CombinedPipeline

pipe = KandinskyV22CombinedPipeline.from_pretrained(output_dir)

prompt='A robot pokemon, 4k photo'
images = pipe(prompt=prompt).images
images[0].save("robot-pokemon.png")
```

Checkpoints只保存 unet，因此要从checkpoints运行推理只需加载 unet：

```python
from ppdiffusers import KandinskyV22CombinedPipeline, UNet2DConditionModel

model_path = "path_to_saved_model"

unet = UNet2DConditionModel.from_pretrained(model_path + "/checkpoint-<N>/unet")

pipe = KandinskyV22CombinedPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", unet=unet)

images = pipe(prompt="A robot pokemon, 4k photo").images
images[0].save("robot-pokemon.png")
```

#### 2.2 微调 prior

您可以使用`train_text_to_image_prior.py`脚本对Kandinsky prior模型进行微调：

```bash
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

python -u train_text_to_image_prior.py \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="kandi2-prior-pokemon-model" 
```

<!-- accelerate_snippet_end -->

要使用微调prior模型进行推理，首先需要将 `output_dir`传给 `DiffusionPipeline`从而创建一个prior pipeline。然后，从一个预训练或微调decoder以及刚刚创建的prior pipeline的所有模块中创建一个`KandinskyV22CombinedPipeline`：

```python
from ppdiffusers import KandinskyV22CombinedPipeline, DiffusionPipeline

pipe_prior = DiffusionPipeline.from_pretrained(output_dir)
prior_components = {"prior_" + k: v for k,v in pipe_prior.components.items()}
pipe = KandinskyV22CombinedPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", **prior_components)

prompt='A robot pokemon, 4k photo'
images = pipe(prompt=prompt, negative_prompt=negative_prompt).images
images[0]
```

如果您想使用微调decoder和微调prior checkpoint，只需将上述代码中的 "kandinsky-community/kandinsky-2-2-decoder "替换为您自定义的模型库名称即可。请注意，要创建 `KandinskyV22CombinedPipeline`，您的模型 repo 必须有prior  tag。如果您使用我们的训练脚本创建了模型 repo，则会自动包含prior tag。

#### 2.3 单机多卡训练

通过设置`--gpus`，我们可以指定 GPU 为 `0,1,2,3` 卡。这里我们只训练了`4000step`，因为这里的`4000 step x 4卡`近似于`单卡训练 16000 step`。

```bash
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

python -u -m paddle.distributed.launch --gpus "0,1,2,3" train_text_to_image_decoder.py \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=4000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="kandi2-decoder-pokemon-model"  
```

# 使用 LoRA 和 Text-to-Image 技术进行模型训练

[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) 是微软研究员引入的一项新技术，主要用于处理大模型微调的问题。目前超过数十亿以上参数的具有强能力的大模型 (例如 GPT-3) 通常在为了适应其下游任务的微调中会呈现出巨大开销。LoRA 建议冻结预训练模型的权重并在每个 Transformer 块中注入可训练层 (秩-分解矩阵)。因为不需要为大多数模型权重计算梯度，所以大大减少了需要训练参数的数量并且降低了 GPU 的内存要求。研究人员发现，通过聚焦大模型的 Transformer 注意力块，使用 LoRA 进行的微调质量与全模型微调相当，同时速度更快且需要更少的计算。

简而言之，LoRA允许通过向现有权重添加一对秩分解矩阵，并只训练这些新添加的权重来适应预训练的模型。这有几个优点：

- 保持预训练的权重不变，这样模型就不容易出现灾难性遗忘 [catastrophic forgetting](https://www.pnas.org/doi/10.1073/pnas.1611835114)；

- 秩分解矩阵的参数比原始模型少得多，这意味着训练的 LoRA 权重很容易移植；

- LoRA 注意力层允许通过一个 `scale` 参数来控制模型适应新训练图像的程度。

[cloneofsimo](https://github.com/cloneofsimo) 是第一个在 [LoRA GitHub](https://github.com/cloneofsimo/lora) 仓库中尝试使用 LoRA 训练 Stable Diffusion 的人。

利用 LoRA，可以在 T4、Tesla V100 等消费级 GPU 上，在自定义图文对数据集上对 Kandinsky 2.2 进行微调。

### 1 训练

#### 1.1 LoRA微调decoder

```bash
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

python -u train_text_to_image_decoder_lora.py \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --lora_rank=4 \
  --validation_prompt="cute dragon creature" \
  --output_dir="kandi22-decoder-pokemon-lora"
```

#### 1.2 LoRA微调prior

```bash
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

python -u train_text_to_image_prior_lora.py \
  --dataset_name=$DATASET_NAME \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --checkpoints_total_limit=3 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --lora_rank=4 \
  --validation_prompt="cute dragon creature" \
  --output_dir="kandi22-prior-pokemon-lora"
```

**___Note: 当我使用 LoRA 训练模型的时候，我们需要使用更大的学习率，因此我们这里使用 *1e-4* 而不是 *1e-5*.___**

### 2 推理

#### 2.1 LoRA微调decoder推理流程

使用上述命令训练好Kandinsky decoder模型后，就可以在加载训练好 LoRA 权重后使用 `KandinskyV22CombinedPipeline` 进行推理。您需要传递用于加载 LoRA 权重的 `output_dir` 目录，在本例中是 `kandi22-decoder-pokemon-lora`。

```python
from ppdiffusers import KandinskyV22CombinedPipeline

pipe = KandinskyV22CombinedPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")
pipe.unet.load_attn_procs(output_dir)

prompt='A robot pokemon, 4k photo'
image = pipe(prompt=prompt).images[0]
image.save("robot_pokemon.png")
```

#### 2.2 LoRA微调prior推理流程

```python
from ppdiffusers import KandinskyV22CombinedPipeline

pipe = KandinskyV22CombinedPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")
pipe.prior_prior.load_attn_procs(output_dir)

prompt='A robot pokemon, 4k photo'
image = pipe(prompt=prompt).images[0]
image.save("robot_pokemon.png")
```

# 参考资料

- https://github.com/huggingface/diffusers/tree/main/examples/kandinsky2_2/text_to_image

- https://github.com/ai-forever/Kandinsky-2

- https://huggingface.co/kandinsky-community
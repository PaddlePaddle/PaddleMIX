# Stable Diffusion

## 1. 模型简介

Stable Diffusion 是一个基于 Latent Diffusion Models（潜在扩散模型，LDMs）的文图生成（text-to-image）模型。具体来说，得益于 [Stability AI](https://stability.ai/) 的计算资源支持和 [LAION](https://laion.ai/) 的数据资源支持，Stable Diffusion 在 [LAION-5B](https://laion.ai/blog/laion-5b/) 的一个子集上训练了一个 Latent Diffusion Models，该模型专门用于文图生成。Latent Diffusion Models 通过在一个潜在表示空间中迭代“去噪”数据来生成图像，然后将表示结果解码为完整的图像，让文图生成能够在消费级 GPU 上，在10秒级别时间生成图片，大大降低了落地门槛，也带来了文图生成领域的大火。所以，如果你想了解 Stable Diffusion 的背后原理，可以先深入解读一下其背后的论文 [High-Resolution Image Synthesis with Latent Diffusion Models](https://ommer-lab.com/research/latent-diffusion-models/)。如果你想了解更多关于 Stable Diffusion 模型的信息，你可以查看由 🤗Huggingface 团队撰写的相关[博客](https://huggingface.co/blog/stable_diffusion)。


<p align="center">
  <img src="https://github.com/CompVis/stable-diffusion/assets/50394665/268401d7-0a90-4a71-aba8-917949b63a2a" align="middle" width = "600" />
</p>
<p align="center">
  <img src="https://github.com/CompVis/latent-diffusion/assets/50394665/502f620b-900b-43c5-a970-9e1b884c3f32" align="middle" width = "600" />
</p>

注：模型结构图引自[CompVis/latent-diffusion仓库](https://github.com/CompVis/latent-diffusion)，生成图片引用自[CompVis/stable-diffusion仓库](https://github.com/CompVis/stable-diffusion)。


### Stable Diffusion Model zoo

<div align="center">

| model name | params | 
|------------|:-------:|
| `CompVis/stable-diffusion-v1-4` | 
| `runwayml/stable-diffusion-v1-5` | 

</div>


## 2. 环境准备
通过 `git clone` 命令拉取 PaddleMIX 源码，并安装必要的依赖库。

```bash
# 克隆 PaddleMIX 仓库
git clone https://github.com/PaddlePaddle/PaddleMIX

# 进入stable diffusion目录
cd PaddleMIX/ppdiffusers/examples/stable_diffusion

# 安装所需的依赖, 如果提示权限不够，请在最后增加 --user 选项
pip install -r requirements.txt
```

> 注：本模型训练与推理需要依赖 CUDA 11.2 及以上版本，如果本地机器不符合要求，建议前往 [AI Studio](https://aistudio.baidu.com/index) 进行模型训练、推理任务。

## 3. 数据准备

预训练 Stable Diffusion 使用 Laion400M 数据集，需要自行下载和处理，处理步骤详见 3.1自定义训练数据。本教程为了方便大家 **体验跑通训练流程**，提供了处理后的 Laion400M 部分数据集，可直接下载获取，详见 3.2。


### 3.1 自定义训练数据

如果需要自定义数据，推荐沿用`coco_karpathy`数据格式处理自己的数据。其中每条数据标注格式示例为:
```text
{"caption": "A woman wearing a net on her head cutting a cake. ", "image": "val2014/COCO_val2014_000000522418.jpg", "image_id": "coco_522418"}
```

在准备好自定义数据集以后，我们可以使用 `create_pretraining_data.py` 生成我们需要的数据。

```bash
python create_pretraining_data.py \
    --input_path ./coco_data/coco_data.jsonl \
    --output_path ./processed_data \
    --caption_key "caption" \
    --image_key "image" \
    --per_part_file_num 1000 \
    --num_repeat 100 \
    --save_gzip_file
```

[create_pretraining_data.py](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/examples/stable_diffusion/create_pretraining_data.py) 可传入的参数解释如下：
* `--input_path`: 输入的 jsonl 文件路径，可以查看 `coco_data` 文件夹的组织结构，自定义我们自己的数据。
* `--output_path`: 处理后的数据保存路径。
* `--output_name`: 输出文件的名称，默认为`custom_dataset`。
* `--caption_key`: jsonl文件中，每一行数据表示文本的 key 值，默认为`caption`。
* `--image_key`: jsonl文件中，每一行数据表示图片的 key 值，默认为`image`。
* `--per_part_file_num`: 每个part文件保存的数据数量，默认为`1000`。
* `--save_gzip_file`: 是否将文件保存为`gzip`的格式，默认为`False`。
* `--num_repeat`: `custom_dataset.filelist`文件中`part数据`的重复次数，默认为`1`。当前我们设置成`100`是为了能够制造更多的`part数据`，可以防止程序运行时会卡住，如果用户有很多数据的时候，无需修改该默认值。

运行上述命令后，会生成 `./processed_data` 文件夹。
```
processed_data
├── filelist
|   ├── custom_dataset.filelist.list
|   └── custom_dataset.filelist
└── laion400m_format_data
    └── part-000001.gz
```

`processed_data/custom_dataset.filelist` 是数据索引文件，包含100行数据，每行都代表一个数据文件的路径。请确保该文件的行数足够多，以防止在训练过程中出现卡顿，内容如下所示：
```
processed_data/laion400m_format_data/part-000001.gz
processed_data/laion400m_format_data/part-000001.gz
processed_data/laion400m_format_data/part-000001.gz
processed_data/laion400m_format_data/part-000001.gz
processed_data/laion400m_format_data/part-000001.gz
processed_data/laion400m_format_data/part-000001.gz
processed_data/laion400m_format_data/part-000001.gz
processed_data/laion400m_format_data/part-000001.gz
...
```
`processed_data/custom_dataset.filelist.list` 为filelist索引文件，内容如下所示：
```
processed_data/filelist/custom_dataset.filelist
```
`processed_data/laion400m_format_data/part-000001.gz` 为实际的数据文件，内容结构如下所示：

每一行以`"\t"`进行分割，第一列为 `caption文本描述`, 第二列为 `占位符空`, 第三列为 `base64编码的图片`，示例：`caption, _, img_b64 = vec[:3]`


### 3.2 Laion400M Demo 数据集（部分数据，约1000条，仅供验证跑通训练）

demo 数据可通过如下命令下载与解压：

```bash
# 删除当前目录下的data
rm -rf data
# 下载 laion400m_demo 数据集
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/laion400m_demo_data.tar.gz
# 解压
tar -zxvf laion400m_demo_data.tar.gz
```

解压后文件目录如下所示：
```
data
├── filelist
|   ├── train.filelist.list
|   └── laion400m_en.filelist
├── laion400m_new
|   └── part-00001.gz
└── laion400m_demo_data.tar.gz # 多余的压缩包，可以删除
```

`laion400m_en.filelist` 是数据索引文件，包含了6000行数据文件的路径（part-00001.gz 仅为部分数据），内容如下所示：
```
./data/laion400m_new/part-00001.gz
./data/laion400m_new/part-00001.gz
./data/laion400m_new/part-00001.gz
./data/laion400m_new/part-00001.gz
./data/laion400m_new/part-00001.gz
./data/laion400m_new/part-00001.gz
./data/laion400m_new/part-00001.gz
...
```

## 4. 训练

Stable Diffusion 模型包含 3 个组成部分：vae、text_encoder、unet，其中预训练仅需随机初始化 unet 部分，其余部分可直接加载预训练权重，本教程中我们加载 `CompVis/stable-diffusion-v1-4` 中的预训练好的 `vae` 以及`text_encoder` 权重，随机初始化了 `unet` 模型权重。

### 4.1 硬件要求

示例脚本配置在显存 ≥40GB 的显卡上可正常训练，如显存不满足要求，可通过修改参数的方式运行脚本：
- 如果本地环境显存不够，请使用 AIStudio 上 32G 显存的 GPU 环境，并修改 `--per_device_train_batch_size` 为 32。
- bf16 混合精度训练模式支持 A100、3090、3080 等硬件，不支持使用 V100 进行训练，如果你的硬件满足要求，修改 `--bf16` 为 `True` 可启动混合精度训练模式，体验更快速的训练。

### 4.2 单机单卡训练

> 注意，我们当前训练的分辨率是 `256x256` ，如果需要训练 `512x512` 分辨率，请修改 `--resolution` 为 512 并且降低`--per_device_train_batch_size` 参数，否则会报显存不足的错误。

单机单卡训练启动脚本如下，建议保存为 `train.sh` 后执行命令 `sh train.sh`：

```bash
export FLAG_FUSED_LINEAR=0
export FLAGS_conv_workspace_size_limit=4096
# 是否开启 ema
export FLAG_USE_EMA=0
# 是否开启 recompute
export FLAG_RECOMPUTE=1
# 是否开启 xformers
export FLAG_XFORMERS=1

# 如果使用自定义数据
FILE_LIST=./processed_data/filelist/custom_dataset.filelist.list
# 如果使用laion400m_demo数据集，需要把下面的注释取消
# FILE_LIST=./data/filelist/train.filelist.list

python -u train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ./laion400m_pretrain_output_trainer \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --max_steps 200000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 1000 \
    --logging_steps 10 \
    --resolution 256 \
    --save_steps 10000 \
    --save_total_limit 20 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --vae_name_or_path CompVis/stable-diffusion-v1-4/vae \
    --text_encoder_name_or_path CompVis/stable-diffusion-v1-4/text_encoder \
    --unet_name_or_path ./sd/unet_config.json \
    --file_list ${FILE_LIST} \
    --model_max_length 77 \
    --max_grad_norm -1 \
    --disable_tqdm True \
    --bf16 False
```

[train_txt2img_laion400m_trainer.py](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/examples/stable_diffusion/train_txt2img_laion400m_trainer.py) 可传入的参数解释如下：
* `--vae_name_or_path`: 预训练 `vae` 模型名称或地址，`CompVis/stable-diffusion-v1-4/vae`为`kl-8.ckpt` ，程序将自动从 BOS 上下载预训练好的权重，默认值为 `None`。
* `--text_encoder_name_or_path`: 预训练 `text_encoder` 模型名称或地址，当前仅支持 `CLIPTextModel`，默认值为 `None`。
* `--unet_name_or_path`: 预训练 `unet` 模型名称或地址，默认值为 `None`。
* `--pretrained_model_name_or_path`: 加载预训练模型的名称或本地路径，如 `CompVis/stable-diffusion-v1-4`，`vae_name_or_path`，`text_encoder_name_or_path` 和 `unet_name_or_path` 的优先级高于 `pretrained_model_name_or_path`。
* `--per_device_train_batch_size`: 训练时每张显卡所使用的 `batch_size批量`，当我们的显存较小的时候，需要将这个值设置的小一点。
* `--gradient_accumulation_steps`: 梯度累积的步数，用户可以指定梯度累积的步数，在梯度累积的 step 中。减少多卡之间梯度的通信，减少更新的次数，扩大训练的 batch_size。
* `--learning_rate`: 学习率。
* `--unet_learning_rate`: `unet` 的学习率，这里的学习率优先级将会高于 `learning_rate`，默认值为 `None`。
* `--train_text_encoder`: 是否同时训练 `text_encoder`，默认值为 `False`。
* `--text_encoder_learning_rate`: `text_encoder` 的学习率，默认值为 `None`。
* `--weight_decay`: AdamW 优化器的 `weight_decay`。
* `--max_steps`: 最大的训练步数。
* `--save_steps`: 每间隔多少步 `（global step步数）`，保存模型。
* `--save_total_limit`: 最多保存多少个模型。
* `--lr_scheduler_type`: 要使用的学习率调度策略。默认为 `constant`。
* `--warmup_steps`: 用于从 0 到 `learning_rate` 的线性 warmup 的步数。
* `--resolution`: 预训练阶段将训练的图像的分辨率，默认为 `512`。
* `--noise_offset`: 预训练阶段生成操作时的偏移量，默认为 `0`。
* `--snr_gamma`: 平衡损失时使用的 SNR 加权 gamma 值。建议为`5.0`，默认为 `None`。更多细节在这里：https://arxiv.org/abs/2303.09556 。
* `--input_perturbation`: 输入扰动的尺度，推荐为 `0.1`，默认值为 `0`。
* `--image_logging_steps`: 每隔多少步，log 训练过程中的图片，默认为 `1000` 步，注意 `image_logging_steps` 需要是 `logging_steps` 的整数倍。
* `--logging_steps`: logging 日志的步数，默认为 `50` 步。
* `--output_dir`: 模型保存路径。
* `--seed`: 随机种子，为了可以复现训练结果，Tips：当前 paddle 设置该随机种子后仍无法完美复现。
* `--dataloader_num_workers`: Dataloader 所使用的 `num_workers` 参数，请确保处理后的`part文件`数量要大于等于`dataloader_num_workers` * `num_gpus`，否则程序会卡住，例如：`dataloader_num_workers=4`、`num_gpus=2`时候，请确保切分后的`part文件`数量要大于等于`8`。
* `--file_list`: file_list 文件地址。
* `--num_inference_steps`: 推理预测时候使用的步数。
* `--model_max_length`: `tokenizer` 中的 `model_max_length` 参数，超过该长度将会被截断。
* `--tokenizer_name`: 我们需要使用的 `tokenizer_name`。
* `--prediction_type`: 预测类型，可从 `["epsilon", "v_prediction"]` 选择。
* `--use_ema`: 是否对 `unet` 使用 `ema`，默认为 `False`。
* `--max_grad_norm`: 梯度剪裁的最大 norm 值，`-1` 表示不使用梯度裁剪策略。
* `--recompute`: 是否开启重计算，(`bool`，可选，默认为 `False`)，在开启后我们可以增大 batch_size，注意在小 batch_size 的条件下，开启 recompute 后显存变化不明显，只有当开大 batch_size 后才能明显感受到区别。
* `--bf16`: 是否使用 bf16 混合精度模式训练，默认是 fp32 训练。(`bool`，可选，默认为 `False`)
* `--fp16`: 是否使用 fp16 混合精度模式训练，默认是 fp32 训练。(`bool`，可选，默认为 `False`)
* `--fp16_opt_level`: 混合精度训练模式，可为 ``O1`` 或 ``O2`` 模式，默认 ``O1`` 模式，默认 ``O1`` 只在 fp16 选项开启时候生效。
* `--enable_xformers_memory_efficient_attention`: 是否开启 `xformers`，开启后训练速度会变慢，但是能够节省显存。注意我们需要安装大于等于 2.5.2 版本的 paddlepaddle！
* `--only_save_updated_model`: 是否仅保存经过训练的权重，比如保存 `unet`、`ema 版 unet`、`text_encoder`，默认值为 `True`。


### 4.3 单机多卡训练
```bash
export FLAG_FUSED_LINEAR=0
export FLAGS_conv_workspace_size_limit=4096
# 是否开启 ema
export FLAG_USE_EMA=0
# 是否开启 recompute
export FLAG_RECOMPUTE=1
# 是否开启 xformers
export FLAG_XFORMERS=1

# 如果使用自定义数据
FILE_LIST=./processed_data/filelist/custom_dataset.filelist.list
# 如果使用laion400m_demo数据集，需要把下面的注释取消
# FILE_LIST=./data/filelist/train.filelist.list

python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ./laion400m_pretrain_output_trainer \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --max_steps 200000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 1000 \
    --logging_steps 10 \
    --resolution 256 \
    --save_steps 10000 \
    --save_total_limit 20 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --vae_name_or_path CompVis/stable-diffusion-v1-4/vae \
    --text_encoder_name_or_path CompVis/stable-diffusion-v1-4/text_encoder \
    --unet_name_or_path ./unet_config.json \
    --file_list ${FILE_LIST} \
    --model_max_length 77 \
    --max_grad_norm -1 \
    --disable_tqdm True \
    --bf16 False
```

### 4.4 多机多卡训练

需在 `paddle.distributed.launch` 后增加参数 `--ips IP1,IP2,IP3,IP4`，分别对应多台机器的 IP，更多信息可参考 [飞桨官网-分布式训练](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/cluster_quick_start_collective_cn.html)。

## 5. 模型推理

请将下面的代码保存到 eval.py 中并运行。你可以选择直接加载训练好的模型权重完成推理，具体做法参考 5.1。如果你使用自定义数据完成了模型训练并保存了 checkpoint，你可以选择加载自行训练的模型参数进行推理，具体做法参考 5.2。

### 5.1 直接加载模型参数推理

未经完整训练，直接加载公开发布的模型参数进行推理。

```python
from ppdiffusers import StableDiffusionPipeline, UNet2DConditionModel
# 加载公开发布的 unet 权重
unet_model_name_or_path = "CompVis/stable-diffusion-v1-4/unet"
unet = UNet2DConditionModel.from_pretrained(unet_model_name_or_path)
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None, unet=unet)
prompt = "a photo of an astronaut riding a horse on mars"  # or a little girl dances in the cherry blossom rain
image = pipe(prompt, guidance_scale=7.5, width=512, height=512).images[0]
image.save("astronaut_rides_horse.png")
```


### 5.2 使用训练的模型参数进行推理

待模型训练完毕，会在 `output_dir` 保存训练好的模型权重，使用自行训练后生成的模型参数进行推理。

```python
from ppdiffusers import StableDiffusionPipeline, UNet2DConditionModel
# 加载上面我们训练好的 unet 权重
unet_model_name_or_path = "./laion400m_pretrain_output_trainer/checkpoint-5000/unet"
unet = UNet2DConditionModel.from_pretrained(unet_model_name_or_path)
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None, unet=unet)
prompt = "a photo of an astronaut riding a horse on mars"
# 当前训练的是256x256分辨率图片,因此请确保训练和推理参数最好一致
image = pipe(prompt, guidance_scale=7.5, width=256, height=256).images[0]
image.save("astronaut_rides_horse.png")
```

## 6. 参考资料
- https://github.com/CompVis/latent-diffusion
- https://github.com/CompVis/stable-diffusion

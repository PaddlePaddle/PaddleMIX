## Stable Diffusion Model 从零训练代码

本教程带领大家体验从零预训练 Stable Diffusion 模型，并完成预训练权重加载与推理。

## 1. 准备工作

### 1.1 安装 PaddleMIX 与依赖

通过 `git clone` 命令拉取 PaddleMIX 源码，并安装必要的依赖库。请确保你的 PaddlePaddle 框架版本在 2.5.0rc1 之后，PaddlePaddle 框架安装可参考 [飞桨官网-安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。
```bash
# paddlepaddle-gpu>=2.5.0rc1
git clone https://github.com/PaddlePaddle/PaddleMIX
cd PaddleMIX/ppdiffusers/examples/stable_diffusion
pip install -r requirements.txt  # 如果提示权限不够，请在最后增加 --user 选项
```

> 注：本模型训练与推理需要依赖 CUDA 11.2 及以上版本，如果本地机器不符合要求，建议前往 [AI Studio](https://aistudio.baidu.com/index) 进行模型训练、推理任务。

### 1.2 准备数据

预训练 Stable Diffusion 使用 laion400m 数据集，需要自行下载和处理，处理步骤详见 1.2.1。本教程为了方便大家 **体验跑通训练流程**，提供了一组假数据 laion400m_demo，可直接下载获取，详见 1.2.2。


#### 1.2.1 laion400m 数据集

下载好 laion400m 数据集之后，需要按照如下步骤进行数据准备：

1. 将数据集放置于 `data/laion400m/` 目录，其中里面的每个 part 的前三列为 `caption文本描述, 占位符空, base64编码的图片`，示例：`caption, _, img_b64 = vec[:3]`；
2. 准备 `filelist`，运行 [write_filelist.py](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/examples/stable_diffusion/data/filelist/write_filelist.py) ，生成完备的具有6万条数据路径的 `laion400m_en.filelist`（当前 `laion400m_en.filelist` 只存放了10条数据路径）。你也可以自定准备其他 `filelist`。

`laion400m_en.filelist` 为数据索引文件，内容如下所示：
```
/data/laion400m/part-00000.gz
/data/laion400m/part-00001.gz
/data/laion400m/part-00002.gz
/data/laion400m/part-00003.gz
/data/laion400m/part-00004.gz
/data/laion400m/part-00005.gz
/data/laion400m/part-00006.gz
/data/laion400m/part-00007.gz
/data/laion400m/part-00008.gz
/data/laion400m/part-00009.gz
```

#### 1.2.2 laion400m_demo 数据集（假数据，仅供验证跑通训练）

demo 数据可通过如下命令下载与解压：

```bash
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
└── laion400m_demo_data.tar.gz
```

`laion400m_en.filelist` 为数据索引文件（假数据），内容如下所示：
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

### 1.3 准备权重

Stable Diffusion 模型包含 3 个组成部分：vae、textencoder、unet，其中预训练仅需随机初始化 unet 部分，其余部分可直接加载预训练权重，本教程中提供的权重基于 sd1-4，将 unet 部分替换成了随机初始化的权重。

```sh
# 下载权重
wget https://bj.bcebos.com/paddlenlp/models/community/CompVis/CompVis-stable-diffusion-v1-4-paddle-init-pd.tar.gz
# 解压
tar -zxvf CompVis-stable-diffusion-v1-4-paddle-init-pd.tar.gz
```



## 2. 开启训练
### 2.1 硬件要求

示例脚本配置在显存 ≥40GB 的显卡上可正常训练，如显存不满足要求，可通过修改参数的方式运行脚本：
- 如果本地环境显存不够，请使用 AIStudio 上 32G 显存的 GPU 环境，并修改 --per_device_train_batch_size 为 32。
- bf16 仅支持 A100 硬件，无法使用 V100 进行训练，如果你的环境是 V100，需要修改 --bf16 为 False。

### 2.2 单机单卡训练

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
python -u train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ./laion400m_pretrain_output_trainer \
    --per_device_train_batch_size 64 \
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
    --dataloader_num_workers 8 \
    --pretrained_model_name_or_path ./CompVis-stable-diffusion-v1-4-paddle-init \
    --file_list ./data/filelist/train.filelist.list \
    --model_max_length 77 \
    --max_grad_norm -1 \
    --disable_tqdm True \
    --bf16 True  
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
* `--dataloader_num_workers`: Dataloader 所使用的 `num_workers` 参数。
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
* `--enable_xformers_memory_efficient_attention`: 是否开启 `xformers`，开启后训练速度会变慢，但是能够节省显存。注意我们需要安装 develop 版本的 paddlepaddle！
* `--only_save_updated_model`: 是否仅保存经过训练的权重，比如保存 `unet`、`ema 版 unet`、`text_encoder`，默认值为 `True`。


### 2.3 单机多卡训练
```bash
export FLAG_FUSED_LINEAR=0
export FLAGS_conv_workspace_size_limit=4096
# 是否开启 ema
export FLAG_USE_EMA=0
# 是否开启 recompute
export FLAG_RECOMPUTE=1
# 是否开启 xformers
export FLAG_XFORMERS=1
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ./laion400m_pretrain_output_trainer \
    --per_device_train_batch_size 64 \
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
    --dataloader_num_workers 8 \
    --pretrained_model_name_or_path ./CompVis-stable-diffusion-v1-4-paddle-init \
    --file_list ./data/filelist/train.filelist.list \
    --model_max_length 77 \
    --max_grad_norm -1 \
    --disable_tqdm True \
    --bf16 True
```

### 2.3 多机多卡训练

需在 `paddle.distributed.launch` 后增加参数 `--ips IP1,IP2,IP3,IP4`，分别对应多台机器的 IP，更多信息可参考 [飞桨官网-分布式训练](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/06_distributed_training/cluster_quick_start_collective_cn.html)。

## 3. 模型推理

请将下面的代码保存到 eval.py 中并运行。你可以选择直接加载训练好的模型权重完成推理，具体做法参考 3.1。如果你使用真实数据完成了模型训练并保存了 checkpoint，你可以选择加载自行训练的模型参数进行推理，具体做法参考 3.2。

### 3.1 直接加载模型参数推理

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


### 3.2 使用训练的模型参数进行推理

待模型训练完毕，会在 `output_dir` 保存训练好的模型权重，使用自行训练后生成的模型参数进行推理。

```python
from ppdiffusers import StableDiffusionPipeline, UNet2DConditionModel
# 加载上面我们训练好的 unet 权重
unet_model_name_or_path = "./output/checkpoint-5000/unet"
unet = UNet2DConditionModel.from_pretrained(unet_model_name_or_path)
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None, unet=unet)
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, guidance_scale=7.5, width=256, height=256).images[0]
image.save("astronaut_rides_horse.png")
```


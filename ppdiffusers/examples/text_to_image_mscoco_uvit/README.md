## U-ViT Latent Diffusion Model 从零训练代码

本教程带领大家如何开启基于U-ViT的**Latent Diffusion Model**的文生图训练。

___注意___:
___官方32层`CompVis/ldm-text2im-large-256`的Latent Diffusion Model使用的是vae，而不是vqvae！而Huggingface团队在设计目录结构的时候把文件夹名字错误的设置成了vqvae！为了与Huggingface团队保持一致，我们同样使用了vqvae文件夹命名！___
___Latent Diffusion Pipeline里默认是unet，此处我们将U-ViT训完的权重文件夹也命名为了unet。___


## 1 本地运行
### 1.1 安装依赖

在运行这个训练代码前，我们需要安装下面的训练依赖。
```bash
# paddlepaddle-gpu>=2.6.0
python -m pip install paddlepaddle-gpu==2.6.0 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install -r requirements.txt
```

安装einops，推荐使用python>=3.8
```bash
pip install git+https://github.com/arogozhnikov/einops.git
```


### 1.2 准备数据

#### MSCOCO文生图训练数据集的特征如下：
```
├── datasets  # 我们指定的输出文件路径
    ├──coco256_features
        ├── empty_context.npy
        ├── run_vis/
        ├── train/
        ├── val/
```

我们提供了下载链接：
- 请查看`datasets/download.txt`
- 特征抽取流程请参考[U-ViT](https://github.com/baofff/U-ViT/blob/main/scripts/extract_mscoco_feature.py)


### 1.3 使用trainner开启训练
#### 1.3.1 硬件要求
Tips：
- FP32 在 40GB 的显卡上可正常训练。

#### 1.3.2 单机多卡训练
```bash
TRAINING_MODEL_RESUME="None"
TRAINER_INSTANCES='127.0.0.1'
MASTER='127.0.0.1:8080'
TRAINERS_NUM=1 # nnodes, machine num
TRAINING_GPUS_PER_NODE=8 # nproc_per_node
DP_DEGREE=8 # dp_parallel_degree
MP_DEGREE=1 # tensor_parallel_degree
SHARDING_DEGREE=1 # sharding_parallel_degree

uvit_config_file=config/uvit_t2i_small.json
output_dir=output_trainer/uvit_t2i_small_trainer

feature_path=./datasets/coco256_features
per_device_train_batch_size=32
dataloader_num_workers=8
max_steps=1000000
save_steps=5000
warmup_steps=5000
logging_steps=50
image_logging_steps=-1
seed=1234

USE_AMP=True
fp16_opt_level="O1"
enable_tensorboard=True
recompute=True
enable_xformers=True

TRAINING_PYTHON="python -m paddle.distributed.launch --master ${MASTER} --nnodes ${TRAINERS_NUM} --nproc_per_node ${TRAINING_GPUS_PER_NODE} --ips ${TRAINER_INSTANCES}"
${TRAINING_PYTHON} train_txt2img_mscoco_uvit_trainer.py \
    --do_train \
    --feature_path ${feature_path} \
    --output_dir ${output_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps 1 \
    --learning_rate 0.0002 \
    --weight_decay 0.03 \
    --adam_beta1 0.9 \
    --adam_beta2 0.9 \
    --max_steps ${max_steps} \
    --lr_scheduler_type "constant_with_warmup" \
    --warmup_steps ${warmup_steps} \
    --image_logging_steps ${image_logging_steps} \
    --logging_steps ${logging_steps} \
    --save_steps ${save_steps} \
    --seed ${seed} \
    --dataloader_num_workers ${dataloader_num_workers} \
    --max_grad_norm -1 \
    --uvit_config_file ${uvit_config_file} \
    --num_inference_steps 50 \
    --model_max_length 77 \
    --use_ema True \
    --overwrite_output_dir True \
    --disable_tqdm True \
    --recompute ${recompute} \
    --fp16 ${USE_AMP} \
    --fp16_opt_level=${fp16_opt_level} \
    --enable_xformers_memory_efficient_attention ${enable_xformers} \
    --dp_degree ${DP_DEGREE} \
    --tensor_parallel_degree ${MP_DEGREE} \
    --sharding_parallel_degree ${SHARDING_DEGREE} \
    --pipeline_parallel_degree 1 \
```


`train_txt2img_mscoco_uvit_trainer.py`代码可传入的参数解释如下：
> * `--vae_name_or_path`: 预训练`vae`模型名称或地址，`CompVis/stable-diffusion-v1-4/vae`为`kl-8.ckpt`，程序将自动从BOS上下载预训练好的权重。
> * `--uvit_config_file`: `uvit`的config配置文件地址，默认为`./config/uvit_t2i_small.json`。
> * `--pretrained_model_name_or_path`: 加载预训练模型的名称或本地路径，如`CompVis/ldm-text2im-large-256`，`pretrained_model_name_or_path`的优先级高于`vae_name_or_path`, `text_encoder_config_file`和`unet_config_file`。
> * `--per_device_train_batch_size`: 训练时每张显卡所使用的`batch_size批量`，当我们的显存较小的时候，需要将这个值设置的小一点。
> * `--gradient_accumulation_steps`: 梯度累积的步数，用户可以指定梯度累积的步数，在梯度累积的step中。减少多卡之间梯度的通信，减少更新的次数，扩大训练的batch_size。
> * `--learning_rate`: 学习率。
> * `--weight_decay`: AdamW优化器的`weight_decay`。
> * `--max_steps`: 最大的训练步数。
> * `--save_steps`: 每间隔多少步`（global step步数）`，保存模型。
> * `--save_total_limit`: 最多保存多少个模型。
> * `--lr_scheduler_type`: 要使用的学习率调度策略。默认为 `constant`。
> * `--warmup_steps`: 用于从 0 到 `learning_rate` 的线性 warmup 的步数。
> * `--image_logging_steps`: 每隔多少步，log训练过程中的图片，默认为`1000`步，注意`image_logging_steps`需要是`logging_steps`的整数倍。
> * `--logging_steps`: logging日志的步数，默认为`50`步。
> * `--output_dir`: 模型保存路径。
> * `--seed`: 随机种子，为了可以复现训练结果，Tips：当前paddle设置该随机种子后仍无法完美复现。
> * `--dataloader_num_workers`: Dataloader所使用的`num_workers`参数。
> * `--num_inference_steps`: 推理预测时候使用的步数。
> * `--prediction_type`: 预测类型，可从`["epsilon", "v_prediction"]`选择。
> * `--use_ema`: 是否对`unet`使用`ema`，默认为`False`。
> * `--max_grad_norm`: 梯度剪裁的最大norm值，`-1`表示不使用梯度裁剪策略。
> * `--recompute`: 是否开启重计算，(`bool`, 可选, 默认为 `False`)，在开启后我们可以增大batch_size，注意在小batch_size的条件下，开启recompute后显存变化不明显，只有当开大batch_size后才能明显感受到区别。
> * `--fp16`: 是否使用 fp16 混合精度训练而不是 fp32 训练。(`bool`, 可选, 默认为 `False`)
> * `--fp16_opt_level`: 混合精度训练模式，可为``O1``或``O2``模式，默认``O1``模式，默认O1. 只在fp16选项开启时候生效。
> * `--enable_xformers_memory_efficient_attention`: 是否开启`xformers`，开启后训练速度会变慢，但是能够节省显存。注意我们需要安装develop版本的paddlepaddle！


## 2 模型推理

```
import paddle
from paddlenlp.trainer import set_seed

from ppdiffusers import DPMSolverMultistepScheduler
from ppdiffusers.pipelines import LDMTextToImageUViTPipeline

dtype = paddle.float32
pipe = LDMTextToImageUViTPipeline.from_pretrained("baofff/ldm-uvit_t2i-small-256-mscoco", paddle_dtype=dtype)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
set_seed(42)

prompt = "People are at a stop light on a snowy street."
image = pipe(prompt, guidance_scale=7.5, num_inference_steps=50).images[0]

image.save("result.png")
```

实际训练完成的权重，需根据代码`scripts/extract_weights.py`抽取其中的uvit部分的权重，然后替换`baofff/ldm-uvit_t2i-small-256-mscoco/unet/`路径下的权重文件，然后运行上面的推理代码。


## 引用
```
@inproceedings{bao2022all,
  title={All are Worth Words: A ViT Backbone for Diffusion Models},
  author={Bao, Fan and Nie, Shen and Xue, Kaiwen and Cao, Yue and Li, Chongxuan and Su, Hang and Zhu, Jun},
  booktitle = {CVPR},
  year={2023}
}
```

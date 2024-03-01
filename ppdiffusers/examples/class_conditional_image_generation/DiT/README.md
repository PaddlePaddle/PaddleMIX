## Scalable Diffusion Models with Transformers (DiT)
## Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers (SiT)


## 1 本地运行
### 1.1 安装依赖

在运行这个训练代码前，我们需要安装下面的训练依赖。
```bash
# paddlepaddle-gpu>=2.6.0
python -m pip install paddlepaddle-gpu==2.6.0 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install -r requirements.txt
```

### 1.2 准备数据

#### ImageNet训练数据集的特征和标签如下：
```
├── data  # 我们指定的输出文件路径
    ├──fastdit_imagenet256
        ├── imagenet256_features
        ├── imagenet256_labels
```

我们提供了下载链接：
- `wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/fastdit_features/fastdit_imagenet256.tar`；
- 特征抽取流程请参考[fast-DiT](https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py)


### 1.3 使用trainner开启训练

#### 1.3.1 硬件要求
Tips：
- FP32 在默认总batch_size=256情况下需占42GB显存。
- FP16 在默认总batch_size=256情况下需占21GB显存。

#### 1.3.2 单机多卡训练
```bash
config_file=config/DiT_XL_patch2.json
OUTPUT_DIR=./output/DiT_XL_patch2_trainer

# config_file=config/SiT_XL_patch2.json
# OUTPUT_DIR=./output/SiT_XL_patch2_trainer

feature_path=./data/fastdit_imagenet256
batch_size=32 # per gpu
num_workers=8
max_steps=7000000
logging_steps=50
seed=0

USE_AMP=True
FP16_OPT_LEVEL="O1"
enable_tensorboard=True
recompute=True
enable_xformers=True

python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_image_generation_trainer.py \
    --do_train \
    --feature_path ${feature_path} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --max_steps ${max_steps} \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 1000 \
    --logging_dir ${OUTPUT_DIR}/tb_log \
    --logging_steps ${logging_steps} \
    --save_steps 10000 \
    --save_total_limit 50 \
    --dataloader_num_workers ${num_workers} \
    --vae_name_or_path stabilityai/sd-vae-ft-mse \
    --config_file ${config_file} \
    --num_inference_steps 25 \
    --use_ema True \
    --max_grad_norm -1 \
    --overwrite_output_dir True \
    --disable_tqdm True \
    --fp16_opt_level ${FP16_OPT_LEVEL} \
    --seed ${seed} \
    --recompute ${recompute} \
    --enable_xformers_memory_efficient_attention ${enable_xformers} \
    --bf16 ${USE_AMP}
```

### 1.4 自定义训练逻辑开启训练

#### 1.4.1 单机多卡训练
```bash
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" \
    train_image_generation_notrainer.py \
    --config_file config/DiT_XL_patch2.json \
    --feature_path ./data/fastdit_imagenet256 \
    --global_batch_size 256
```


## 2 模型推理

待模型训练完毕，会在`output_dir`保存训练好的模型权重。注意DiT模型推理可以使用ppdiffusers中的DiTPipeline，但是SiT模型推理暂时不支持生成`Pipeline`。
可以参照运行`python infer_demo_dit.py`或者`python infer_demo_dit.py`。

DiT可以使用`tools/convert_dit_to_ppdiffusers.py`生成推理所使用的`Pipeline`。

```bash
python tools/convert_dit_to_ppdiffusers.py
```

输出的模型目录结构如下：
```shell
├── DiT_XL_2_256  # 我们指定的输出文件路径
    ├── model_index.json
    ├── scheduler
    │   └── scheduler_config.json
    ├── transformer
    │   ├── config.json
    │   └── model_state.pdparams
    └── vae
        ├── config.json
        └── model_state.pdparams
```
注意生成后的`model_index.json`里需要有`"id2label"`的1000类的id和标签对应字典，如果没有则需要手动复制`tools/ImageNet_id2label.json`里的加进去。


在生成`Pipeline`的权重后，我们可以使用如下的代码进行推理。

```python
from ppdiffusers import DiTPipeline, DPMSolverMultistepScheduler, DDIMScheduler
import paddle
from paddlenlp.trainer import set_seed
dtype=paddle.float32
pipe=DiTPipeline.from_pretrained("./DiT_XL_2_256", paddle_dtype=dtype)
#pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

words = ["white shark"]
class_ids = pipe.get_label_ids(words)

set_seed(42)
generator = paddle.Generator().manual_seed(0)
image = pipe(class_labels=class_ids, num_inference_steps=25, generator=generator).images[0]
image.save("white_shark.png")
print(f'\nGPU memory usage: {paddle.device.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB')
```


## 引用
```
@article{Peebles2022DiT,
  title={Scalable Diffusion Models with Transformers},
  author={William Peebles and Saining Xie},
  year={2022},
  journal={arXiv preprint arXiv:2212.09748},
}
```

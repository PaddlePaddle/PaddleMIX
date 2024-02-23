## Scalable Diffusion Models with Transformers (DiT)

## 1 本地运行
### 1.1 安装依赖

在运行这个训练代码前，我们需要安装下面的训练依赖。
```bash
# paddlepaddle-gpu>=2.6.0
python -m pip install paddlepaddle-gpu==2.6.0 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install -r requirements.txt
```

### 1.2 准备数据

#### ImageNet训练数据集如下
```
├── data  # 我们指定的输出文件路径
    ├──fastdit_imagenet256
        ├── imagenet256_features
        ├── imagenet256_labels
```

我们提供了下载链接：
- `wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/fastdit_features/fastdit_imagenet256.tar`；


### 1.3 使用trainner开启训练
#### 1.3.1 硬件要求
Tips：
- FP32 在 40GB 的显卡上可正常训练。

#### 1.3.1 单机多卡训练
```bash
python -u train_image_generation_dit_trainer.py \
    --do_train \
    --output_dir ./image_generation_dit_output_trainer \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --max_steps 1000000000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 1000 \
    --logging_steps 50 \
    --save_steps 5000 \
    --save_total_limit 50 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --vae_name_or_path stabilityai/sd-vae-ft-mse \
    --num_inference_steps 25 \
    --max_grad_norm -1
```

### 1.4 自定义训练逻辑开启训练

#### 1.4.1 单机多卡训练
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m paddle.distributed.launch \
    --nnodes=1 --nproc_per_node=8 --use_env \
    train.py \
    --model DiT-XL/2 \
    --feature-path ./data/fastdit_imagenet256 \
    --global-batch-size 16
```


## 2 模型推理

待模型训练完毕，会在`output_dir`保存训练好的模型权重，我们可以使用`scripts/convert_dit_to_ppdiffusers.py`生成推理所使用的`Pipeline`。

```bash
python scripts/convert_dit_to_ppdiffusers.py
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

在生成`Pipeline`的权重后，我们可以使用如下的代码进行推理。

```python
from ppdiffusers import DiTPipeline, DPMSolverMultistepScheduler, DDIMScheduler
import paddle
from paddlenlp.trainer import set_seed

dtype=paddle.float16
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

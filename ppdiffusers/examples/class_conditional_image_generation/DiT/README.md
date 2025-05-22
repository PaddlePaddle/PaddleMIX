# Class-Conditional Image Generation

## Scalable Diffusion Models with Transformers (DiT)
## Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers (SiT)


## 1 本地运行
### 1.1 安装依赖

在运行这个训练代码前，我们需要安装下面的训练依赖。
```bash
# paddlepaddle-gpu>=2.6.0
python -m pip install paddlepaddle-gpu==2.6.0 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install -r requirements.txt

git clone https://github.com/PaddlePaddle/PaddleNLP.git -b release/2.6
cd PaddleNLP/model_zoo/gpt-3/external_ops/
python setup.py install
cd ../../../../
```

训练Large-DiT(DiT-LLaMA)模型需安装`fused_ln`，需要安装[此目录](https://github.com/PaddlePaddle/PaddleNLP/tree/release/2.6/model_zoo/gpt-3/external_ops)下的自定义OP， `python setup.py install`。


### 1.2 准备数据

#### ImageNet训练数据集的特征和标签如下：
```
├── data  # 我们指定的输出文件路径
    ├──fastdit_imagenet256
        ├── imagenet256_features
        ├── imagenet256_labels
```

我们提供了下载链接：

```bash
cd data
sh download_data.sh
cd ..
```

- `wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/fastdit_features/fastdit_imagenet256.tar`；
- 特征抽取流程请参考[fast-DiT](https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py)


### 1.3 使用trainner开启训练

#### 1.3.1 硬件要求
Tips：
- FP32 在默认总batch_size=256情况下需占 42GB 显存每卡。
- FP16 在默认总batch_size=256情况下需占 21GB 显存每卡。

#### 1.3.2 单机多卡训练

可以直接运行`sh 0_run_train_dit_trainer.sh`，或者

```bash
TRAINING_MODEL_RESUME="None"
TRAINER_INSTANCES='127.0.0.1'
MASTER='127.0.0.1:8080'
TRAINERS_NUM=1 # nnodes, machine num
TRAINING_GPUS_PER_NODE=8 # nproc_per_node
DP_DEGREE=1 # dp_parallel_degree
MP_DEGREE=1 # tensor_parallel_degree
SHARDING_DEGREE=8 # sharding_parallel_degree

# real dp_parallel_degree = nnodes * nproc_per_node / tensor_parallel_degree / sharding_parallel_degree
# Please make sure: nnodes * nproc_per_node >= tensor_parallel_degree * sharding_parallel_degree

config_file=config/DiT_XL_patch2.json
OUTPUT_DIR=./output_trainer/DiT_XL_patch2_trainer
feature_path=./data/fastdit_imagenet256

per_device_train_batch_size=32
gradient_accumulation_steps=1

num_workers=8
max_steps=7000000
logging_steps=20
save_steps=5000
image_logging_steps=-1
seed=0

max_grad_norm=-1

USE_AMP=True
FP16_OPT_LEVEL="O2"

enable_tensorboard=True
recompute=True
enable_xformers=True

transformer_engine_backend=False
use_fp8=False # This option takes effect only when transformer_engine_backend=True

TRAINING_PYTHON="python -m paddle.distributed.launch --master ${MASTER} --nnodes ${TRAINERS_NUM} --nproc_per_node ${TRAINING_GPUS_PER_NODE} --ips ${TRAINER_INSTANCES}"
${TRAINING_PYTHON} train_image_generation_trainer.py \
    --do_train \
    --feature_path ${feature_path} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --max_steps ${max_steps} \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps ${image_logging_steps} \
    --logging_dir ${OUTPUT_DIR}/tb_log \
    --logging_steps ${logging_steps} \
    --save_steps ${save_steps} \
    --save_total_limit 50 \
    --dataloader_num_workers ${num_workers} \
    --vae_name_or_path stabilityai/sd-vae-ft-mse \
    --config_file ${config_file} \
    --num_inference_steps 25 \
    --use_ema True \
    --max_grad_norm ${max_grad_norm} \
    --overwrite_output_dir True \
    --disable_tqdm True \
    --fp16_opt_level ${FP16_OPT_LEVEL} \
    --seed ${seed} \
    --recompute ${recompute} \
    --enable_xformers_memory_efficient_attention ${enable_xformers} \
    --bf16 ${USE_AMP} \
    --dp_degree ${DP_DEGREE} \
    --tensor_parallel_degree ${MP_DEGREE} \
    --sharding_parallel_degree ${SHARDING_DEGREE} \
    --sharding "stage1" \
    --hybrid_parallel_topo_order "sharding_first" \
    --amp_master_grad 1 \
    --pipeline_parallel_degree 1 \
    --sep_parallel_degree 1 \
    --transformer_engine_backend ${transformer_engine_backend} \
    --use_fp8 ${use_fp8}
```

### 1.4 自定义训练逻辑开启训练

#### 1.4.1 单机多卡训练

注意显存约占 21GB 每卡。

可以直接运行`sh 1_run_train_dit_notrainer.sh`，或者

```bash
config_file=config/DiT_XL_patch2.json
results_dir=./output_notrainer/DiT_XL_patch2_notrainer

feature_path=./data/fastdit_imagenet256

image_size=256
global_batch_size=256
num_workers=8
epochs=1400
logging_steps=50
save_steps=5000

global_seed=0

python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" \
    train_image_generation_notrainer.py \
    --image_size ${image_size} \
    --config_file ${config_file} \
    --feature_path ${feature_path} \
    --results_dir ${results_dir} \
    --epochs ${epochs} \
    --global_seed ${global_seed} \
    --global_batch_size ${global_batch_size} \
    --num_workers ${num_workers} \
    --log_every ${logging_steps} \
    --ckpt_every ${save_steps} \
```

### 1.5 使用分布式自动并行进行训练

#### 1.5.1 硬件要求
同1.3.1，与之前手动并行方式显存占用差别不大

#### 1.5.2 单机多卡训练Dit
可以直接运行`sh 0_run_train_dit_trainer_auto.sh`，或者

```bash
FLAGS_enable_pir_api=true

TRAINING_MODEL_RESUME="None"
TRAINER_INSTANCES='127.0.0.1'
MASTER='127.0.0.1:8080'
TRAINERS_NUM=1 # nnodes, machine num
TRAINING_GPUS_PER_NODE=8 # nproc_per_node
DP_DEGREE=1 # data_parallel_degree
MP_DEGREE=1 # tensor_parallel_degree
PP_DEGREE=1 # pipeline_parallel_degree
SHARDING_DEGREE=8 # sharding_parallel_degree

# real dp_parallel_degree = nnodes * nproc_per_node / tensor_parallel_degree / sharding_parallel_degree
# Please make sure: nnodes * nproc_per_node >= tensor_parallel_degree * sharding_parallel_degree

config_file=config/DiT_XL_patch2.json
OUTPUT_DIR=./output_trainer/DiT_XL_patch2_auto_trainer
feature_path=./data/fastdit_imagenet256

per_device_train_batch_size=32
gradient_accumulation_steps=1

resolution=256
num_workers=8
max_steps=7000000
logging_steps=1
save_steps=5000
image_logging_steps=-1
seed=0

max_grad_norm=-1

USE_AMP=True
FP16_OPT_LEVEL="O2"

enable_tensorboard=True
recompute=True
enable_xformers=True
to_static=0  # whether we use dynamic to static training

TRAINING_PYTHON="python -m paddle.distributed.launch --master ${MASTER} --nnodes ${TRAINERS_NUM} --nproc_per_node ${TRAINING_GPUS_PER_NODE} --ips ${TRAINER_INSTANCES}"
${TRAINING_PYTHON} train_image_generation_trainer_auto.py \
    --do_train \
    --feature_path ${feature_path} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --resolution ${resolution} \
    --max_steps ${max_steps} \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps ${image_logging_steps} \
    --logging_dir ${OUTPUT_DIR}/tb_log \
    --logging_steps ${logging_steps} \
    --save_steps ${save_steps} \
    --save_total_limit 50 \
    --dataloader_num_workers ${num_workers} \
    --vae_name_or_path stabilityai/sd-vae-ft-mse \
    --config_file ${config_file} \
    --num_inference_steps 25 \
    --use_ema True \
    --max_grad_norm ${max_grad_norm} \
    --overwrite_output_dir True \
    --disable_tqdm True \
    --fp16_opt_level ${FP16_OPT_LEVEL} \
    --seed ${seed} \
    --recompute ${recompute} \
    --enable_xformers_memory_efficient_attention ${enable_xformers} \
    --bf16 ${USE_AMP} \
    --amp_master_grad 1 \
    --dp_degree ${DP_DEGREE} \
    --tensor_parallel_degree ${MP_DEGREE} \
    --pipeline_parallel_degree ${PP_DEGREE} \
    --sharding_parallel_degree ${SHARDING_DEGREE} \
    --sharding "stage1" \
    --sharding_parallel_config "enable_stage1_overlap enable_stage1_tensor_fusion" \
    --hybrid_parallel_topo_order "sharding_first" \
    --sep_parallel_degree 1 \
    --enable_auto_parallel 1 \
    --to_static $to_static
```

#### 1.5.3 单机多卡训练LargeDit
可以直接运行`sh 4_run_train_largedit_3b_trainer_auto.sh`，或者

```bash
FLAGS_enable_pir_api=true

TRAINING_MODEL_RESUME="None"
TRAINER_INSTANCES='127.0.0.1'
MASTER='127.0.0.1:8080'
TRAINERS_NUM=1 # nnodes, machine num
TRAINING_GPUS_PER_NODE=8 # nproc_per_node
DP_DEGREE=1 # data_parallel_degree
MP_DEGREE=4 # tensor_parallel_degree
PP_DEGREE=1 # pipeline_parallel_degree
SHARDING_DEGREE=2 # sharding_parallel_degree

# real dp_parallel_degree = nnodes * nproc_per_node / tensor_parallel_degree / sharding_parallel_degree
# Please make sure: nnodes * nproc_per_node >= tensor_parallel_degree * sharding_parallel_degree

config_file=config/LargeDiT_3B_patch2.json
OUTPUT_DIR=./output_trainer/LargeDiT_3B_patch2_auto_trainer
feature_path=./data/fastdit_imagenet256

per_device_train_batch_size=32
gradient_accumulation_steps=1

resolution=256
num_workers=8
max_steps=7000000
logging_steps=1
save_steps=5000
image_logging_steps=-1
seed=0

max_grad_norm=2.0

USE_AMP=True
FP16_OPT_LEVEL="O2"

enable_tensorboard=True
recompute=True
enable_xformers=True
to_static=0  # whether we use dynamic to static training

TRAINING_PYTHON="python -m paddle.distributed.launch --master ${MASTER} --nnodes ${TRAINERS_NUM} --nproc_per_node ${TRAINING_GPUS_PER_NODE} --ips ${TRAINER_INSTANCES}"
${TRAINING_PYTHON} train_image_generation_trainer_auto.py \
    --do_train \
    --feature_path ${feature_path} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate 1e-4 \
    --resolution ${resolution} \
    --weight_decay 0.0 \
    --max_steps ${max_steps} \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps ${image_logging_steps} \
    --logging_dir ${OUTPUT_DIR}/tb_log \
    --logging_steps ${logging_steps} \
    --save_steps ${save_steps} \
    --save_total_limit 50 \
    --dataloader_num_workers ${num_workers} \
    --vae_name_or_path stabilityai/sd-vae-ft-mse \
    --config_file ${config_file} \
    --num_inference_steps 25 \
    --use_ema True \
    --max_grad_norm ${max_grad_norm} \
    --overwrite_output_dir True \
    --disable_tqdm True \
    --fp16_opt_level ${FP16_OPT_LEVEL} \
    --seed ${seed} \
    --recompute ${recompute} \
    --enable_xformers_memory_efficient_attention ${enable_xformers} \
    --bf16 ${USE_AMP} \
    --amp_master_grad 1 \
    --dp_degree ${DP_DEGREE} \
    --tensor_parallel_degree ${MP_DEGREE} \
    --pipeline_parallel_degree ${PP_DEGREE} \
    --sharding_parallel_degree ${SHARDING_DEGREE} \
    --sharding "stage1" \
    --sharding_parallel_config "enable_stage1_overlap enable_stage1_tensor_fusion" \
    --hybrid_parallel_topo_order "sharding_first" \
    --sep_parallel_degree 1 \
    --enable_auto_parallel 1 \
    --to_static $to_static
```

## 2 模型推理

### 2.1 使用公开权重推理

可以直接运行`python infer_demo_dit.py`、`python infer_demo_sit.py`、`python infer_demo_largedit_3b.py`或者`python infer_demo_largedit_7b.py`

### 2.2 使用训练完的权重推理

待模型训练完毕，会在`output_dir`保存训练好的模型权重。注意DiT模型推理可以使用ppdiffusers中的DiTPipeline，**但是SiT模型推理暂时不支持生成`Pipeline`**。

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
import paddle
from paddlenlp.trainer import set_seed

from ppdiffusers import DDIMScheduler, DiTPipeline

dtype = paddle.float32
pipe = DiTPipeline.from_pretrained("./DiT_XL_2_256", paddle_dtype=dtype)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

words = ["golden retriever"]  # class_ids [207]
class_ids = pipe.get_label_ids(words)

set_seed(42)
generator = paddle.Generator().manual_seed(0)
image = pipe(class_labels=class_ids, num_inference_steps=25, generator=generator).images[0]
image.save("result_DiT_golden_retriever.png")
```

### 2.3 Paddle Inference 高性能推理

- Paddle Inference提供DIT模型高性能推理实现，推理性能提升80%+
环境准备：
```shell
# 安装develop版本的paddle
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu123/

# 安装 triton并适配paddle
python -m pip install triton
python -m pip install git+https://github.com/zhoutianzi666/UseTritonInPaddle.git
python -c "import use_triton_in_paddle; use_triton_in_paddle.make_triton_compatible_with_paddle()"
````

一键推理指令：
```shell
python ppdiffusers/examples/inference/class_conditional_image_generation-dit.py --inference_optimize 1
```

- 在 NVIDIA A100-SXM4-40GB 上测试的性能如下：

| Paddle Inference| TensorRT-LLM |  Paddle动态图 |
| --------------- | ------------ | ------------ |
|      219 ms     |    242 ms    |    1200 ms   |




## 引用
```
@article{Peebles2022DiT,
  title={Scalable Diffusion Models with Transformers},
  author={William Peebles and Saining Xie},
  year={2022},
  journal={arXiv preprint arXiv:2212.09748},
}
```

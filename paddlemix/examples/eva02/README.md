# EVA-02

## 1. 模型简介

[EVA-02: A Visual Representation for Neon Genesis](https://arxiv.org/abs/2303.11331), Paddle实现版本。

### MIM pre-trained EVA-02

<div align="center">

| model name | #params | MIM pretrain dataset | MIM pretrain epochs | weight |
|------------|:-------:|:--------------:|:-------------:|:------:|
| `eva02_Ti_pt_in21k_p14` | 6M | IN-21K | 240 | [link](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_p14/model_state.pdparams) |
| `eva02_S_pt_in21k_p14` | 22M | IN-21K | 240 | [link](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_S_pt_in21k_p14/model_state.pdparams) |
| `eva02_B_pt_in21k_p14` | 86M | IN-21K | 150 | [link](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_B_pt_in21k_p14/model_state.pdparams) |
| `eva02_B_pt_in21k_p14to16` | 86M | IN-21K | 150 | [link](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_B_pt_in21k_p14to16/model_state.pdparams) |
| `eva02_L_pt_in21k_p14` | 304M | IN-21K | 150 | [link](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_L_pt_in21k_p14/model_state.pdparams) |
| `eva02_L_pt_in21k_p14to16` | 304M | IN-21K | 150 | [link](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_L_pt_in21k_p14to16/model_state.pdparams) |
| `eva02_L_pt_m38m_p14` | 304M | Merged-38M | 56 | [link](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_L_pt_m38m_p14/model_state.pdparams) |
| `eva02_L_pt_m38m_p14to16` | 304M | Merged-38M | 56 | [link](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_L_pt_m38m_p14to16/model_state.pdparams) |

</div>

- MIM pre-trained 的 EVA-02 模型的输入尺度是`224x224`，patch尺度是`14x14`。
- `eva02_psz14to16` 表示对 `patch_embed` 的卷积核大小从 `14x14` 插值到到 `16x16`, 并将 `pos_embed` 从 `16x16` 插值到 `14x14`。这对于物体检测、实例分割和语义分割任务非常有用。



### IN-21K intermediate fine-tuned EVA-02

<div align="center">

| model name | init.ckpt | IN-21K finetune epochs | weight |
|------------|------------|:----------------:|:------:|
| `eva02_B_pt_in21k_medft_in21k_p14` | `eva02_B_pt_in21k_p14` | 40 | [link](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_B_pt_in21k_medft_in21k_p14/model_state.pdparams) |
| `eva02_L_pt_in21k_medft_in21k_p14` | `eva02_L_pt_in21k_p14` | 20 | [link](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_L_pt_in21k_medft_in21k_p14/model_state.pdparams) |
| `eva02_L_pt_m38m_medft_in21k_p14`  | `eva02_L_pt_m38m_p14`  | 30 | [link](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_L_pt_m38m_medft_in21k_p14/model_state.pdparams) |

</div>

- IN-21K intermediate fine-tuned 的 EVA-02 模型的输入尺度是`448x448`，patch尺度是`14x14`。



### IN-1K fine-tuned EVA-02 (*w/o* IN-21K intermediate fine-tuning)

<div align="center">

| model name | init.ckpt | IN-1K finetune epochs | finetune image size | ema？| top-1 | weight |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `eva02_Ti_pt_in21k_ft_in1k_p14` | `eva02_Ti_pt_in21k_p14` | 100 | ``336x336`` | `x` | 80.7 |[link](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_ft_in1k_p14/model_state.pdparams) |
| `eva02_S_pt_in21k_ft_in1k_p14` | `eva02_S_pt_in21k_p14` | 100 | ``336x336`` | `x` | 85.8 | [link](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_S_pt_in21k_ft_in1k_p14/model_state.pdparams) |
| `eva02_B_pt_in21k_ft_in1k_p14` | `eva02_B_pt_in21k_p14` | 30 | ``448x448`` | `x` | 88.3 | [link](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_B_pt_in21k_ft_in1k_p14/model_state.pdparams) |
| `eva02_L_pt_in21k_ft_in1k_p14` | `eva02_L_pt_in21k_p14` | 30 | ``448x448`` | `o`| 89.6 | [link](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_L_pt_in21k_ft_in1k_p14/model_state.pdparams) |
| `eva02_L_pt_m38m_ft_in1k_p14` | `eva02_L_pt_m38m_p14` | 30 | `448x448` | `o` | 89.6 | [link](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_L_pt_m38m_ft_in1k_p14/model_state.pdparams) |

</div>

- `o`: 表示使用的是ema模型权重。



### IN-1K fine-tuned EVA-02 (*w/* IN-21K intermediate fine-tuning)

<div align="center">

| model name | init.ckpt | IN-1K finetune epochs | finetune image size | ema？| top-1 | weight |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `eva02_B_pt_in21k_medft_in21k_ft_in1k_p14` | `eva02_B_pt_in21k_medft_in21k_p14` | 15 | ``448x448`` | `o` | 88.6 | [link](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_B_pt_in21k_medft_in21k_ft_in1k_p14/model_state.pdparams) |
| `eva02_L_pt_in21k_medft_in21k_ft_in1k_p14` | `eva02_L_pt_in21k_medft_in21k_p14` | 20 | ``448x448`` | `o` | 89.9 | [link](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_L_pt_in21k_medft_in21k_ft_in1k_p14/model_state.pdparams) |
| `eva02_L_pt_m38m_medft_in21k_ft_in1k_p14` | `eva02_L_pt_m38m_medft_in21k_p14` | 20 | `448x448` | `o` | 90.0 | [link](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_L_pt_m38m_medft_in21k_ft_in1k_p14/model_state.pdparams) |

</div>

- `o`: 表示使用的是ema模型权重。



## 2. 环境准备

1） 安装PaddleNLP develop版本

```
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

2）安装环境依赖包

```
pip install -r requirements.txt
```

3）安装`FusedLayerNorm`，在`paddlemix/external_ops/`目录下，安装fusedln包。

```
# 安装fusedln到python环境
python setup.py install --prefix=$INSTALL_DIR
# 添加安装路径到系统环境路径中
export PATH=$PATH:$INSTALL_DIR
```


## 3. 数据集和预训练权重

1) ImageNet-1k数据

我们使用标准的ImageNet-1K数据集（ILSVRC 2012，1000类的120万张图像），从 http://image-net.org 下载，然后使用[shell script](https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh) 将训练和验证图像移动并提取到标记的子文件夹中。注意其train和val文件夹里均需为1000个子文件夹即1000类。


## 4. 使用说明

### 4.1 Pretrain预训练

使用`paddlemix/examples/eva02/run_eva02_pretrain_dist.py`。

注意：

1. 如果采用分布式策略，分布式并行关系有：`nnodes * nproc_per_node == tensor_parallel_degree * sharding_parallel_degree * dp_parallel_degree`，其中`dp_parallel_degree`参数根据其他几个值计算出来，因此需要保证`nnodes * nproc_per_node >= tensor_parallel_degree * sharding_parallel_degree`；
2. 设定具体的 `model_name` 可以直接使用创建模型，但如果更换teacher，则需自己改写`paddlemix/EVA/EVA02/eva02_Ti_for_pretrain`中config.json and model_config.json的`teacher_config`这个字段的内容，比如将默认的 `paddlemix/EVA/EVA01-CLIP-g-14` 改为 `paddlemix/EVA/EVA02-CLIP-bigE-14`。而`student_config`这个字段是dict，student模型本身是train from scratch的；
3. 如果设定 `model_name=None`，也可以通过设定具体的 `teacher_name` 和 `student_name` 来创建模型，**但它们必须都各自具有config.json和model_state.pdparams**，一般**eval评估或加载全量权重debug时**，常采用 `model_name=None` 的形式；
4. `TEA_PRETRAIN_CKPT`通常情况下设置为None，模型训练前会自动加载来自`teacher_name`中的对应teacher预训练权重。但是**如果设置 MP_DEGREE > 1**时，则必须再次设置`TEA_PRETRAIN_CKPT`的路径去加载，一般设置为其绝对路径，也可从对应的下载链接单独下载相应的`model_state.pdparams`并放置；


训练命令及参数配置示例，这里示例采用单机8卡程序：
```shell
export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export NVIDIA_TF32_OVERRIDE=0
export NCCL_ALGO=Tree

optim="adamw"
lr=3e-3
warmup_lr=1e-6
min_lr=1e-5
weight_decay=0.05
CLIP_GRAD=3.0

num_train_epochs=240
save_epochs=5 # save every 5 epochs

warmup_epochs=1
drop_path=0.0

TRAINING_MODEL_RESUME="None"
TRAINER_INSTANCES='127.0.0.1'
MASTER='127.0.0.1:8080'

TRAINERS_NUM=1 # nnodes, machine num
TRAINING_GPUS_PER_NODE=8 # nproc_per_node
DP_DEGREE=8 # dp_parallel_degree
MP_DEGREE=1 # tensor_parallel_degree
SHARDING_DEGREE=1 # sharding_parallel_degree

model_name="paddlemix/EVA/EVA02/eva02_Ti_for_pretrain"
# model_name=None # if set None, will use teacher_name and student_name from_pretrained, both should have config and pdparams
teacher_name="paddlemix/EVA/EVA01-CLIP-g-14"
student_name="paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_p14"

TEA_PRETRAIN_CKPT=None # /root/.paddlenlp/models/paddlemix/EVA/EVA01-CLIP-g-14/model_state.pdparams # must add if MP_DEGREE > 1
STU_PRETRAIN_CKPT=None

OUTPUT_DIR=./output/eva02_Ti_pt_in21k_p14

DATA_PATH=./dataset/ILSVRC2012 # put your ImageNet-1k val data path
input_size=224
num_mask_patches=105 ### 224*224/14/14 * 0.4
batch_size=10 # 100(bsz_per_gpu)*8(#gpus_per_node)*5(#nodes)*1(update_freq)=4000(total_bsz)
num_workers=10
accum_freq=1 # update_freq
logging_steps=10 # print_freq
seed=0

USE_AMP=False
FP16_OPT_LEVEL="O1"
enable_tensorboard=True

TRAINING_PYTHON="python -m paddle.distributed.launch --master ${MASTER} --nnodes ${TRAINERS_NUM} --nproc_per_node ${TRAINING_GPUS_PER_NODE} --ips ${TRAINER_INSTANCES}"
${TRAINING_PYTHON} run_eva02_pretrain_dist.py \
        --do_train \
        --data_path ${DATA_PATH}/train \
        --model ${model_name} \
        --teacher ${teacher_name} \
        --student ${student_name} \
        --input_size ${input_size} \
        --drop_path ${drop_path} \
        --optim ${optim} \
        --learning_rate ${lr} \
        --weight_decay ${weight_decay} \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --adam_epsilon 1e-6 \
        --max_grad_norm ${CLIP_GRAD} \
        --lr_scheduler_type cosine \
        --warmup_lr ${warmup_lr} \
        --min_lr ${min_lr} \
        --num_train_epochs ${num_train_epochs} \
        --save_epochs ${save_epochs} \
        --warmup_epochs ${warmup_epochs} \
        --per_device_train_batch_size ${batch_size} \
        --dataloader_num_workers ${num_workers} \
        --output_dir ${OUTPUT_DIR} \
        --logging_dir ${OUTPUT_DIR}/tb_log \
        --logging_steps ${logging_steps} \
        --accum_freq ${accum_freq} \
        --dp_degree ${DP_DEGREE} \
        --tensor_parallel_degree ${MP_DEGREE} \
        --sharding_parallel_degree ${SHARDING_DEGREE} \
        --pipeline_parallel_degree 1 \
        --disable_tqdm True \
        --tensorboard ${enable_tensorboard} \
        --stu_pretrained_model_path ${STU_PRETRAIN_CKPT} \
        --tea_pretrained_model_path ${TEA_PRETRAIN_CKPT} \
        --fp16_opt_level ${FP16_OPT_LEVEL} \
        --seed ${seed} \
        --recompute True \
        --bf16 ${USE_AMP} \
```


默认teacher为`paddlemix/EVA/EVA01-CLIP-g-14`，如果更换teacher，可改为类似如下：

```
model_name="paddlemix/EVA/EVA02/eva02_Ti_for_pretrain" # should modify teacher_config in config.json and model_config.json
# model_name=None # if set None, will use teacher_name and student_name from_pretrained, both should have config and pdparams
teacher_name="paddlemix/EVA/EVA02-CLIP-bigE-14"
student_name="paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_p14"

TEA_PRETRAIN_CKPT=None # /root/.paddlenlp/models/paddlemix/EVA/EVA02-CLIP-bigE-14/model_state.pdparams # must add if MP_DEGREE > 1
STU_PRETRAIN_CKPT=None
```

注意：
1. `model_name` 可单独使用创建模型，如果更换teacher，则需自己改写`paddlemix/EVA/EVA02/eva02_Ti_for_pretrain`中config.json and model_config.json的teacher_config这个字段的内容，比如将默认的 `paddlemix/EVA/EVA01-CLIP-g-14` 改为 "paddlemix/EVA/EVA02-CLIP-bigE-14"。而student_config是dict，student模型本身是train from scratch的；
2. 如果model_name=None，也可采用teacher_name 和 student_name来创建模型，但它们必须都各自具有config.json和model_state.pdparams，一般eval或加载全量权重debug时采用model_name=None的形式；
3. `TEA_PRETRAIN_CKPT`通常情况下设置为None，模型训练前已加载来自`teacher_name`中的对应teacher预训练权重。但是**如果设置 MP_DEGREE > 1**时，则必须再次设置`TEA_PRETRAIN_CKPT`的路径去加载，一般设置绝对路径，也可从对应的下载链接单独下载相应的`model_state.pdparams`并放置；



### 4.2 Finetune训练

使用`paddlemix/examples/eva02/run_eva02_finetune_dist.py`。

注意：

1. 如果采用分布式策略，分布式并行关系有：`nnodes * nproc_per_node == tensor_parallel_degree * sharding_parallel_degree * dp_parallel_degree`，其中`dp_parallel_degree`参数根据其他几个值计算出来，因此需要保证`nnodes * nproc_per_node >= tensor_parallel_degree * sharding_parallel_degree`；

2. tiny/s是336尺度训练，B/L是448尺度训练，而它们的预训练权重均是224尺度训练得到的。

3. 如果训练`paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_ft_in1k_p14`， 则必须加载**其对应的预训练权重**`paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_p14`，然后设置`PRETRAIN_CKPT`即预训练权重的`model_state.pdparams`的绝对路径；或者可以单独从[模型对应下载链接](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_p14/model_state.pdparams)下载并放置。其他模型同理。


训练命令及参数配置示例，这里示例采用单机8卡程序，运行前请先**确保预训练权重即`PRETRAIN_CKPT`的路径是存在的**：
```shell
export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export NVIDIA_TF32_OVERRIDE=0
export NCCL_ALGO=Tree

optim="adamw"
lr=2e-4
layer_decay=0.9
warmup_lr=0.0
min_lr=0.0
weight_decay=0.05
CLIP_GRAD=0.0

num_train_epochs=100
save_epochs=2 # save every 2 epochs

warmup_epochs=5 # set 0 will fast convergence in 1 epoch
warmup_steps=0
drop_path=0.1

TRAINING_MODEL_RESUME="None"
TRAINER_INSTANCES='127.0.0.1'
MASTER='127.0.0.1:8080'

TRAINERS_NUM=1 # nnodes, machine num
TRAINING_GPUS_PER_NODE=8 # nproc_per_node
DP_DEGREE=8 # dp_parallel_degree
MP_DEGREE=1 # tensor_parallel_degree
SHARDING_DEGREE=1 # sharding_parallel_degree

MODEL_NAME="paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_ft_in1k_p14"
# wget https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_p14/model_state.pdparams
PRETRAIN_CKPT=/root/.paddlenlp/models/paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_p14/model_state.pdparams # must be added, pretrained model, input_size is 224

OUTPUT_DIR=./output/eva02_Ti_pt_in21k_ft_in1k_p14

DATA_PATH=./dataset/ILSVRC2012 # put your ImageNet-1k val data path

input_size=336
batch_size=128 # 128(bsz_per_gpu)*8(#gpus_per_node)*1(#nodes)*1(update_freq)=1024(total_bsz)
num_workers=10
accum_freq=1 # update_freq
logging_steps=10 # print_freq
seed=0

USE_AMP=False
FP16_OPT_LEVEL="O1"
enable_tensorboard=True

TRAINING_PYTHON="python -m paddle.distributed.launch --master ${MASTER} --nnodes ${TRAINERS_NUM} --nproc_per_node ${TRAINING_GPUS_PER_NODE} --ips ${TRAINER_INSTANCES}"
${TRAINING_PYTHON} run_eva02_finetune_dist.py \
        --do_train \
        --data_path ${DATA_PATH}/train \
        --eval_data_path ${DATA_PATH}/val \
        --pretrained_model_path ${PRETRAIN_CKPT} \
        --model ${MODEL_NAME} \
        --input_size ${input_size} \
        --layer_decay ${layer_decay} \
        --drop_path ${drop_path} \
        --optim ${optim} \
        --learning_rate ${lr} \
        --weight_decay ${weight_decay} \
        --adam_beta1 0.9 \
        --adam_beta2 0.999 \
        --adam_epsilon 1e-8 \
        --max_grad_norm ${CLIP_GRAD} \
        --lr_scheduler_type cosine \
        --lr_end 1e-7 \
        --warmup_lr ${warmup_lr} \
        --min_lr ${min_lr} \
        --num_train_epochs ${num_train_epochs} \
        --save_epochs ${save_epochs} \
        --warmup_epochs ${warmup_epochs} \
        --per_device_train_batch_size ${batch_size} \
        --dataloader_num_workers ${num_workers} \
        --output_dir ${OUTPUT_DIR} \
        --logging_dir ${OUTPUT_DIR}/tb_log \
        --logging_steps ${logging_steps} \
        --accum_freq ${accum_freq} \
        --dp_degree ${DP_DEGREE} \
        --tensor_parallel_degree ${MP_DEGREE} \
        --sharding_parallel_degree ${SHARDING_DEGREE} \
        --pipeline_parallel_degree 1 \
        --disable_tqdm True \
        --tensorboard ${enable_tensorboard} \
        --recompute True \
        --fp16_opt_level ${FP16_OPT_LEVEL} \
        --seed ${seed} \
        --fp16 ${USE_AMP} \
```


### 4.3 评估

使用`paddlemix/examples/eva02/run_eva02_finetune_eval.py`。

注意：

1. 默认加载的是下载的`paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_ft_in1k_p14`里的训好的权重，所以`PRETRAIN_CKPT=None`，**如果是本地新训好的权重**，则可设置`PRETRAIN_CKPT`的具体路径去加载和评估；


```shell
MODEL_NAME="paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_ft_in1k_p14"
DATA_PATH=./dataset/ILSVRC2012 # put your ImageNet-1k val data path
OUTPUT_DIR=./outputs

input_size=336
batch_size=128
num_workers=10

PRETRAIN_CKPT=None # output/eva02_Ti_pt_in21k_ft_in1k_p14/checkpoint-xxx/model_state.pdparams

CUDA_VISIBLE_DEVICES=0 python run_eva02_finetune_eval.py \
        --do_eval \
        --model ${MODEL_NAME} \
        --pretrained_model_path ${PRETRAIN_CKPT} \
        --eval_data_path ${DATA_PATH}/val \
        --input_size ${input_size} \
        --per_device_eval_batch_size ${batch_size} \
        --dataloader_num_workers ${num_workers} \
        --output_dir ${OUTPUT_DIR} \
        --recompute True \
        --fp16 False \
```

```
# 参数说明

--model #设置实际使用的模型，示例为`paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_ft_in1k_p14`，会自动下载，也可自己写本地机器上的路径，后面的模型可自行替换

--eval_data_path #评估数据路径

--input_size #模型输入尺度，注意 Tiny/S 是336尺度评估，B/L 是448尺度评估

--per_device_eval_batch_size #评估时单卡batch_size

--dataloader_num_workers #数据加载线程数量

--output_dir #模型输出文件路径

--recompute #是否开启recompute节省显存

--fp16 #是否开启fp16推理
```

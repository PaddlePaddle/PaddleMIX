# CLIP

## 1. 模型简介

[CLIP](https://arxiv.org/abs/2103.00020): Contrastive Language-Image Pre-Training 论文的Paddle实现版本，实现了`CLIP`的`ViT`系列模型，对齐的是huggingface上的[OpenCLIP LAION-2B](https://huggingface.co/collections/laion/openclip-laion-2b-64fcade42d20ced4e9389b30)系列和[OpenCLIP DataComp](https://huggingface.co/collections/laion/openclip-datacomp-64fcac9eb961d0d12cb30bc3)系列，以及提供了[apple/DFN](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378)系列。

语言-图像预训练对比学习，在很多语言、图像应用场景表现出性能优势，有着广泛的应用。

<p align="center">
  <img src="https://github.com/openai/CLIP/blob/main/CLIP.png?raw=true" align="middle" width = "800" />
</p>

注：图片引用自[openai/CLIP](https://github.com/openai/CLIP)。


### 2. CLIP Model Zoo

#### OpenCLIP models trained on LAION-2B

<div align="center">

| model name                          |   params (M)      | IN-1K zero-shot top-1 | weight(fp16) |
|:------------------------------------|:------------------|:---------------------:|:------------:|
| `CLIP-ViT-bigG-14-laion2B-39B-b160k`|       2539.57     | **80.1** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/CLIP/CLIP-ViT-bigG-14-laion2B-39B-b160k/model_state.pdparams) |
| `CLIP-ViT-g-14-laion2B-s34B-b88K`   |       1366.68     | **78.5** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/CLIP/CLIP-ViT-g-14-laion2B-s34B-b88K/model_state.pdparams) |
| `CLIP-ViT-H-14-laion2B-s32B-b79K`   |         986.11    | **78.0** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/CLIP/CLIP-ViT-H-14-laion2B-s32B-b79K/model_state.pdparams) |
| `CLIP-ViT-L-14-laion2B-s32B-b82K`   |         427.62    | **75.2** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/CLIP/CLIP-ViT-L-14-laion2B-s32B-b82K/model_state.pdparams) |
| `CLIP-ViT-B-16-laion2B-s34B-b88K`   |         149.62    | **70.2** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/CLIP/CLIP-ViT-B-16-laion2B-s34B-b88K/model_state.pdparams) |
| `CLIP-ViT-B-32-laion2B-s34B-b79K`   |         151.28    | **66.5** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/CLIP/CLIP-ViT-B-32-laion2B-s34B-b79K/model_state.pdparams) |

</div>

#### OpenCLIP models trained on DataComp

<div align="center">

| model name                          |   params (M)      | IN-1K zero-shot top-1 | weight(fp16) |
|:------------------------------------|:------------------|:---------------------:|:------------:|
| `CLIP-ViT-L-14-DataComp.XL-s13B-b90K`|       427.62     | **79.2** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/CLIP/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/model_state.pdparams) |
| `CLIP-ViT-B-16-DataComp.XL-s13B-b90K`|         149.62   | **73.5** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/CLIP/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/model_state.pdparams) |
| `CLIP-ViT-B-32-DataComp.XL-s13B-b90K`|         151.28   | **69.2** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/CLIP/CLIP-ViT-B-32-DataComp.XL-s13B-b90K/model_state.pdparams) |
| `CLIP-ViT-B-32-256x256-DataComp-s34B-b86K`|    151.28   | **72.8** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/CLIP/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K/model_state.pdparams) |

</div>

#### CLIP models trained on Data Filtering Networks (DFNs)

<div align="center">

| model name                |   params (M)      | IN-1K zero-shot top-1 | weight(fp16) |
|:--------------------------|:------------------|:---------------------:|:------------:|
| `DFN5B-CLIP-ViT-H-14-378` |       986.71      | **84.2** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/CLIP/DFN5B-CLIP-ViT-H-14-378/model_state.pdparams) |
| `DFN5B-CLIP-ViT-H-14`     |       986.71      | **83.4** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/CLIP/DFN5B-CLIP-ViT-H-14/model_state.pdparams) |
| `DFN2B-CLIP-ViT-L-14`     |       427.62      | **81.4** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/CLIP/DFN2B-CLIP-ViT-L-14/model_state.pdparams) |
| `DFN2B-CLIP-ViT-B-16`     |       149.62      | **76.2** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/CLIP/DFN2B-CLIP-ViT-B-16/model_state.pdparams) |

</div>

注意:
配置文件下载均为权重weight链接同目录下的`config.json`，如`CLIP-ViT-L-14-laion2B-s32B-b82K`模型的[config.json](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/CLIP/CLIP-ViT-L-14-laion2B-s32B-b82K/config.json)。


## 2. 环境准备

1） 安装PaddleNLP develop版本

```
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

2）安装环境依赖包

```
pip install -r requirements.txt
```

3）安装`FusedLayerNorm`，在`paddlemix/external_ops/`目录下，安装fusedln包。(CLIP模型可不安装)

```
# 安装fusedln到python环境
python setup.py install --prefix=$INSTALL_DIR
# 添加安装路径到系统环境路径中
export $PATH=$PATH:$INSTALL_DIR
```


## 3. 数据准备

1) coco数据

数据部分，默认使用`coco_karpathy`数据，使用该数据不需另外配置，会自动下载。解析部分参考`paddlemix/datasets/coco_clip.py`文件。

如果想手动下载，请点击[DownLoadCOCO 20G](https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/coco.tar)下载数据，可解压后放在`/root/.paddlemix/datasets/`目录下，此目录也为自动下载并解压的目录。

2) laion数据
该数据集较大，对于训练速度和内存占用有限制的情况，建议使用`coco_karpathy`数据。使用该数据集时，参数`--task_name`需要是指向laion.filelist文件的路径。
laion.filelist文件格式示例如下：
```
laiondata-pathdir/part-00000
laiondata-pathdir/part-00001
...
```
具体解析代码参考`paddlemix/datasets/laiondata.py`文件。
该数据集暂不提供，请自行下载。

3) 自定义数据

如果需要自定义数据，推荐沿用`coco_karpathy`数据格式处理自己的数据。其中每条数据标注格式示例为:
```
{'caption': 'A woman wearing a net on her head cutting a cake. ', 'image': 'val2014/COCO_val2014_000000522418.jpg', 'image_id': 'coco_522418'}
```
更多可参考数据集中的`annotations/coco_karpathy_train.json`文件。

4) 验证数据集-ImageNet1K

请点击[DownLoad imagenet-val.tar](https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/ILSVRC2012/imagenet-val.tar)下载并解压数据，并将下述训练脚本里的`IN_1K_DIR`内容填写为解压后的路径。

## 4. 使用说明

### 4.1 训练

训练时使用`paddlemix/examples/clip/run_pretrain_dist.py`程序进行训练，**训练前请先检查数据集路径**，如COCO数据集一般会被默认解压存放在`/root/.paddlemix/datasets/coco`目录。

训练命令及参数配置示例：

这里示例采用单机8卡程序，sharding_degree=8.

注意如果采用分布式策略，分布式并行关系有：`nnodes * nproc_per_node == tensor_parallel_degree * sharding_parallel_degree * dp_parallel_degree`，其中`dp_parallel_degree`参数根据其他几个值计算出来，因此需要保证`nnodes * nproc_per_node >= tensor_parallel_degree * sharding_parallel_degree`。

```
MODEL_NAME="paddlemix/CLIP/CLIP-ViT-L-14-laion2B-s32B-b82K"
TEXT_MODEL_NAME="CLIP-ViT-L-14-laion2B-s32B-b82K"
IN_1K_DIR=[YOUR ImageNet1K val data path]

python -m paddle.distributed.launch --nproc_per_node 8 run_pretrain_dist.py \
    --dataloader_num_workers=2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --model ${MODEL_NAME}  \
    --warmup_steps 2000 \
    --learning_rate 5e-4 \
    --weight_decay 0.05 \
    --adam_beta1 0.9  \
    --adam_beta2 0.999  \
    --adam_epsilon 1e-8  \
    --max_grad_norm 5.0 \
    --num_train_epochs 200 \
    --tensor_parallel_degree 1 \
    --sharding_parallel_degree 8 \
    --sharding "stage2" \
    --bf16 False \
    --output_dir "./output" \
    --logging_steps 1 \
    --do_train \
    --disable_tqdm True \
    --save_steps 50000 \
    --local_loss true \
    --gather_with_grad true \
    --pretrained_text_model ${TEXT_MODEL_NAME} \
    --classification_eval ${IN_1K_DIR} \

```


```
# 参数说明

--model #设置实际使用的模型，示例'paddlemix/CLIP/CLIP-ViT-L-14-laion2B-s32B-b82K'，或者可换为'paddlemix/CLIP/CLIP-ViT-B-16-laion2B-s34B-b88K'

--dataloader_num_workers #数据加载线程数量

--per_device_train_batch_size #训练时单卡batch_size

--learning_rate 5e-4 #global默认学习率，优先级低于visual_lr、text_lr

--weight_decay 0.05 #global默认weight decay, 优先级低于visual_wd、text_wd

--adam_beta1 0.9  #optimizer中beta1参数

--adam_beta2 0.999  #optimizer中beta2参数

--adam_epsilon 1e-8  #optimizer中epsilon参数

--max_grad_norm 5.0 #最大梯度裁剪，将norm大于该值的grad裁剪到该值

--num_train_epochs 200 #整体训练epoch次数

--tensor_parallel_degree 1 #模型并行系数，设置为N则进行N卡间模型并行

--sharding_parallel_degree 8 #显存优化策略，默认stage1，详情参考 [《ZeRO: Memory Optimizations Toward Training Trillion Parameter Models》]（https://arxiv.org/abs/1910.02054）

--sharding "stage1" #显存优化策略stage选择，目前支持stage1、stage2

--fp16 False #是否开启float16训练

--output_dir "./output" #模型存储路径

--logging_steps 1 #logging显示间隔steps

--do_train #执行训练

--save_steps 50000 #每多少个steps保存一次模型

--local_loss true #loss中是否开启local loss

--gather_with_grad true #loss中是否打开gather_with_grad

--pretrained_text_model ${TEXT_MODEL_NAME} #预提取text features的模型

--classification_eval ${IN_1K_DIR} #IN_1K测试数据路径
```

### 4.2 评估

评估时使用`paddlemix/examples/evaclip/run_zero_shot_eval.py`程序进行评估。

评估命令及参数配置示例：

```
MODEL_NAME="paddlemix/CLIP/CLIP-ViT-L-14-laion2B-s32B-b82K"
TEXT_MODEL_NAME="CLIP-ViT-L-14-laion2B-s32B-b82K"

IN_1K_DIR=[YOUR ImageNet1K val data path]

python paddlemix/examples/clip/run_zero_shot_eval.py \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers=2 \
    --model ${MODEL_NAME}  \
    --fp16 False \
    --pretrained_text_model ${TEXT_MODEL_NAME} \
    --classification_eval ${IN_1K_DIR} \
    --output_dir "output" \
    --disable_tqdm False \
```

```
# 参数说明

--model #设置实际使用的模型，示例'paddlemix/CLIP/CLIP-ViT-L-14-laion2B-s32B-b82K'

--dataloader_num_workers #数据加载线程数量

--per_device_eval_batch_size #评估时单卡batch_size

--fp16 False #是否开启fp16推理

--pretrained_text_model ${TEXT_MODEL_NAME} #预提取text features的模型

--classification_eval ${IN_1K_DIR} #IN_1K测试数据路径

--output_dir "output" #模型输出文件路径

--disable_tqdm False #是否关闭tqdm进度条
```

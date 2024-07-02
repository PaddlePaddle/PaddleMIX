# CoCa

## 1. 模型简介

[CoCa](https://arxiv.org/abs/2205.01917): Contrastive Captioners are Image-Text Foundation Models论文的 Paddle实现版本。

CLIP: 语言-图像预训练对比学习，在很多语言、图像应用场景表现出性能优势，有着广泛的应用。

CoCa: 在CLIP基础上增加decoder形成编码、解码结构，并结合使用contrastive loss和captioning loss。该模型在ImageNet上实现了91.0% top-1准确率的SOTA精度。

<p align="center">
  <img src="https://github.com/lucidrains/CoCa-pytorch/blob/main/coca.png" align="middle" width = "800" />
</p>

注：图片引用自[CoCa-pytorch](https://github.com/lucidrains/CoCa-pytorch).


### CoCa Model Zoo

<div align="center">

| model name                      |   params (M)        | IN-1K zero-shot top-1 | weight(bf16) |
|:--------------------------------|:--------------------|:---------------------:|:------------:|
| `CoCa-ViT-B-32-laion2B-s13B-b90k`|         253.56     | **63.6** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/CoCa/CoCa-ViT-B-32-laion2B-s13B-b90k/model_state.pdparams) |
| `CoCa-ViT-L-14-laion2B-s13B-b90k`|         638.45     | **75.7** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/CoCa/CoCa-ViT-L-14-laion2B-s13B-b90k/model_state.pdparams) |
| `mscoco_finetuned_CoCa-ViT-B-32-laion2B-s13B-b90k`| 253.56 | - | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/CoCa/mscoco_finetuned_CoCa-ViT-B-32-laion2B-s13B-b90k/model_state.pdparams) |
| `mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k`| 638.45 | **72.0** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/CoCa/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/model_state.pdparams) |

</div>


## 2. 环境准备

1） 安装PaddleNLP develop版本

```
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

2）安装环境依赖包

```
pip install -r requirements.txt
```

3）安装`FusedLayerNorm`，在`paddlemix/external_ops/`目录下，安装fusedln包。(CoCa模型可不安装)

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

## 4. 使用说明

我们在Paddle中实现了`CoCa`系列模型，包括`CoCa-ViT-B-32-laion2B-s13B-b90k`、`CoCa-ViT-L-14-laion2B-s13B-b90k`、`mscoco_finetuned_CoCa-ViT-B-32-laion2B-s13B-b90k`、`mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k`。

### 4.1 训练

训练时使用`paddlemix/examples/coca/run_pretrain_dist.py`程序进行训练，**训练前请先检查数据集路径**，如COCO数据集一般会被默认解压存放在`/root/.paddlemix/datasets/coco`目录。

训练命令及参数配置示例：

这里示例采用单机8卡程序，sharding_degree=8。

注意如果采用分布式策略，分布式并行关系有：`nnodes * nproc_per_node == tensor_parallel_degree * sharding_parallel_degree * dp_parallel_degree`，其中`dp_parallel_degree`参数根据其他几个值计算出来，因此需要保证`nnodes * nproc_per_node >= tensor_parallel_degree * sharding_parallel_degree`。

```
MODEL_NAME="paddlemix/CoCa/CoCa-ViT-L-14-laion2B-s13B-b90k"
TEXT_MODEL_NAME="CoCa-ViT-L-14-laion2B-s13B-b90k"

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

--model #设置实际使用的模型，示例'paddlemix/CoCa/CoCa-ViT-L-14-laion2B-s13B-b90k'，也可替换为'paddlemix/CoCa/CoCa-ViT-B-32-laion2B-s13B-b90k'

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

评估时使用`paddlemix/examples/coca/run_zero_shot_eval.py`程序进行评估。

评估命令及参数配置示例：

```
MODEL_NAME="paddlemix/CoCa/CoCa-ViT-L-14-laion2B-s13B-b90k"
TEXT_MODEL_NAME="CoCa-ViT-L-14-laion2B-s13B-b90k"

IN_1K_DIR=[YOUR ImageNet1K val data path]

python paddlemix/examples/coca/run_zero_shot_eval.py \
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

--model #设置实际使用的模型,示例'paddlemix/CoCa/CoCa-ViT-L-14-laion2B-s13B-b90k'

--dataloader_num_workers #数据加载线程数量

--per_device_eval_batch_size #评估时单卡batch_size

--fp16 False #是否开启fp16推理

--pretrained_text_model ${TEXT_MODEL_NAME} #预提取text features的模型

--classification_eval ${IN_1K_DIR} #IN_1K测试数据路径

--output_dir "output" #模型输出文件路径

--disable_tqdm False #是否关闭tqdm进度条
```

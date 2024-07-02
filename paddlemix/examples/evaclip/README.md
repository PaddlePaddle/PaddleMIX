# EVA-CLIP

## 1. 模型简介

[EVA-CLIP](https://arxiv.org/abs/2303.15389): Improved Training Techniques for CLIP at Scale, Paddle实现版本.

CLIP：语言-图像预训练对比学习，在很多语言、图像应用场景表现出性能优势，有着广泛的应用。

EVA-CLIP：针对CLIP训练过程进行优化，使得训练效率和效果都得到明显提升。

<p align="center">
  <img src="https://github.com/baaivision/EVA/blob/master/EVA-CLIP/assets/teaser.png?raw=true" align="middle" width = "800" />
</p>

注：图片引用自[EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP).


### EVA-01-CLIP Series

> Image encoder MIM teacher: ``OpenAI CLIP-Large``.

| model name | image enc. init. ckpt | text enc. init. ckpt | total #params | IN-1K zero-shot top-1 | weight(bf16) |
|:-----|:-----|:-----------|:------:|:------:|:------:|
| `EVA01-CLIP-g-14` | `EVA01_g_psz14` | `openai/clip-vit-large-patch14` | 1.1B | **78.5** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA01-CLIP-B-14/model_state.pdparams) |
| `EVA01-CLIP-g-14-plus` | `EVA01_g_psz14` | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | 1.3B | **79.3** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA01-CLIP-B-14-plus/model_state.pdparams) |

</div>


### EVA-02-CLIP Series

> Image encoder MIM teacher: ``EVA01_CLIP_g_14_psz14_s11B``.

<div align="center">

| model name | image enc. init. ckpt | text enc. init. ckpt | total #params | IN-1K zero-shot top-1 | weight(bf16) |
|:-----|:-----|:-----------|:------:|:------:|:------:|
| `EVA02-CLIP-B-16` | `EVA02_B_psz14to16` | `openai/clip-vit-base-patch16` | 149M | **74.6** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02-CLIP-B-16/model_state.pdparams) |
| `EVA02-CLIP-L-14` | `EVA02_L_psz14` | `openai/clip-vit-large-patch14` | 428M | **79.6** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02-CLIP-L-14/model_state.pdparams) |
| `EVA02-CLIP-L-14-336` | `EVA02_CLIP_L_psz14_224to336` | `EVA02_CLIP_L_psz14_224to336` | 428M | **80.3** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02-CLIP-L-14-336/model_state.pdparams) |
| `EVA02-CLIP-bigE-14` | `EVA02_E_psz14` | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | 4.7B | **82.0** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02-CLIP-bigE-14/model_state.pdparams) |
| `EVA02-CLIP-bigE-14-plus` | `EVA02_E_psz14` | `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` | 5.0B | **82.0** | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/EVA/EVA02-CLIP-bigE-14-plus/model_state.pdparams) |

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

3）安装`FusedLayerNorm`，在`paddlemix/external_ops/`目录下，安装fusedln包。

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

我们在Paddle中实现了`EVA-CLIP`系列模型，包括`EVA01-CLIP-g-14`、`EVA01-CLIP-g-14-plus`、`EVA02-CLIP-B-16`、`EVA02-CLIP-L-14`、`EVA02-CLIP-L-14-336`、`EVA02-CLIP-bigE-14`、`EVA02-CLIP-bigE-14-plus`.

### 4.1 训练

训练时使用`paddlemix/examples/evaclip/run_pretrain_dist.py`程序进行训练，**训练前请先检查数据集路径**，如COCO数据集一般会被默认解压存放在`/root/.paddlemix/datasets/coco`目录。

训练命令及参数配置示例：

这里示例采用单机8卡程序，sharding_degree=8.

注意如果采用分布式策略，分布式并行关系有：`nnodes * nproc_per_node == tensor_parallel_degree * sharding_parallel_degree * dp_parallel_degree`，其中`dp_parallel_degree`参数根据其他几个值计算出来，因此需要保证`nnodes * nproc_per_node >= tensor_parallel_degree * sharding_parallel_degree`。

```
MODEL_NAME="paddlemix/EVA/EVA02-CLIP-L-14"
IN_1K_DIR=[YOUR ImageNet1K val data path]

python -m paddle.distributed.launch --nproc_per_node 8 run_pretrain_dist.py \
    --dataloader_num_workers=2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --model ${MODEL_NAME}  \
    --optimizer 'lamb' \
    --warmup_steps 2000 \
    --learning_rate 5e-4 \
    --visual_lr 2e-4 \
    --text_lr 2e-5 \
    --weight_decay 0.05 \
    --visual_wd 0.05 \
    --text_wd 0.05 \
    --layer_decay 1.0 \
    --visual_ld 0.75 \
    --text_ld 0.75 \
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
    --pretrained_text_model ${MODEL_NAME} \
    --classification_eval ${IN_1K_DIR} \

```


```
# 参数说明

--model #设置实际使用的模型,示例'paddlemix/EVA/EVA02-CLIP-B-16'、'paddlemix/EVA/EVA02-CLIP-L-14'

--dataloader_num_workers #数据加载线程数量

--per_device_train_batch_size #训练时单卡batch_size

--optimizer #optimizer选择，当前支持[lamb、adamw]

--learning_rate 5e-4 #global默认学习率，优先级低于visual_lr、text_lr

--visual_lr 2e-4 #visual tower默认学习率

--text_lr 2e-5 #text tower默认学习率

--weight_decay 0.05 #global默认weight decay, 优先级低于visual_wd、text_wd

--visual_wd 0.05 #visual tower weight decay

--text_wd 0.05 #text tower weight decay

--layer_decay 1.0 #全局分层学习率设置参数, 优先级低于visual_ld、text_ld

--visual_ld 0.75 #visual tower学习率衰减系数，随层数加深学习率衰减比率

--text_ld 0.75 #text tower学习率衰减系数，随层数加深学习率衰减比率

--adam_beta1 0.9  #optimizer中beta1参数，适用于adamw、lamb

--adam_beta2 0.999  #optimizer中beta2参数，适用于adamw、lamb

--adam_epsilon 1e-8  #optimizer中epsilon参数，适用于adamw、lamb

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

--pretrained_text_model EVA02-CLIP-L-14 #预提取text features的模型

--classification_eval ${IN_1K_DIR} #IN_1K测试数据路径
```

### 4.2 评估

评估时使用`paddlemix/examples/evaclip/run_zero_shot_eval.py`程序进行评估。

评估命令及参数配置示例：

```
MODEL_NAME="paddlemix/EVA/EVA02-CLIP-L-14"

IN_1K_DIR=[YOUR ImageNet1K val data path]

python paddlemix/examples/evaclip/run_zero_shot_eval.py \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers=2 \
    --model ${MODEL_NAME}  \
    --fp16 False \
    --pretrained_text_model EVA02-CLIP-L-14 \
    --classification_eval ${IN_1K_DIR} \
    --output_dir "output" \
    --disable_tqdm True \
```

```
# 参数说明

--model #设置实际使用的模型,示例'paddlemix/EVA/EVA02-CLIP-B-16'、'paddlemix/EVA/EVA02-CLIP-L-14'

--dataloader_num_workers #数据加载线程数量

--per_device_eval_batch_size #评估时单卡batch_size

--fp16 False #是否开启fp16推理

--pretrained_text_model EVA02-CLIP-L-14 #预提取text features的模型

--classification_eval ${IN_1K_DIR} #IN_1K测试数据路径

--output_dir "output" #模型输出文件路径

--disable_tqdm True #是否关闭tqdm进度条
```

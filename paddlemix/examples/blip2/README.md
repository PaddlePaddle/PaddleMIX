# BLIP-2

## 1. 模型简介

[BLIP-2](https://arxiv.org/abs/2301.12597): Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models, Paddle实现版本.

BLIP-2：使用冻结图像编码器和大型语言模型的语言图像预训练，在VQA,Caption等多图文任务上性能爆表表现出性能优势。

<p align="center">
  <img src="https://github.com/salesforce/LAVIS/blob/main/projects/blip2/blip2_illustration.png" align="middle" width = "600" />
</p>

注：图片引用自[BLIP-2](https://github.com/salesforce/LAVIS/blob/main/projects/blip2).


## 2. 环境准备

1） 安装PaddleNLP develop版本

```
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

2）安装环境依赖包

```
pip install -r requirements.txt
```

## 3. 数据准备


1) coco数据

数据部分，默认使用`coco_karpathy`数据，使用该数据不需另外配置，会自动下载。

2) 自定义数据

如果需要自定义数据，推荐沿用`coco_karpathy`数据格式处理自己的数据。其中每条数据标注格式示例为:
```
{'caption': 'A woman wearing a net on her head cutting a cake. ', 'image': 'val2014/COCO_val2014_000000522418.jpg', 'image_id': 'coco_522418'}
```
更多可参考数据集中的`annotations/coco_karpathy_train.json`文件。

## 4. 使用说明

我们在Paddle中实现了`BLIP-2`系列模型，目前包括`BLIP-2-OPT`、`BLIP-2-FlanT5`



### 4.1 训练

无需更改参数即可开始训练BLIP-2
如需调整参数请见以下参数配置示例：

MODEL_NAME="paddlemix/blip2-stage2"

fleetrun --master '127.0.0.1' --nnodes 1 --nproc_per_node 8 --ips '127.0.0.1:8080' run_pretrain_stage2.py \
    --per_device_train_batch_size 256 \
    --model_name_or_path ${MODEL_NAME}  \
    --warmup_steps 2000 \
    --eta_min 1e-5 \
    --learning_rate 0.0001 \
    --weight_decay 0.05 \
    --num_train_epochs 10 \
    --tensor_parallel_degree 1 \
    --sharding_parallel_degree 1 \
    --sharding "stage1" \
    --output_dir "./output" \
    --logging_steps 1 \
    --do_train \
    --disable_tqdm True \
    --save_steps 5000 \
```
model_name_or_path目前支持: 
    blip2-stage1预训练模型: "paddlemix/blip2-stage1"
    blip2-stage2预训练模型: "paddlemix/blip2-stage2"
    blip2-vqa模型/微调预训练模型: "paddlemix/blip2-pretrained-opt2.7b", "paddlemix/blip2-pretrained-opt6.7b", "paddlemix/blip2_pretrained_flant5xl", "paddlemix/blip2_pretrained_flant5xxl“
    blip2-caption模型: "paddlemix/blip2-caption-opt2.7b", "paddlemix/blip2-caption-opt6.7b", "paddlemix/blip2_caption_flant5xl.7b"
```

#### stage1
```
# 单卡训练
CUDA_VISIBLE_DEVICES=0 python paddlevlp/examples/blip2/run_pretrain_stage1.py
# 多卡训练
fleetrun --gpus=0,1,2,3 paddlevlp/examples/blip2/run_pretrain_stage1.py
```
#### stage2
```
# 单卡训练
CUDA_VISIBLE_DEVICES=0 python paddlevlp/examples/blip2/run_pretrain_stage2.py
# 多卡训练
fleetrun --gpus=0,1,2,3 paddlevlp/examples/blip2/run_pretrain_stage2.py
```
### 4.2 评估

#### task_vqa
```
fleetrun --gpus=0,1,2,3 paddlevlp/examples/blip2/run_eval_vqa2_zeroshot.py
```
#### task_caption
```
fleetrun --gpus=0,1,2,3 paddlevlp/examples/blip2/run_eval_caption.py
```

### 4.3 预测
```
CUDA_VISIBLE_DEVICES=0 python paddlevlp/examples/blip2/run_predict.py
```
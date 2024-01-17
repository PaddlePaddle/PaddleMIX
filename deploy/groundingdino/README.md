# Grounding DINO

## 1.模型简介

[Grounding DINO](https://arxiv.org/abs/2303.05499) 是一个开集（Open-Set）的目标检测模型，根据输入的文本提示，进行目标检测。
本仓库是Grounding DINO的Paddle实现，提供部署代码。



## 2.自定义op安装 （可选）
```bash
#Multi-scale deformable attention custom OP compilation
cd /paddlemix/models/groundingdino/csrc/
python setup_ms_deformable_attn_op.py install

```

## 3.快速开始
## 静态图导出与预测
```bash
cd deploy/groundingdino

#静态图模型导出
python export.py \
--dino_type GroundingDino/groundingdino-swint-ogc


#静态图预测
 python predict.py  \
 --text_encoder_type GroundingDino/groundingdino-swint-ogc \
 --model_path output_groundingdino/GroundingDino/groundingdino-swint-ogc \
 --input_image https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg \
 --output_dir ./groundingdino_predict_output \
 --prompt "bus"

```

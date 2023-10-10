# Grounding DINO

## 1. 模型简介

该模型是 [Grounding DINO](https://arxiv.org/abs/2303.05499) 的 paddle 实现。


## 2. 示例

## 2.1 依赖安装 （可选）
```bash
#Multi-scale deformable attention custom OP compilation
cd /paddlemix/models/groundingdino/csrc/
python setup_ms_deformable_attn_op.py install

```
## 2.2 静态图导出与预测
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

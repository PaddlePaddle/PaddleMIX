# Segment Anything

## 1. 模型简介

该模型是 [Segment Anything](https://ai.facebook.com/research/publications/segment-anything/) 的 paddle 实现, 可输入点或框进行分割。


## 2. 示例

## 2.1 静态图导出与预测
```bash
#导出输入类型是 bbox 的静态图
python export.py --model_type Sam/SamVitH-1024 --input_type boxs  --save_dir sam_export

#导出输入类型是 points 的静态图
python export.py --model_type Sam/SamVitH-1024 --input_type points  --save_dir sam_export



#bbox 提示词推理
python predict.py \
--input_image https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg \
--box_prompt 112 118 513 382 \
--input_type boxs \
--model_name_or_path Sam/SamVitH-1024 \
--cfg Sam/SamVitH-1024_boxs/deploy.yaml


#points 提示词推理
python predict.py \
--input_image https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg \
--points_prompt 548 372 \
--input_type points \
--model_name_or_path Sam/SamVitH-1024 \
--cfg Sam/SamVitH-1024_points/deploy.yaml
```

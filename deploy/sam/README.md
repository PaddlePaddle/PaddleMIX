# Segment Anything

## 1. 模型简介

[Segment Anything](https://ai.facebook.com/research/publications/segment-anything/) 是 Meta AI Research, FAIR
的图像分割模型。根据输入提示（如点或框）生成高质量mask，可为图像中的所有对象进行分割。它已经在1100万张图像和11亿个掩模的数据集上进行了训练，并在各种分割任务上具有强大的零样本性能。
本仓库提供该模型的Paddle部署实现。

## 2. 快速开始

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

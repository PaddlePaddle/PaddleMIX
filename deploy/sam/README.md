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
python predict.py
--input_image image_you_want_to_detect.jpg \
--box_prompt 548 372 593 429 443 374 482 418 \
--input_type boxs \
--model_name_or_path Sam/SamVitH-1024 \
--cfg sam_export_SamVitH_boxs/deploy.yaml

#points 提示词推理
python predict.py \
--input_image mage_you_want_to_detect.jpg \
--points_prompt 548 372 \
--input_type points \
--model_name_or_path Sam/SamVitH-1024 \
--cfg sam_export_SamVitH_points/deploy.yaml
```

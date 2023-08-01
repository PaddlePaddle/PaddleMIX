# 

## 1. 模型简介

Paddle implementation of [Segment Anything](https://ai.facebook.com/research/publications/segment-anything/), produces high quality object masks from input prompts such as points or boxes.


## 2. Demo

## 2.2 Export model for static inference
```bash
#export sam model input_type box
python export.py --model_type Sam/SamVitH-1024 --input_type boxs  --save_dir sam_export

#export sam model input_type points
python export.py --model_type Sam/SamVitH-1024 --input_type points  --save_dir sam_export



#boxs prompt
python predict.py 
--input_image image_you_want_to_detect.jpg \
--box_prompt 548 372 593 429 443 374 482 418 \
--input_type boxs \
--cfg sam_export_SamVitH_boxs/deploy.yaml 

#points prompt
python predict.py \
--input_image mage_you_want_to_detect.jpg \
--points_prompt 548 372 \
--input_type points \
--cfg sam_export_SamVitH_points/deploy.yaml
```


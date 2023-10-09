# BLIP2

## 1. 模型简介

该模型是 [BLIP2](https://arxiv.org/abs/2301.12597) 的 paddle 实现。


## 2. 示例
提供两种方式方式转出语言模型:1.直接下载静态图推理所需语言模型;2.手动转出静态图推理所需语言模型
## 2.1（1）直接下载静态图推理所需语言模型
```bash

bash prepare.sh

```
## 2.1（2）手动转出静态图推理所需语言模型
用户可以去PaddleNLP的llm目录下执行此命令，即可获得第二部分的静态图
```python
python export_model.py --model_name_or_path /root/.paddlenlp/models/facebook/opt-2.7b --output_path opt-2.7b-export --dtype float16 --inference_model --model_prefix=opt --model_type=opt-img2txt
```
## 2.2 静态图导出与预测
```bash
#visual encoder 和 Qformer 静态图模型导出
python export.py \
--model_name_or_path paddlemix/blip2-caption-opt2.7b


#静态图预测
 python predict.py  \
 --first_model_path blip2_export/image_encoder \
 --second_model_path opt-2.7b-infer_static/opt \
 --image_path https://paddlenlp.bj.bcebos.com/data/images/mugs.png\
 --prompt "a photo of"

```

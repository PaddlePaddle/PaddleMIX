# PaddleMIX推理部署

PaddleMIX基于Paddle Inference，提供了python的部署方案。

## 1.一键预测部署

在使用 PaddleMIX 一键预测 **APPflow** 时，可通过设置 static_mode = True 变量开启静态图推理，同时可配合trt加速推理。

### 1.1 示例

```python
>>> from paddlemix import Appflow
>>> from PIL import Image

>>> task = Appflow(app="openset_det_sam",
                   models=["GroundingDino/groundingdino-swint-ogc","Sam/SamVitH-1024"],
                   static_mode=True,
                   precision="fp32")
>>> image_pil = Image.open("beauty.png").convert("RGB")
>>> result = task(image=image_pil,prompt="women")
```

### 1.2 参数说明
| 参数 | 是否必须| 含义                                                                                          |
|-------|-------|---------------------------------------------------------------------------------------------|
| --app | Yes| 应用名称                                                                                   |
| --models | Yes | 需要使用的模型，可以是单个模型，也可以多个组合                                                                                     |
| --static_mode  | Option | 是否静态图推理，默认False                                                                                 |
| --precision | Option | 当 static_mode == True 时使用，默认fp32,可选择trt_fp32、trt_fp16                                                                                    |

说明：
- 部分模型不支持静态图以及trt，具体可参考[跨模态多场景应用
](../applications/README.md)
- 生成的静态图将在模型名字对应的文件夹下 如:GroundingDino/groundingdino-swint-ogc/


## 2. Python端单模型预测部署

Python端预测部署主要包含两个步骤：
- 导出预测模型
- 基于Python进行预测

以 groundingdino 为例子。

### 2.1 导出预测模型

```bash
# 导出groundingdino模型
python deploy/groundingdino/export.py -dt "GroundingDino/groundingdino-swint-ogc" --output_dir=./output
```
导出后目录下，包括 `model_state.pdiparams`,  `model_state.pdiparams.info`, `model_state.pdmodel`等文件。

### 2.2 基于python的预测

```bash
 python deploy/groundingdino/predict.py  \
 --text_encoder_type GroundingDino/groundingdino-swint-ogc
 --model_path output_groundingdino \
 --input_image image_you_want_to_detect.jpg \
 -output_dir "dir you want to save the output" \
 -prompt "Detect Cat"

```

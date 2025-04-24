# PaddleMIX推理部署

[[English](README_en.md)]

PaddleMIX基于Paddle Inference，提供了python的部署方案。部署方式分为两种：
- 通过 **APPflow** ,设置static_mode = True 变量开启静态图推理，同时可配合trt加速推理；该方式部分模型不支持静态图以及trt，具体模型可参考[跨模态多场景应用](../applications/README.md/#跨模态多场景应用)；

- 单模型部署


## 1.APPflow部署

在使用 PaddleMIX 一键预测 **APPflow** 时，可通过设置 static_mode = True 变量开启静态图推理，同时可配合trt加速推理。

### 1.1 示例

```python
>>> from paddlemix.appflow import Appflow
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
- 部分模型不支持静态图以及trt，具体可参考[跨模态多场景应用](../applications/README.md)
- 生成的静态图将在模型名字对应的文件夹下 如:GroundingDino/groundingdino-swint-ogc/


## 2. 单模型预测部署

Python端预测部署主要包含两个步骤：
- 导出预测模型
- 基于Python进行预测

当前支持模型：
- [blip2](./blip2/README.md)
- [groundingdino](./groundingdino/README.md)
- [sam](./sam/README.md)

以 groundingdino 为例子。

### 2.1 导出预测模型

```bash
cd deploy/groundingdino
# 导出groundingdino模型
python export.py \
--dino_type GroundingDino/groundingdino-swint-ogc
```
导出后目录下，包括 `model_state.pdiparams`,  `model_state.pdiparams.info`, `model_state.pdmodel`等文件。

### 2.2 基于python的预测

```bash
 python predict.py  \
 --text_encoder_type GroundingDino/groundingdino-swint-ogc \
 --model_path output_groundingdino/GroundingDino/groundingdino-swint-ogc \
 --input_image https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg \
 --output_dir ./groundingdino_predict_output \
 --prompt "bus"

```

## 3. 推理 BenchMark

> Note:
> 测试环境为:
Paddle 3.0，
PaddleMIX release/2.0
PaddleNLP2.7.2
A100 80G单卡。

### 3.1 benchmark命令

在 `deploy` 对应模型目录下的运行后加 --benchmark,
如 GroundingDino 的benchmark命令为：

```bash
 cd deploy/groundingdino
 python predict.py  \
 --text_encoder_type GroundingDino/groundingdino-swint-ogc \
 --model_path output_groundingdino/GroundingDino/groundingdino-swint-ogc \
 --input_image https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg \
 --output_dir ./groundingdino_predict_output \
 --prompt "bus" \
 --benchmark True
```

# A100性能数据
|模型|图片分辨率|数据类型 |Paddle Deploy |
|-|-|-|-|
|groundingDino/groundingdino-swint-ogc|800*1193|fp32|100 ms|
|Sam/SamVitH-1024|1024*1024|fp32|121 ms|

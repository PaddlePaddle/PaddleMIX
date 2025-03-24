# PaddleMIX Inference Deployment

[[中文文档](README.md)]

PaddleMIX utilizes Paddle Inference and provides a Python-based deployment solution. There are two deployment methods:

1. **APPflow Deployment**: 
   - By setting the `static_mode = True` variable in APPflow, you can enable static graph inference. Additionally, you can accelerate inference using TensorRT. Note that not all models support static graph or TensorRT. Please refer to the [Multi Modal And Scenario](../applications/README_en.md/#multi-modal-and-scenario) section for specific model support.

2. **Single Model Deployment**: 

For APPflow usage, you can set the `static_mode = True` variable to enable static graph inference and optionally accelerate inference using TensorRT.

### 1.1 Examples

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

### 1.2 Parameter Explanation
| Parameter | Required? | Meaning                                                                                          |
|-------|-------|---------------------------------------------------------------------------------------------|
| --app | Yes| Application name                                                                                   |
| --models | Yes | Model(s) used. Can be one model, or multiple models                                                                                    |
| --static_mode  | Optional | Whether to use static graph inference, default to False                                                                                 |
| --precision | Optional | When `static_mode == True`, it defaults to using FP32. You can optionally select `trt_fp32` or `trt_fp16`.                                                                                   |

Instructions：
- Some models do not support static graph or TensorRT. For specific information, please refer to [Multi Modal And Scenario](../applications/README_en.md/#multi-modal-and-scenario).

- The generated static graph will be located in the folder corresponding to the model name, for example: `GroundingDino/groundingdino-swint-ogc/`.

## 2. Single Model Prediction Deployment

Python-based prediction deployment mainly involves two steps:
- Exporting the predictive model
- Performing prediction using Python

Currently supported models:
- [blip2](./blip2/README.md)
- [groundingdino](./groundingdino/README.md)
- [sam](./sam/README.md)
- [qwen_vl](./qwen_vl/README.md)

Using groundingdino as an example.

### 2.1 Exporting Predictive Model

```bash
cd deploy/groundingdino
# 导出groundingdino模型
python export.py \
--dino_type GroundingDino/groundingdino-swint-ogc
```
Will be exported to the following directory, including `model_state.pdiparams`,  `model_state.pdiparams.info`, `model_state.pdmodel`and other files.

### 2.2 Python-based Inference

```bash
 python predict.py  \
 --text_encoder_type GroundingDino/groundingdino-swint-ogc \
 --model_path output_groundingdino/GroundingDino/groundingdino-swint-ogc \
 --input_image https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg \
 --output_dir ./groundingdino_predict_output \
 --prompt "bus"

```

## 3. BenchMark

> Note: 
> environment
Paddle 3.0
PaddleMIX release/2.0 
PaddleNLP 2.7.2
A100 80G。

### 3.1 benchmark cmd

Add -- benchmark after running in the 'deploy' corresponding model directory to obtain the running time of the model.
example: GroundingDino benchmark：

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

|Model|image size|dtype |Paddle Deploy |
|-|-|-|-|
|qwen-vl-7b|448*448|fp16|669.8 ms|
|llava-1.5-7b|336*336|fp16|981.2 ms|
|llava-1.6-7b|336*336|fp16|778.7 ms|
|groundingDino/groundingdino-swint-ogc|800*1193|fp32|100 ms|
|Sam/SamVitH-1024|1024*1024|fp32|121 ms|
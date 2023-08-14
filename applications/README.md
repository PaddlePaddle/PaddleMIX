**简体中文** | [English](./README_en.md)
<p align="center">
  <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/22989727/2cd19298-1c52-4d73-a0f7-dcdab6a8ec90" align="middle" width = "600" />
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleMIX/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleMIX?color=ccf"></a>
</p>

<h4 align="center">
  <a href=#特性> 特性 </a> |
  <a href=#安装> 安装 </a> |
  <a href=#快速开始> 快速开始 </a> |
</h4>

**PaddleMIX**应用示例基于paddlevlp、ppdiffusers和paddlenlp开发，**简单易用**且**功能强大**。聚合业界**优质预训练模型**并提供**开箱即用**的开发体验，覆盖跨模态和多场景的模型库搭配，可满足开发者**灵活定制**的需求。


## 特性

#### <a href=#开箱即用的工具集> 开箱即用的工具集 </a>

#### <a href=#跨模态多场景应用> 跨模态多场景应用 </a>



### 开箱即用的工具集

Appflow提供丰富的开箱即用工具集，覆盖跨模态多场景应用，提供产业级的效果与极致的推理性能。
```python
from paddlemix import Appflow
from PIL import Image
task = Appflow(app="openset_det_sam",
               models=["GroundingDino/groundingdino-swint-ogc","Sam/SamVitH-1024"]
               )
image_pil = Image.open("beauty.png").convert("RGB")
result = task(image=image_pil,prompt="women")
```


### 跨模态多场景应用
| 应用名称                           | 调用模型                         | 静态图推理   |
| :--------------------------------- | -------------------------------- | ----------|
| [开放世界检测分割（Openset-Det-Sam）](./CVinW/README.md/#开放世界检测分割grounded-sam-detect-and-segment-everything-with-text-prompt)              | `grounded sam`  |     ✅      |
| [自动标注（AutoLabel）](./Automatic_label/README.md/#自动标注autolabel)              | `blip2 grounded sam`        |      ✅       |
| [检测框引导的图像编辑（Det-Guided-Inpainting）](./Inpainting/README.md)      | `chatglm-6b stable-diffusion-2-inpainting grounded sam`                 |     ✅     |
| [文图生成（Text-to-Image Generation）](../ppdiffusers/README.md/#文图生成-text-to-image-generation)      | `stable-diffusion-2`  |         |
| [文本引导的图像放大（Text-Guided Image Upscaling）](../ppdiffusers/README.md/#文本引导的图像放大--超分text-guided-image-upscaling--super-resolution)           | `ldm-super-resolution-4x-openimages`|         |
| [文本引导的图像编辑（Text-Guided Image Inpainting）](../ppdiffusers/README.md/#文本引导的图像编辑text-guided-image-inpainting) | `stable-diffusion-2-inpainting`     |         |
| [文本引导的图像变换（Image-to-Image Text-Guided Generation）](../ppdiffusers/README.md/#文本引导的图像变换image-to-image-text-guided-generation)              | `stable-diffusion-v1-5`    |        |
| [文本图像双引导图像生成（Dual Text and Image Guided Generation）](../ppdiffusers/README.md/#文本图像多模)          | `versatile-diffusion`    |         |
| [文本条件的视频生成（Text-to-Video Generation）](../ppdiffusers/README.md/#文本视频多模)      | `text-to-video-ms-1.7b`  |         |

更多应用持续开发中......


## 安装

### 环境依赖

```
pip install -r requirements.txt
```
更多关于PaddlePaddle和PaddleNLP安装的详细教程请查看 [Installation](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/get_started/installation.rst)。

### 源码安装

```shell
git clone https://github.com/PaddlePaddle/PaddleMIX
python setup.py install
```
## 快速开始

这里以开放世界检测分割为例:

### 一键预测

PaddleMIX提供[一键预测功能]()，无需训练，直接输入数据即可输出结果：

```python
>>> from paddlemix.applications import Appflow
>>> from PIL import Image

>>> task = Appflow(task="openset_det_sam",
                   models=["GroundingDino/groundingdino-swint-ogc","Sam/SamVitH-1024"],
                   static_mode=False) #如果开启静态图推理，设置为True,默认动态图
>>> image_pil = Image.open("beauty.png").convert("RGB")
>>> result = task(image=image_pil,prompt="women")
```

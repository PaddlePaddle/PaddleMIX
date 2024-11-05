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
  <a href=#快速开始> 快速开始 </a>
</h4>



**PaddleMIX**应用示例基于paddlemix、ppdiffusers和paddlenlp开发，**简单易用**且**功能强大**。聚合业界**优质预训练模型**并提供**开箱即用**的开发体验，覆盖跨模态和多场景的模型库搭配，可满足开发者**灵活定制**的需求。

<img src="https://github.com/user-attachments/assets/4c695140-bf4c-46db-bbb5-5dd8197be947" align="center" />

## 快速开始

请先确认是否已安装 [PaddleMIX](../README.md/#安装) 和 [ppdiffusers](../README.md/#安装)

### 1. appflow 依赖安装
```shell
pip install -r paddlemix/appflow/requirements.txt
```


### 2.一键预测

PaddleMIX提供一键预测功能，无需训练，这里以开放世界检测分割为例。直接在终端运行如下命令，即可完成模型推理。

```python
>>> python
>>> from paddlemix.appflow import Appflow
>>> from ppdiffusers.utils import load_image

>>> task = Appflow(app="openset_det_sam",
                   models=["GroundingDino/groundingdino-swint-ogc","Sam/SamVitH-1024"],
                   static_mode=False) #如果开启静态图推理，设置为True,默认动态图
>>> url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
>>> image_pil = load_image(url)
>>> result = task(image=image_pil,prompt="dog")
```

参数说明
| 参数 | 是否必须| 含义                                                                                          |
|-------|-------|---------------------------------------------------------------------------------------------|
| --app | Yes| 应用名称                                                                                   |
| --models | Yes | 需要使用的模型，可以是单个模型，也可以多个组合                                                                                     |
| --static_mode  | Option | 是否静态图推理，默认False                                                                                 |
| --precision | Option | 当 static_mode == True 时使用，默认fp32,可选择trt_fp32、trt_fp16                                                                                    |


## 特性

#### <a href=#开箱即用的工具集> 开箱即用的工具集 </a>

#### <a href=#跨模态多场景应用> 跨模态多场景应用 </a>



### 开箱即用的工具集

Appflow提供丰富的开箱即用工具集，覆盖跨模态多场景应用，提供产业级的效果与极致的推理性能。

![appflow](https://github.com/LokeZhou/PaddleMIX/assets/13300429/f80a7aa0-4cd5-4f86-90d6-2fc6da3eb42f)




### 跨模态多场景应用
| 应用名称                           | 调用模型                         | 静态图推理    |
| :--------------------------------- | -------------------------------- | ----------|
| [视觉语言对话（Vision-Language-Chat）](./VLChat/README.md)              | `qwen-vl-chat-7b`  |     🚧     |
| [开放世界检测分割（Openset-Det-Sam）](./CVinW/README.md/#开放世界检测分割grounded-sam-detect-and-segment-everything-with-text-prompt)              | `grounded sam`  |     ✅      |
| [自动标注（AutoLabel）](./Automatic_label/README.md/#自动标注autolabel)              | `blip2 grounded sam`        |      ✅       |
| [检测框引导的图像编辑（Det-Guided-Inpainting）](./Inpainting/README.md/#检测框引导的图像编辑det-guided-inpainting)      | `chatglm-6b stable-diffusion-2-inpainting grounded sam`                 |     ✅     |
| [文图生成（Text-to-Image Generation）](./text2image/README.md/#文图生成text-to-image-generation)      | `runwayml/stable-diffusion-v1-5 stabilityai/stable-diffusion-xl-base-1.0`   |    [fastdeploy](../ppdiffusers/deploy/README.md/#文图生成text-to-image-generation)     |
| [文本引导的图像放大（Text-Guided Image Upscaling）](./image2image/README.md/#文本引导的图像放大text-guided-image-upscaling)           | `ldm-super-resolution-4x-openimages`|    ❌     |
| [文本引导的图像编辑（Text-Guided Image Inpainting）](./Inpainting/README.md/#文本引导的图像编辑text-guided-image-inpainting) | `stable-diffusion-2-inpainting`     |   [fastdeploy](../ppdiffusers/deploy/README.md/#文本引导的图像编辑text-guided-image-inpainting)     |
| [文本引导的图像变换（Image-to-Image Text-Guided Generation）](./image2image/README.md/#文本引导的图像变换image-to-image-text-guided-generation)              | `stable-diffusion-v1-5`    |    [fastdeploy](../ppdiffusers/deploy/README.md/#文本引导的图像变换image-to-image-text-guided-generation)    |
| [文本条件的视频生成（Text-to-Video Generation）](./text2video/README.md/#文本条件的视频生成text-to-video-generation)      | `text-to-video-ms-1.7b`  |     ❌     |
| [音频生成图像（Audio-to-Image Generation）](./Audio2Img/README.md/#audio-to-image)  | `imagebind stable-diffusion-2-1-unclip`  |          |
| [音频描述（Audio-to-Caption Generation）](./Audio2Caption/README.md/#音频描述audio-to-caption-generation)  | `chatglm-6b whisper`  |          |
| [音频对话（Audio-to-Chat Generation）](./AudioChat/README.md/#音频对话audio-to-chat-generation)  | `chatglm-6b whisper fastspeech2`  |          |
| [音乐生成（Music Generation）](./MusicGeneration/README.md/#音乐生成music-generation)  | `chatglm-6b minigpt4 audioldm`  |          |

更多应用持续开发中......

* ✅: Supported
* 🚧: In Progress
* ❌: Not Supported

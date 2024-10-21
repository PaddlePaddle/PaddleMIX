**English** | [简体中文](./README.md)
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
  <a href=#Features> Features </a> |
  <a href=#quick-start> Quick Start </a> |
</h4>

**PaddleMIX** application example is developed based on paddlemix, ppdiffusers, and Paddlenlp，which is **simple** and **easy** to use  and **powerful**. Aggregating industry high-quality pre trained models and providing out of the box development experience, covering cross modal and multi scenario model library matching, can meet the needs of developers flexible customization .

<img src="https://github.com/user-attachments/assets/4c695140-bf4c-46db-bbb5-5dd8197be947" align="center" />


## Quick Start
Please confirm if it has been installed first [PaddleMIX](../README_EN.md/#installation) and [ppdiffusers](../README_EN.md/#installation)

### 1.requirements
```shell
pip install -r paddlemix/appflow/requirements.txt
```

### 2.Appflow

PaddleMIX provides Appflow without training, and can directly input data to output results:

```
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

Parameter Description
| parameter | required| meaning                                                                                          |
|-------|-------|---------------------------------------------------------------------------------------------|
| --app | Yes| app name                                                                                   |
| --models | Yes | model list,can be a single model or multiple combinations                               |
| --static_mode  | Option | static graph inference, default : False                                          |
| --precision | Option | when static_mode == True used，default: fp32, option trt_fp32、trt_fp16                                                                                    |

## Features

#### <a href=#out-of-box-toolset> Out-of-Box Toolset </a>

#### <a href=#multi-modal-and-scenario> Multi Modal And Scenario </a>



### Out-of-Box Toolset

Appflow provides a rich set of out of the box tools that cover cross modal and multi scenario applications, providing industry level effects and ultimate reasoning performance.
![appflow](https://github.com/LokeZhou/PaddleMIX/assets/13300429/f80a7aa0-4cd5-4f86-90d6-2fc6da3eb42f)

### Multi Modal And Scenario
| name                           | models                         | static mode    |
| :--------------------------------- | -------------------------------- | ----------|
| [视觉语言对话（Vision-Language-Chat）](./VLChat/README.md)              | `qwen-vl-chat-7b`  |     🚧     |
| [开放世界检测分割（Openset-Det-Sam）](./CVinW/README.md/#开放世界检测分割grounded-sam-detect-and-segment-everything-with-text-prompt)              | `grounded sam`  |     ✅      |
| [自动标注（AutoLabel）](./Automatic_label/README.md/#自动标注autolabel)              | `blip2 grounded sam`        |      ✅       |
| [检测框引导的图像编辑（Det-Guided-Inpainting）](./Inpainting/README.md/#检测框引导的图像编辑det-guided-inpainting)      | `chatglm-6b stable-diffusion-2-inpainting grounded sam`                 |     ✅     |
| [文图生成（Text-to-Image Generation）](./text2image/README.md/#文图生成text-to-image-generation)      | `runwayml/stable-diffusion-v1-5`   |    [fastdeploy](../ppdiffusers/deploy/README.md/#文图生成text-to-image-generation)     |
| [文本引导的图像放大（Text-Guided Image Upscaling）](./image2image/README.md/#文本引导的图像放大text-guided-image-upscaling)           | `ldm-super-resolution-4x-openimages`|    ❌     |
| [文本引导的图像编辑（Text-Guided Image Inpainting）](./Inpainting/README.md/#文本引导的图像编辑text-guided-image-inpainting) | `stable-diffusion-2-inpainting`     |   [fastdeploy](../ppdiffusers/deploy/README.md/#文本引导的图像编辑text-guided-image-inpainting)     |
| [文本引导的图像变换（Image-to-Image Text-Guided Generation）](./image2image/README.md/#文本引导的图像变换image-to-image-text-guided-generation)              | `stable-diffusion-v1-5`    |    [fastdeploy](../ppdiffusers/deploy/README.md/#文本引导的图像变换image-to-image-text-guided-generation)    |
| [文本条件的视频生成（Text-to-Video Generation）](./text2video/README.md/#文本条件的视频生成text-to-video-generation)      | `text-to-video-ms-1.7b`  |     ❌     |


More applications under continuous development......

* ✅: Supported
* 🚧: In Progress
* ❌: Not Supported

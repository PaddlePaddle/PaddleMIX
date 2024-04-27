<p align="center">
  <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/22989727/2cd19298-1c52-4d73-a0f7-dcdab6a8ec90" align="middle" width = "600" />
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleMIX/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleMIX?color=ccf"></a>
  <br>
    <br>
    <a href="./README.md">中文文档</a>
</p>
</div>

## Introduction

PaddleMIX is a large multi-modal development kit based on PaddlePaddle, which aggregates multiple functions such as images, texts, and videos, and covers a variety of multi-modal tasks such as visual language pre-training, textual images, and textual videos. It provides an out-of-the-box development experience while meeting developers’ flexible customization needs and exploring general artificial intelligence.

## Updates
**2024.04.17**
* [PPDiffusers](./ppdiffusers/README.md) published version 0.24.0, it supports DiT and other Sora-related technologies. Supporting SVD and other video generation models

**2023.10.7**
* Published PaddleMIX version 1.0
* Newly added distributed training capability for image-text pre-training models. BLIP-2 supports training on scales up to one hundred billion parameters.
* Newly added cross-modal application pipeline [AppFlow](./applications/README.md), which supports automatic annotation, image editing, sound-to-image, and 11 other cross-modal applications with just one click.
* [PPDiffusers](./ppdiffusers/README.md) has released version 0.19.3, introducing SDXL and related tasks.

**2023.7.31**
* Published PaddleMIX version 0.1
* The PaddleMIX large multi-modal model development toolkit is released for the first time, integrating the PPDiffusers multi-modal diffusion model toolbox and widely supporting the PaddleNLP large-language models.
* Added 12 new large multi-modal models including EVA-CLIP, BLIP-2, miniGPT-4, Stable Diffusion, ControlNet, etc.

## Main Features

- **Rich Multi-Modal Functionality:** Encompassing image-text pre-training, text-to-image, multi-modal visual tasks, enabling diverse functions like image editing, image description, data annotation, and more.
- **Simplified Development Experience:** Unified model development interface facilitating efficient custom model development and feature implementation.
- **Efficient Training and Inference Workflow:** Streamlined end-to-end development process for training and inference, with standout performance in training and inference for key models such as BLIP-2, Stable Diffusion, etc., leading the industry.
- **Support for Ultra-Large Scale Training:** Capable of training models up to the scale of hundreds of billions for image-text pre-training, and base models up to the scale of tens of billions for text-to-image.

## Demo

- video Demo

https://github.com/PaddlePaddle/PaddleMIX/assets/29787866/8d32722a-e307-46cb-a8c0-be8acd93d2c8



## Installation

1. Environment Dependencies
```
pip install -r requirements.txt
```

Detailed [installation]((https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)) tutorials for PaddlePaddle

> Note: parts of Some models in ppdiffusers require CUDA 11.2 or higher. If your local machine does not meet the requirements, it is recommended to go to [AI Studio](https://aistudio.baidu.com/index) for model training and inference tasks.

> If you wish to train and infer using **bf16**, please use a GPU that supports **bf16**, such as the A100.

2. Manual Installation
```
git clone https://github.com/PaddlePaddle/PaddleMIX
cd PaddleMIX
pip install -e .

#ppdiffusers 安装
cd ppdiffusers
pip install -e .
```

## Tutorial

- [Quick Start](applications/README_en.md/#quick-start)
- [Fine-Tuning](paddlemix/tools/README.md)
- [Inference Deployment](deploy/README_en.md)

## Specialized Applications

1. Artistic Style QR Code Model

<div align="center">
<img src="https://github.com/PaddlePaddle/Paddle/assets/22989727/ba091291-a1ee-49dc-a1af-fc501c62bfc8" height = "300",caption='' />
<p>Try it out: https://aistudio.baidu.com/community/app/1339</p>
</div>

2. Image Mixing

<div align="center">
<img src="https://github.com/PaddlePaddle/Paddle/assets/22989727/a71be5a0-b0f3-4aa8-bc20-740ea8ae6785" height = "300",caption='' />
<p>Try it out: https://aistudio.baidu.com/community/app/1340</p>
</div>


## Datasets

<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>Multi-modal Pre-training</b>
      </td>
      <td>
        <b>Diffusion-based Models</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        </ul>
          <li><b>Image-Text Pre-training</b></li>
        <ul>
            <li><a href="paddlemix/examples/evaclip">EVA-CLIP</a></li>
            <li><a href="paddlemix/examples/coca">CoCa</a></li>
            <li><a href="paddlemix/examples/clip">CLIP</a></li>
            <li><a href="paddlemix/examples/blip2">BLIP-2</a></li>
            <li><a href="paddlemix/examples/minigpt4">miniGPT-4</a></li>
            <li><a href="paddlemix/examples/visualglm">VIsualGLM</a></li>
            <li><a href="paddlemix/examples/qwen_vl">qwen_vl</a></li>
            <li><a href="paddlemix/examples/llava">llava</a></li>
      </ul>
      </ul>
          <li><b>Open World Vision Models</b></li>
        <ul>
            <li><a href="paddlemix/examples/groundingdino">Grounding DINO</a></li>
            <li><a href="paddlemix/examples/sam">SAM</a></li>
      </ul>
      </ul>
          <li><b>More Multi-Modal Pre-trained Models</b></li>
        <ul>
            <li><a href="paddlemix/examples/imagebind">ImageBind</a></li>
      </ul>
      </td>
      <td>
        <ul>
        </ul>
          <li><b>Text-to-Image</b></li>
        <ul>
           <li><a href="ppdiffusers/examples/stable_diffusion">Stable Diffusion</a></li>
            <li><a href="ppdiffusers/examples/controlnet">ControlNet</a></li>
            <li><a href="ppdiffusers/examples/text_to_image_laion400m">LDM</a></li>
            <li><a href="ppdiffusers/ppdiffusers/pipelines/unidiffuser">Unidiffuser</a></li>
        </ul>
        </ul>
          <li><b>Text-to-Video</b></li>
        <ul>
           <li><a href="ppdiffusers/ppdiffusers/pipelines/lvdm">LVDM</a></li>
        </ul>
        </ul>
          <li><b>Audio Generation</b></li>
        <ul>
           <li><a href="ppdiffusers/ppdiffusers/pipelines/audioldm">AudioLDM</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

For more information on additional model capabilities, please refer to the [Model Capability Matrix](./paddlemix/examples/README.md).
## LICENSE

This repository is licensed under the [Apache 2.0 license](LICENSE)

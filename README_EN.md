简体中文 | [English](README_EN.md)

<p align="center">
  <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/22989727/2cd19298-1c52-4d73-a0f7-dcdab6a8ec90" align="middle" width = "600" />
</p>

<p align="center">
    <a href="https://github.com/PaddlePaddle/PaddleMix/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleMix?color=ffa"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="#📌社区交流"><img src="https://img.shields.io/badge/微信-小助手加群-green?logo=wechat&amp"></a>
    <a href="https://github.com/PaddlePaddle/PaddleMIX/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleMIX?color=ccf"></a>

</p>
</div>

## 💌 Table of Contents
- [💌 Table of Contents](#table-of-contents)
- [📰 News](#news)
- [📣 Latest Developments](#latest-developments)
- [🌈 Introduction](#introduction)
- [✨ Key Features](#key-features)
    - [📱 Rich Multimodal Capabilities](#rich-multimodal-capabilities)
    - [🧩 Simple Development Experience](#simple-development-experience)
    - [💡 High-Performance Distributed Training and Inference Capabilities](#high-performance-distributed-training-and-inference-capabilities)
    - [🔧 Unique Features and Tools](#unique-features-and-tools)
- [🔍 Installation](#installation)
- [🔥 Tutorials](#tutorials)
- [🤔 FAQ](#faq)
- [📱 Model Library](#model-library)
- [📝 License](#license)
- [📌 Community](#community)


## 📰 News
**🔥 Live Class on October 31, 2024，PaddleMIX release v2.1**

- 🎉 The PaddleMIX Multimodal Model Suite Development Competition release v2.1! Baidu's R&D engineers will provide a detailed explanation of the updated content, as well as the implementation details and case applications of the multi-modal data capability tagging model, PP-InsCapTagger, at 20:00 on Thursday, October 31st. Hurry up and scan the QR code on the poster below to register!

<div align="center">
<img src="https://github.com/user-attachments/assets/a32745a1-34bb-4096-a367-664ae58e3565" width="500px" align="middle"></img>
</div>



## 📣 Latest Developments

**🎉 2024.10.31 Welcome to the Update of External Developer's Creative [Tutorial Page](paddlemix_applications.md)**
* 🌟 Since the launch of our Large Model Suite Premium Project Collection activity on September 6th, we have received 30 high-quality developer projects. Among them, 25 premium projects have successfully passed the platform evaluation and been featured.

* 🙏 We sincerely thank all developers for their wonderful creations based on our suite! 🚀 We cordially invite you to share your creativity as well - welcome to publish your tutorials on public web pages or in the [PaddlePaddle AI Studio](https://aistudio.baidu.com/aistudio/community/multimodal?from=singlemessage) community!

**🔥 PaddleMIX v2.1 Released on 2024.10.11**
* Supports the [PaddleNLP 3.0 beta](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v3.0.0-beta0) version, allowing early access to its latest features.
* Added cutting-edge models like [Qwen2-VL](./paddlemix/examples/qwen2_vl/), [InternVL2](./paddlemix/examples/internvl2/), and [Stable Diffusion 3 (SD3)](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/examples/dreambooth/README_sd3.md).
* Released our self-developed multimodal data capability tagging model [PP-InsCapTagger](./paddlemix/datacopilot/example/pp_inscaptagger/), which can be used for data analysis and filtering. Experimental cases show that it can reduce data volume by 50% while maintaining model performance, significantly improving training efficiency.

* The multimodal large models InternVL2, LLaVA, SD3, and SDXL are now adapted to the Ascend 910B, offering training and inference capabilities on domestic computing chips.


**PaddleMIX v2.0 Released on 2024.07.25**
* Multimodal Understanding: Added LLaVA series, Qwen-VL, etc.; introduced Auto module to unify the SFT training process; introduced Mixtoken training strategy, increasing SFT throughput by 5.6 times.
* Multimodal Generation: Released [PPDiffusers 0.24.1](./ppdiffusers/README.md), supporting video generation capabilities, and added LCM to the text-to-image model. Also added a PaddlePaddle version of PEFT and the Accelerate backend. Provided a ComfyUI plugin developed with PaddlePaddle.
* Multimodal Data Processing Toolbox [DataCopilot](./paddlemix/datacopilot/): Supports custom data structures, data transformation, and offline format checks. Includes basic statistical information and data visualization functionality.

**PaddleMIX v1.0 Released on 2023.10.7**
* Added distributed training capabilities for vision-language pre-training models, and BLIP-2 now supports trillion-scale training.
* Introduced the cross-modal application pipeline [AppFlow](./applications/README.md), which supports 11 cross-modal applications such as automatic annotation, image editing, and audio-to-image with one click.
* [PPDiffusers](./ppdiffusers/README.md) released version 0.19.3, adding SDXL and related tasks.
---

## 🌈 Introduction

PaddleMIX is a multimodal large model development suite based on PaddlePaddle, integrating various modalities such as images, text, and video. It covers a wide range of multimodal tasks, including vision-language pre-training, fine-tuning, text-to-image, text-to-video, and multimodal understanding. It offers an out-of-the-box development experience while supporting flexible customization to meet diverse needs, empowering the exploration of general artificial intelligence.

<p align="center">
  <img src="https://github.com/user-attachments/assets/764b32a4-3933-4ef8-a0b2-dd425af49ef8" align="middle" width = 100% />
</p>

The PaddleMIX toolchain includes data processing, model development, pre-training, fine-tuning, and inference deployment, supporting mainstream multimodal models such as EVA-CLIP, BLIP-2, and Stable Diffusion. With cross-modal task pipelines like AppFlow and text-to-image application pipelines, developers can quickly build multimodal applications.

### An example of multimodal understanding is shown below:

<img src="https://github.com/user-attachments/assets/4c9a0427-57c7-4e1b-80f0-428c03119cc3"></img>


Multimodal understanding 🤝 integrates visual 👀 and linguistic 💬 processing capabilities. It includes functions such as basic perception, fine-grained image understanding, and complex visual reasoning 🧠. Our [Model Library](#model-library) offers practical applications for single-image, multi-image, and video inference. Features include natural image summarization 📝, question answering 🤔, OCR 🔍, sentiment recognition ❤️😢, specialized image analysis 🔬, and code interpretation 💻. These technologies can be applied in various fields such as education 📚, healthcare 🏥, industry 🏭, and more, enabling comprehensive intelligent analysis from static images 🖼️ to dynamic videos 🎥. We invite you to experience and explore these capabilities!

### An example of multimodal generation is shown below:

<div style="display: flex; justify-content: center; gap: 5px;">
    <img src="https://github.com/user-attachments/assets/f4768f08-f7a3-45e0-802c-c91554dc5dfc" style="height: 250px; object-fit: fill;">
    <img src="https://github.com/user-attachments/assets/9bf4a333-af57-4ddd-a514-617dea8da435" style="height: 250px; object-fit: fill;">
</div>

Multimodal generation ✍️ combines the creative power of text 💬 and visuals 👀. It includes various technologies ranging from text-to-image 🖼️ to text-to-video 🎥, featuring advanced models like Stable Diffusion 3 and Open-Sora. We provide practical applications for single-image generation, multi-image synthesis, and video generation in [ppdiffusers](ppdiffusers/README.md). These features cover areas such as artistic creation 🎨, animation production 📽️, and content generation 📝. With these technologies, creative generation from static images to dynamic videos can be applied in fields like education 📚, entertainment 🎮, advertising 📺, and more. We invite you to experience and explore these innovations!

### Example of featured applications (click the titles for a quick jump to the online experience):
|                                                  [**ComfyUI Creative Workflow**](https://aistudio.baidu.com/community/app/106043)                                                  |                                                [**Art Style QR Code Model**](https://aistudio.baidu.com/community/app/1339)                                                |                                                  [**Mix Image Overlay**](https://aistudio.baidu.com/community/app/1340)                                                  |
| :--------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------: |
| <img src='https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/36ba7261-1744-41a4-b1cb-c9e99f6931f2' width="300px"> | <img src='https://github.com/PaddlePaddle/Paddle/assets/22989727/ba091291-a1ee-49dc-a1af-fc501c62bfc8'  width="300px"> | <img src='https://github.com/PaddlePaddle/Paddle/assets/22989727/a71be5a0-b0f3-4aa8-bc20-740ea8ae6785'  width="300px"> |
|                                                  [**Anime Text-to-Image**](https://aistudio.baidu.com/community/app/2/webUI?source=appCenter)                                                   |                                                     [**AI Art｜50+ Lora Style Overlays**](https://aistudio.baidu.com/community/app/2848/webUI?source=appCenter)                                                     |                                               [**ControlNet｜Partial Image Repainting**](https://aistudio.baidu.com/community/app/1981/webUI?source=appCenter)                                               |
| <img src='https://github.com/user-attachments/assets/a4af8f8a-08c7-4da7-8575-9dbfedaba56c' width="200px"> | <img src='https://github.com/user-attachments/assets/fa92c229-a885-46a1-b23f-a076855c93ec'  width="200px"> | <img src='https://github.com/user-attachments/assets/78625876-d8ec-4c15-ae96-655c50f562ab'  width="200px"> |





-----


## ✨ Key Features

### 📱 Rich Multimodal Capabilities
PaddleMIX supports a wide range of the latest mainstream algorithm benchmarks and pre-trained models, covering vision-language pre-training, text-to-image, cross-modal visual tasks, and enabling diverse functionalities such as image editing, image description, and data annotation. `Gateway`: [📱 Model Library](#model-library)

### 🧩 Simple Development Experience
PaddleMIX provides a unified model development interface, allowing developers to quickly integrate and customize models. With the Auto module, users can efficiently load pre-trained models, perform tokenization, and easily complete model training, fine-tuning (SFT), inference, and deployment through a simplified API. Additionally, the Auto module supports developers in customizing automated model integration, ensuring flexibility and scalability while enhancing development efficiency.

### 💡 High-Performance Distributed Training and Inference Capabilities
PaddleMIX offers high-performance distributed training and inference capabilities, integrating acceleration operators like ✨Fused Linear✨ and ✨Flash Attention✨. It supports 🌀BF16 mixed-precision training and 4D mixed-parallel strategies. By optimizing inference performance through convolution layout, GroupNorm fusion, and rotating positional encoding optimization, it significantly enhances large-scale pre-training and efficient inference performance.

<img src="https://github.com/user-attachments/assets/9ab9540a-fa89-41cb-838d-95df86e33382" width = 100% />

### 🔧 Unique Features and Tools
The multimodal data processing toolbox, DataCopilot, accelerates model iteration and upgrades. It allows developers to perform basic data operations with low code based on specific tasks. `Gateway`: [🏆 Featured Models | Tools](#featured-models-tools)


## 🔍 Installation
### 1. Clone the PaddleMIX Repository

```
git clone https://github.com/PaddlePaddle/PaddleMIX
cd PaddleMIX
```

### 2. Create a Virtual Environment

```
conda create -n paddlemix python=3.10 -y
conda activate paddlemix
```
### 3. Install PaddlePaddle

#### Method 1: One-Click Installation (Recommended for GPU/CPU)

- CUDA 11.x or 12.3
- PaddlePaddle 3.0.0b1

```
sh build_paddle_env.sh
```

#### Method 2: Manual Installation
For detailed instructions on installing PaddlePaddle, please refer to the [Installation Guide](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html).

### 4. Ascend Environment Installation (Optional)

Currently, PaddleMIX supports the Ascend 910B chip (more models are in progress; if you have other model requirements, please submit an issue to let us know). The Ascend driver version is 23.0.3. Considering the variability in environments, we recommend using the standard image provided by PaddlePaddle to prepare your environment.

* Refer to the command below to start the container; `ASCEND_RT_VISIBLE_DEVICES` specifies the visible NPU card numbers.


```shell
docker run -it --name paddle-npu-dev -v $(pwd):/work \
    --privileged --network=host --shm-size=128G -w=/work \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    registry.baidubce.com/device/paddle-npu:cann80T13-ubuntu20-$(uname -m)-gcc84-py39 /bin/bash
```

* Install PaddlePaddle inside the container

```shell
# Note: You need to install the CPU version of PaddlePaddle first. Currently, only Python 3.9 is supported.
python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python -m pip install --pre paddle-custom-npu -i https://www.paddlepaddle.org.cn/packages/nightly/npu/
```


### 5. Install Dependencies

#### Method 1: One-Click Installation (Recommended)

Run the following command to automatically install all necessary dependencies:

```
sh build_env.sh
```

#### Method 2: Manual Installation (Please refer to build_env.sh)
## 🔥 Tutorials

**Quick Start**
- [Multimodal Understanding: Beginner's Experience](paddlemix/examples/internvl2/README.md)
- [Multimodal Generation: Zero-Basics Getting Started Guide](ppdiffusers/examples/inference/README.md)
- [Cross-Modal Task Pipeline: End-to-End Process Demonstration](applications/README.md/#quick-start)

**Hands-On Practice & Examples**
- [LLaVA Model: Full Process Practice from Training to Inference](https://aistudio.baidu.com/projectdetail/7917712)
- [SDXL Application: Create Your Own Olympic Poster Generator](https://aistudio.baidu.com/projectdetail/8251202)
- [PaddleMIX Multimodal AI Applications: Project Classification Overview](./paddlemix_applications.md)

**Multi-Hardware Usage**
- For the model list and usage supported by Ascend 910B, please refer to [Ascend Hardware Usage](./docs/hardware_support/ascend_usage.md)

**Data Preparation & Fine-Tuning**
- [Model Training and Fine-Tuning Techniques](paddlemix/tools/README.md)

**Inference Deployment**
- [Deployment Guide: From Development to Production Environment](deploy/README.md)



## 📱 Model Library
<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>Multimodal Understanding</b>
      </td>
      <td>
        <b>Multimodal Generation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        </ul>
          <li><b>Image-Text Pre-training</b></li>
        <ul>
            <li><a href="paddlemix/examples/clip">CLIP</a></li>
            <li><a href="paddlemix/examples/evaclip">EVA-CLIP</a></li>
            <li><a href="paddlemix/examples/llava">LLaVA</a></li>
            <li><a href="paddlemix/examples/llava">LLaVA-1.5</a></li>
            <li><a href="paddlemix/examples/llava">LLaVA-1.6</a></li>
            <li><a href="paddlemix/examples/llava">LLaVA-NeXT</a></li>
            <li><a href="paddlemix/examples/qwen_vl">Qwen-VL</a></li>
            <li><a href="paddlemix/examples/qwen2_vl">Qwen2-VL</a></li>
            <li><a href="paddlemix/examples/internvl2">InternVL2</a></li>
            <li><a href="paddlemix/examples/minimonkey">Mini-Monkey</a></li>
            <li><a href="paddlemix/examples/coca">CoCa</a></li>
            <li><a href="paddlemix/examples/blip2">BLIP-2</a></li>
            <li><a href="paddlemix/examples/minigpt4">miniGPT-4</a></li>
            <li><a href="paddlemix/examples/visualglm">VIsualGLM</a></li>
            <li><a href="paddlemix/examples/cogvlm">CogVLM && CogAgent</a></li>
            <li><a href="paddlemix/examples/internlm_xcomposer2">InternLM-XComposer2</a></li>
      </ul>
      </ul>
          <li><b>Open-World Visual Model</b></li>
        <ul>
            <li><a href="paddlemix/examples/groundingdino">Grounding DINO</a></li>
            <li><a href="paddlemix/examples/sam">SAM</a></li>
            <li><a href="paddlemix/examples/YOLO-World">YOLO-World</a></li>
      </ul>
      </ul>
          <li><b>More Multimodal Pre-trained Models</b></li>
        <ul>
            <li><a href="paddlemix/examples/imagebind">ImageBind</a></li>
      </ul>
      </ul>
        <li><b>Data Analysis</b></li>
      <ul>
          <li><a href="./paddlemix/datacopilot/example/pp_inscaptagger/">PP-InsCapTagger</a></li>
      </ul>
      </td>
      <td>
        <ul>
        </ul>
          <li><b>Text-to-Image</b></li>
        <ul>
           <li><a href="ppdiffusers/examples/stable_diffusion">Stable Diffusion</a></li>
           <li><a href="ppdiffusers/examples/dreambooth/README_sd3.md">Stable Diffusion 3 (SD3)</a></li>
            <li><a href="ppdiffusers/examples/controlnet">ControlNet</a></li>
            <li><a href="ppdiffusers/examples/t2i-adapter">T2I-Adapter</a></li>
            <li><a href="ppdiffusers/examples/text_to_image_laion400m">LDM</a></li>
            <li><a href="ppdiffusers/ppdiffusers/pipelines/unidiffuser">Unidiffuser</a></li>
            <li><a href="ppdiffusers/examples/class_conditional_image_generation/DiT">DiT</a></li>
            <li><a href="ppdiffusers/examples/HunyuanDiT">HunyuanDiT</a></li>
        </ul>
        </ul>
          <li><b>Text-to-Video</b></li>
        <ul>
           <li><a href="ppdiffusers/examples/lvdm">LVDM</a></li>
           <li><a href="ppdiffusers/examples/stable_video_diffusion">SVD</a></li>
           <li><a href="ppdiffusers/examples/AnimateAnyone">AnimateAnyone</a></li>
           <li><a href="ppdiffusers/examples/Open-Sora">OpenSora</a></li>
        </ul>
        </ul>
          <li><b>Audio Generation</b></li>
        <ul>
           <li><a href="ppdiffusers/ppdiffusers/pipelines/audioldm">AudioLDM</a></li>
           <li><a href="ppdiffusers/ppdiffusers/pipelines/audioldm2">AudioLDM2</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

For more model capabilities, please refer to the [Model Capability Matrix](./paddlemix/examples/README.md)

## 🏆 Featured Models | Tools

### 💎 Cross-Modal Task Pipeline AppFlow
<details>
<summary><b> Introduction (Click to Expand)</b></summary>

AppFlow, as the cross-modal application task pipeline of PaddleMIX, possesses powerful functionality and ease of use. By integrating cutting-edge algorithms such as LLaVA and Stable Diffusion, AppFlow has comprehensively covered various modalities including images, text, audio, and video. Through a flexible pipeline approach, it has constructed over ten multimodal applications, encompassing text-image generation, text-video generation, text-audio generation, image understanding, and more, providing users with rich demo examples. The highlight of AppFlow is its one-click prediction feature, allowing users to complete model inference with simple commands, eliminating cumbersome training and extensive coding, significantly lowering the barrier to use. Additionally, AppFlow fully leverages the dynamic-static unification advantages of the PaddlePaddle framework; users only need to set simple parameters to automatically complete model dynamic-to-static export and high-performance inference, enhancing work efficiency and optimizing model performance for one-stop application deployment.

`Gateway`: [Application Documentation Example](applications/README.md/#quick-start).

</details>

### 💎 Multimodal Data Processing Toolbox DataCopilot
<details>
<summary><b> Introduction (Click to Expand)</b></summary>

In real-world application scenarios, there is a substantial demand for fine-tuning multimodal large models using proprietary data to enhance model performance, making data elements the core of this process. Based on this, PaddleMIX provides the DataCopilot tool for data processing and analysis, allowing developers to achieve an end-to-end development experience within the PaddleMIX suite.

PP-InsCapTagger (Instance Capability Tagger) is a dataset capability tagging model implemented by DataCopilot based on PaddleMIX. It is used to label the capabilities of multimodal data instances. By optimizing the dataset through instance capability distribution, it can improve model training efficiency and provide an efficient solution for dataset analysis and evaluation. Combining the model inference labeling results with the LLaVA SFT dataset optimization can **improve LLaVA model training efficiency by 50% during the SFT phase.**

`Gateway`: [Application Documentation Example](paddlemix/datacopilot/readme.md).

</details>

<details>
<summary><b> PP-InsCapTagger (Click to Expand)</b></summary>

| Model                           | ScienceQA                               | TextVQA                                | VQAv2                                  | GQA                                    | MMMU                                   | MME                                     |
|----------------------------------|-----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|-----------------------------------------|
| llava-1.5-7b (origin)            | 66.8                                    | 58.2                                   | 78.5                                   | 62                                     | -                                      | -                                       |
| llava-1.5-7b (rerun)             | 69.01                                   | 57.6                                   | 79                                     | 62.95                                  | 36.89                                  | 1521<br>323                             |
| llava-1.5-7b (random 50%)        | 67.31                                   | 55.6                                   | 76.89                                  | 61.01                                  | 34.67                                  | 1421<br>286                             |
| **llava-1.5-7b (our 50%)**       | **70.24** *(+2.93)*                     | **57.12** *(+1.52)*                    | **78.32** *(+1.43)*                    | **62.14** *(+1.13)*                    | **37.11** *(+2.44)*                    | **1476** *(+55)*<br>**338** *(+52)*    |
`Gateway`: [Application Documentation Example](paddlemix/datacopilot/example/pp_inscaptagger/readme.md).
</details>

## 🤔 FAQ
For answers to some common questions about our project, please refer to the [FAQ](docs/FAQ.md). If your question is not addressed, feel free to raise it in the [Issues](https://github.com/PaddlePaddle/PaddleMIX/issues).

## 📝 License
This project is released under the [Apache 2.0 license](LICENSE).

## 📌 Community Communication

- Scan the QR code and fill out the questionnaire to join the communication group and engage deeply with numerous community developers and the official team.
<div align="center">
    <img src="https://github.com/user-attachments/assets/ecf292da-9ac6-41cb-84b6-df726ef4522d" width="300" height="300" />
</div>

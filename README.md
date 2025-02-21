简体中文 | [English](README_EN.md)

<p align="center">
  <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/22989727/2cd19298-1c52-4d73-a0f7-dcdab6a8ec90" align="middle" width = "600" />
</p>

<p align="center">
    <a href="https://github.com/PaddlePaddle/PaddleMix/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleMix?color=ffa"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux-pink.svg"></a>
    <a href="#📌社区交流"><img src="https://img.shields.io/badge/微信-小助手加群-green?logo=wechat&amp"></a>
    <a href="https://github.com/PaddlePaddle/PaddleMIX/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleMIX?color=ccf"></a>

</p>
</div>

## 💌目录
- [💌目录](#目录)
- [📰新闻](#新闻)
- [📣最新进展](#最新进展)
- [🌈简介](#简介)
- [✨主要特性](#主要特性)
    - [📱丰富的多模态功能](#丰富的多模态功能)
    - [🧩简洁的开发体验](#简洁的开发体验)
    - [💡高性能分布式训推能力](#高性能分布式训推能力)
    - [🔧特色功能与工具](#特色功能与工具)
- [🔍安装](#安装)
- [🔥教程](#教程)
- [🤔FAQ](#faq)
- [📱模型库](#模型库)
- [📝许可证书](#许可证书)
- [📌社区交流](#社区交流)


## 📰新闻
**🔥2025.01.16日直播课 飞桨PP系列模型上新！**

- 🎉PaddleMIX推出新一代统一视频生成控制模型PP-VCtrl！高效应用于人物动画、场景转换等视频生成精确控制任务。辅助条件编码器架构实现各类控制模块灵活接入，通过稀疏残差连接实现特征的高效控制传播，保持原训练视频扩散模型生成器架构不变，避免全量重训练。基于高质量可用数据集做recaption、人体关节点提取和视频分割，并采用多样化数据增强和训练策略，依次满足边缘控制、人体姿态与蒙版控制视频编辑任务的数据需求，显著提升适应能力和生成质量。控制能力和视频质量指标超越开源特定任务方法。**1月16日（周四）19:00**，直播为您详细解读PP-VCtrl的核心技术与产业应用。🚀报名链接：https://www.wjx.top/vm/m4sb0rh.aspx?udsid=664921

**🔥2025.01.07日直播课（已结束） 飞桨PP系列模型上新！**
- 🔗PaddleMIX的直播回放链接：[点击](https://aistudio.baidu.com/course/introduce/32178)
- 🎉PaddleMIX推出轻量级多模态文档理解模型PP-DocBee！基于多模态大模型实现端到端文档图像理解，破解业界复杂图表文档解析难题。采用ViT+MLP+LLM架构，优化数据合成策略、数据预处理、训练方式、OCR后处理辅助等环节。OCR小模型与LLM大模型结合、基于渲染引擎生产图像数据等众多策略实现问答质量更高，生成成本可控。支持本地gradio部署、OpenAI服务部署，提供飞桨星河社区在线环境快速体验。**1月7日（周二）19:00**，直播为您详细解读PP-DocBee的核心技术与产业应用。🚀报名链接：https://www.wjx.top/vm/mlDdpSb.aspx?udsid=309483


<details>
<summary>点击展开活动海报</summary>
<p align="center">
<img src='https://github.com/user-attachments/assets/5836c9df-4ea6-421b-acef-89f928e0763e'  width="80%">
</p>
</details>

## 📣最新进展

<!-- 📚《飞桨多模态大模型开发套件PaddleMIX 2.1 震撼发布》，图文音视频场景全覆盖，多模态高效助力产业创新。超大规模训练支持，覆盖图文预训练、文生图、跨模态视觉任务，覆盖金融、教育、电商、医疗等产业场景。8月8日（周四）20：00 带你直播了解多模态大模型最新架构，深度解析PaddleMIX高性能模型库，手把手演示LLaVA模型训推全流程。[报名链接](https://www.wjx.top/vm/wKqysjx.aspx?udsid=449688)   -->

**🎉 2025.01.20 喜迎外部开发者在星河(AIStudio)社区开发的[创作教程页面](paddlemix_applications.md)更新**
* 自9月6日，已收集星河(AIStudio)社区69个精品项目！快来动手体验应用开发吧～
* 新增「PaddleMIX套件体验官」和「飞桨PaddleMIX开发大赛」活动推出的22个精品项目，涵盖圣诞主题贺卡生成🎄、人物生成👤等多种应用，期待您的体验！此外，还有AI鉴宝🔍、图生音乐🎶等有趣的应用等待您来探索～
* 🙏 衷心感谢各位开发者基于套件的精彩创作！🚀 诚挚邀请您也来分享您的创意 - 欢迎将教程发布到公开网页或[飞桨AI Studio](https://aistudio.baidu.com/aistudio/community/multimodal?from=singlemessage)社区 

**🎉 2025.01.20 支持[Aria](./paddlemix/examples/aria)推理**

**🎉 2025.01.14 支持[deepseek-vl2](./paddlemix/examples/deepseek_vl2)推理**

**🎉 2025.01.02 新增自研文档理解模型[PP-DocBee](./paddlemix/examples/ppdocbee)推理和训练，支持[高性能推理](./deploy/ppdocbee)**

**🎉 2024.12.17 支持[GOT-OCR2_0](./paddlemix/examples/GOT_OCR_2_0)推理和训练**

**🎉 2024.12.17 支持[InternVL2_5(1B、2B、4B、8B)](./paddlemix/examples/internvl2)推理**

**🎉 2024.11.27 支持[Janus/JanusFlow](./paddlemix/examples/janus)推理**

**🎉 2024.11.21 支持[MiniCPM-V-2_6](./paddlemix/examples/minicpm-v-2_6)推理**

**🎉 2024.11.8 支持[DenseConnector](./paddlemix/examples/llava_denseconnector)和[Aquila-VL-2B-llava-qwen](./paddlemix/examples/llava_onevision/)推理**

**🎉 2024.11.1 支持[LLaVA-OneVision](./paddlemix/examples/llava_onevision/)和[LLaVA-Critic](./paddlemix/examples/llava_critic/)推理**


<details>
<summary>点击展开更多</summary>

**🎉 2024.10.31 喜迎外部开发者的[创作教程页面](paddlemix_applications.md)更新**

* 🌟 自9月6日发起大模型套件精品项目征集活动以来,我们收到了30个优质开发者项目,其中25个精品项目已通过平台评估并成功加精。

* 🙏 衷心感谢各位开发者基于套件的精彩创作！🚀 诚挚邀请您也来分享您的创意 - 欢迎将教程发布到公开网页或[飞桨AI Studio](https://aistudio.baidu.com/aistudio/community/multimodal?from=singlemessage)社区！

**🔥2024.10.11 发布PaddleMIX v2.1**
* 支持[PaddleNLP 3.0 beta](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v3.0.0-beta0)版本，抢先体验其最新功能。
* 新增[Qwen2-VL](./paddlemix/examples/qwen2_vl/)、[InternVL2](./paddlemix/examples/internvl2/)、[Stable Diffusion 3 (SD3)](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/examples/dreambooth/README_sd3.md)等前沿模型。
* 发布自研多模数据能力标签模型[PP-InsCapTagger](./paddlemix/datacopilot/example/pp_inscaptagger/)；可用于数据的分析和过滤，试验案例表明在保持模型效果的条件下可减少50%的数据量，大幅提高训练效率。

* 多模态大模型InternVL2、LLaVA、SD3、SDXL适配昇腾910B，提供国产计算芯片上的训推能力。


**2024.07.25 发布PaddleMIX v2.0**
* 多模态理解：新增LLaVA系列，Qwen-VL等；新增Auto模块统一SFT训练流程；新增mixtoken训练策略，SFT吞吐量提升5.6倍。
* 多模态生成：发布[PPDiffusers 0.24.1](./ppdiffusers/README.md)版本，支持视频生成能力，文生图模型新增LCM。新增飞桨版peft，accelerate后端。提供基于飞桨开发的ComfyUI插件。
* 多模态数据处理工具箱[DataCopilot](./paddlemix/datacopilot/)：支持自定义数据结构，数据转换，离线格式检查；支持基本的统计信息，数据可视化功能。

**2023.10.7 发布 PaddleMIX v1.0**
* 新增图文预训练模型分布式训练能力，BLIP-2支持千亿规模训练
* 新增跨模态应用流水线[AppFlow](./applications/README.md)，一键支持自动标注，图像编辑，音生图等11种跨模态应用
* [PPDiffusers](./ppdiffusers/README.md)发布 0.19.3 版本，新增SDXL及相关任务

</details>

---

## 🌈简介

PaddleMIX是基于飞桨的多模态大模型开发套件，聚合图像、文本、视频等多种模态，覆盖视觉语言预训练，微调，文生图，文生视频，多模态理解等丰富的多模态任务。它提供开箱即用的开发体验，同时支持灵活定制，满足不同需求，助力探索通用人工智能。

<p align="center">
  <img src="https://github.com/user-attachments/assets/764b32a4-3933-4ef8-a0b2-dd425af49ef8" align="middle" width = 100% />
</p>

PaddleMIX工具链包括数据处理、模型开发、预训练、精调和推理部署，支持主流多模态模型如 EVA-CLIP、BLIP-2、Stable Diffusion 等。通过跨模态任务流水线 AppFlow 和文生图应用 pipeline，开发者可以快速构建多模态应用。

### 多模态理解效果示例如下：

<img src="https://github.com/user-attachments/assets/4c9a0427-57c7-4e1b-80f0-428c03119cc3"></img>


多模态理解🤝融合了视觉👀和语言💬处理能力。包含基础感知、细粒度图像理解和复杂视觉推理🧠等功能。我们的[模型库](#模型库)调用提供了单图、多图和视频推理的功能实际应用，功能包括自然图像摘要📝、问答🤔、OCR🔍、情感识别❤️😢、专业图像分析🔬和代码解析💻。这些技术可应用于教育📚、医疗🏥、工业🏭等多个领域，实现从静态图像🖼️到动态视频🎥的全面智能分析。欢迎您的体验和探索～

### 多模态生成效果示例如下：
<div style="display: flex; justify-content: center; gap: 5px;">
    <img src="https://github.com/user-attachments/assets/f4768f08-f7a3-45e0-802c-c91554dc5dfc" style="height: 250px; object-fit: fill;">
    <img src="https://github.com/user-attachments/assets/9bf4a333-af57-4ddd-a514-617dea8da435" style="height: 250px; object-fit: fill;">
</div>


多模态生成✍️融合了文本💬与视觉👀的创造能力。涵盖了从文字生成图像🖼️到文字生成视频🎥的各类技术，包括 Stable Diffusion 3、Open-Sora等先进模型。我们在[ppdiffusers](ppdiffusers/README.md)提供了单图生成、多图合成和视频生成的实际应用，功能涉及艺术创作🎨、动画制作📽️、内容生成📝等。通过这些技术，可以在教育📚、娱乐🎮、广告📺等领域实现从静态图像到动态视频的创意生成。欢迎您的体验和探索～

### 特色应用效果示例如下（点击标题可快速跳转在线体验）：
|                                                  [**ComfyUI创作工作流**](https://aistudio.baidu.com/community/app/106043)                                                  |                                                [**艺术风格二维码模型**](https://aistudio.baidu.com/community/app/1339)                                                |                                                  [**Mix叠图**](https://aistudio.baidu.com/community/app/1340)                                                  |
| :--------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------: |
| <img src='https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/36ba7261-1744-41a4-b1cb-c9e99f6931f2' width="300px"> | <img src='https://github.com/PaddlePaddle/Paddle/assets/22989727/ba091291-a1ee-49dc-a1af-fc501c62bfc8'  width="300px"> | <img src='https://github.com/PaddlePaddle/Paddle/assets/22989727/a71be5a0-b0f3-4aa8-bc20-740ea8ae6785'  width="300px"> |
|                                                  [**二次元文生图**](https://aistudio.baidu.com/community/app/2/webUI?source=appCenter)                                                   |                                                     [**AI绘画｜50+Lora风格叠加**](https://aistudio.baidu.com/community/app/2848/webUI?source=appCenter)                                                     |                                               [**ControlNet｜图片局部重绘**](https://aistudio.baidu.com/community/app/1981/webUI?source=appCenter)                                               |
| <img src='https://github.com/user-attachments/assets/a4af8f8a-08c7-4da7-8575-9dbfedaba56c' width="200px"> | <img src='https://github.com/user-attachments/assets/fa92c229-a885-46a1-b23f-a076855c93ec'  width="200px"> | <img src='https://github.com/user-attachments/assets/78625876-d8ec-4c15-ae96-655c50f562ab'  width="200px"> |





-----








## ✨主要特性

### 📱丰富的多模态功能
PaddleMIX支持大量最新主流的算法基准以及预训练模型，覆盖图文预训练，文生图，跨模态视觉任务，实现图像编辑、图像描述、数据标注等多样功能。`传送门`：[📱模型库](#模型库)

### 🧩简洁的开发体验
PaddleMIX 提供统一的模型开发接口，支持开发者快速集成和定制模型。借助 Auto 模块，用户可以高效加载预训练模型、实现 Tokenization，并通过简化的 API 轻松完成模型的训练、微调（SFT）、推理与部署。此外，Auto 模块支持开发者自定义模型的自动化集成，确保灵活性与可扩展性，同时提升开发效率。

### 💡高性能分布式训推能力
PaddleMIX提供高性能分布式训练与推理能力，融合✨Fused Linear✨、✨Flash Attention✨等加速算子，支持🌀BF16混合精度训练和4D混合并行策略，并通过优化推理性能，包括卷积布局、GroupNorm融合及旋转位置编码优化，显著提升大规模预训练和高效推理性能。

<img src="https://github.com/user-attachments/assets/9ab9540a-fa89-41cb-838d-95df86e33382" width = 100% />



### 🔧特色功能与工具
多模态数据处理工具箱DataCopilot，加速模型迭代升级。让开发者根据特定任务以低代码量实现数据的基本操作。`传送门`：[🏆特色模型|工具](#特色模型工具)


## 🔍安装
### 1. 克隆PaddleMIX仓库
```
git clone https://github.com/PaddlePaddle/PaddleMIX
cd PaddleMIX
```

### 2. 创建虚拟环境
```
conda create -n paddlemix python=3.10 -y
conda activate paddlemix
```
### 3. ‼️安装PaddlePaddle

#### 方法 1: 一键安装（GPU/CPU推荐）

- CUDA 11.x或12.3
- PaddlePaddle 3.0.0b1
```
sh build_paddle_env.sh
```

#### 方法 2: 手动安装
关于PaddlePaddle安装的详细教程请查看[Installation](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)。

### 4. ‼️安装依赖

#### 方法 1: 一键安装（推荐）

运行以下命令来自动安装所有必要的依赖:
```
sh build_env.sh
```

#### 方法 2: 手动安装
```bash
# 安装 PaddleMIX
pip install -e .
# 安装 ppdiffusers
cd ppdiffusers
pip install -e .
cd ..
```
### 5. ‼️验证安装
```bash
sh check_env.sh

环境和依赖推荐版本:
- paddlepaddle: 3.0.0b2或develop版本
- paddlenlp: 3.0.0b2
- ppdiffusers: 0.29.0
- huggingface_hub: 0.23.0
```

### 6. 安装自定义算子（可选）
* 部分模型需要安装自定义算子（FastLayerNorm、FusedLayerNorm），例如EVA-CLIP、DIT_LLAMA等。
* 非CUDA环境（例如昇腾环境）则跳过
```bash
cd paddlemix/external_ops
python setup.py install
```

## 🔥教程

**快速开始**
- [多模态理解：新手入门体验 [示例：InternVL2模型]](paddlemix/examples/internvl2/README.md)
- [多模态生成：零基础上手指南 [示例：Stable Diffusion模型]](ppdiffusers/examples/stable_diffusion/README.md)
- [跨模态任务流水线：快速开始](applications/README.md/#快速开始)

**实操演练&范例**
- [LLaVA模型：从训练到推理的全流程实践](https://aistudio.baidu.com/projectdetail/7917712)
- [SDXL应用：打造专属奥运海报生成器](https://aistudio.baidu.com/projectdetail/8251202)
- [飞桨PaddleMIX跨模态AI应用：项目分类汇总](./paddlemix_applications.md)

**多硬件使用**
- 昇腾910B支持的模型列表和使用方式，可以参考[昇腾硬件使用](./docs/hardware_support/ascend_usage.md)


**数据准备&训练微调**
- [模型训练与微调技巧](paddlemix/tools/README.md)

**推理部署**
- [部署指南：从开发到生产环境](deploy/README.md)


## 📱模型库
<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>多模态理解</b>
      </td>
      <td>
        <b>多模态生成</b>
      </td>
      <td>
        <b>多模态大一统</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        </ul>
          <li><b>图文预训练</b></li>
        <ul>
            <li><a href="paddlemix/examples/clip">CLIP</a></li>
            <li><a href="paddlemix/examples/evaclip">EVA-CLIP</a></li>
            <li><a href="paddlemix/examples/llava">LLaVA-1.5</a></li>
            <li><a href="paddlemix/examples/llava">LLaVA-1.6</a></li>
            <li><a href="paddlemix/examples/llava">LLaVA-NeXT</a></li>
            <li><a href="paddlemix/examples/llava_onevision">LLaVA-onevision</a></li>
            <li><a href="paddlemix/examples/llava_onevision">Aquila-VL-2B-llava-qwen</a></li>
            <li><a href="paddlemix/examples/llava_critic">LLaVA-Critic</a></li>
            <li><a href="paddlemix/examples/llava_denseconnector">LLaVA-DenseConnector</a></li>
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
            <li><a href="paddlemix/examples/deepseek_vl2">DeepSeek-VL2</a></li> 
            <li><a href="paddlemix/examples/aria">Aria</a></li> 
        </ul>
      </ul>
          <li><b>开放世界视觉模型</b></li>
        <ul>
            <li><a href="paddlemix/examples/groundingdino">Grounding DINO</a></li>
            <li><a href="paddlemix/examples/sam">SAM</a></li>
            <li><a href="paddlemix/examples/sam2">SAM2</a></li>
            <li><a href="paddlemix/examples/YOLO-World">YOLO-World</a></li>
      </ul>
      </ul>
          <li><b>更多模态预训练模型</b></li>
        <ul>
            <li><a href="paddlemix/examples/imagebind">ImageBind</a></li>
      </ul>
      </td>
      <td>
        <ul>
        </ul>
          <li><b>文生图</b></li>
        <ul>
           <li><a href="ppdiffusers/examples/stable_diffusion">Stable Diffusion</a></li>
           <li><a href="ppdiffusers/ppdiffusers/pipelines/stable_diffusion_xl">SDXL</a></li>
           <li><a href="ppdiffusers/examples/dreambooth/README_sd3.md">Stable Diffusion 3 (SD3)</a></li>
            <li><a href="ppdiffusers/examples/controlnet">ControlNet</a></li>
            <li><a href="ppdiffusers/examples/t2i-adapter">T2I-Adapter</a></li>
            <li><a href="ppdiffusers/examples/text_to_image_laion400m">LDM</a></li>
            <li><a href="ppdiffusers/ppdiffusers/pipelines/consistency_models">Consistency Models</a></li>
            <li><a href="ppdiffusers/ppdiffusers/pipelines/deepfloyd_if">DeepFloyd IF</a></li>
            <li><a href="ppdiffusers/ppdiffusers/pipelines/shap_e">Shap-E</a></li>
            <li><a href="ppdiffusers/examples/kandinsky2_2">Kandinsky-2</a></li>
            <li><a href="ppdiffusers/ppdiffusers/pipelines/wuerstchen">Würstchen</a></li>
            <li><a href="ppdiffusers/ppdiffusers/pipelines/hotshot_xl">Hotshot-XL</a></li>
            <li><a href="ppdiffusers/ppdiffusers/pipelines/latent_consistency_models">LCMs</a></li>
            <li><a href="ppdiffusers/ppdiffusers/pipelines/unidiffuser">Unidiffuser</a></li>
            <li><a href="ppdiffusers/examples/class_conditional_image_generation/DiT">DiT</a></li>
            <li><a href="ppdiffusers/examples/HunyuanDiT">HunyuanDiT</a></li>
        </ul>
        </ul>
          <li><b>文生视频</b></li>
        <ul>
           <li><a href="ppdiffusers/examples/text_to_video_lvdm">LVDM</a></li>
           <li><a href="ppdiffusers/examples/stable_video_diffusion">SVD</a></li>
           <li><a href="ppdiffusers/examples/AnimateAnyone">AnimateAnyone</a></li>
           <li><a href="ppdiffusers/examples/Open-Sora">OpenSora</a></li>
           <li><a href="ppdiffusers/ppdiffusers/pipelines/animatediff">AnimateDiff</a></li>
           <li><a href="ppdiffusers/ppdiffusers/pipelines/text_to_video_synthesis">zeroscope_v2_XL</a></li>
           <li><a href="ppdiffusers/examples/cogvideo">CogVideoX</a></li>
        </ul>
        </ul>
          <li><b>音频生成</b></li>
        <ul>
           <li><a href="ppdiffusers/ppdiffusers/pipelines/audioldm">AudioLDM</a></li>
           <li><a href="ppdiffusers/ppdiffusers/pipelines/audioldm2">AudioLDM2</a></li>
        </ul>
      </td>
      <td>
        <ul>
        </ul>
          <li><b>统一多模态模型</b></li>
        <ul>
          <li><a href="paddlemix/examples/janus">Janus</a></li>
          <li><a href="paddlemix/examples/showo">Showo</a></li>
          <li><a href="paddlemix/examples/emu3">Emu3</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

更多模型能力，可参考[模型能力矩阵](./paddlemix/examples/README.md)



## 📊多模数据处理工具箱DataCopilot
<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>基础能力</b>
      </td>
      <td>
        <b>数据分析</b>
      </td>
      <td>
        <b>数据生成</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        </ul>
          <li><b>使用文档</b></li>
        <ul>
            <li><a href="paddlemix/datacopilot">DataCopilot</a></li>
      </ul>
      </td>
      <td>
        <ul>
        </ul>
          <li><b>能力标签模型</b></li>
        <ul>
           <li><a href="paddlemix/datacopilot/example">PP-InsCapTagger</a></li>
        </ul>
      </td>
      <td>
        <ul>
        </ul>
          <li><b>文档类数据生成方案</b></li>
        <ul>
          <li><a href="paddlemix/datacopilot/example">PP-InfinityDocData</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

更多数据相关功能，可参考[DataCopilot](./paddlemix/datacopilot)主页


## 🏆特色模型|工具

### 💎跨模态任务流水线AppFlow
<details>
<summary><b> 简介(点击展开)</b></summary>

AppFlow作为PaddleMIX的跨模态应用任务流水线，具备强大的功能与易用性。通过接入LLaVA、Stable Diffusion等前沿算法，AppFlow已全面覆盖图像、文本、音频、视频等多种模态，并通过流水线式的灵活组合，构建了10余种多模态应用，涵盖图文生成、文本视频生成、文本音频生成、图像理解等多个方面，为用户提供丰富的demo示例。AppFlow的特色在于其一键预测功能，用户无需繁琐训练与大量编码，仅需简单命令即可完成模型推理，极大地降低了使用门槛。同时，AppFlow充分利用飞桨框架动静统一优势，用户只需设置简单参数，即可自动完成模型的动转静导出及高性能推理，提高工作效率并优化模型性能，实现一站式应用部署。

`传送门`：[应用文档示例](applications/README.md/#快速开始)。

</details>

### 💎多模态数据处理工具箱DataCopilot
<details>
<summary><b> 简介(点击展开)</b></summary>

在真实的应用场景有大量使用专有数据微调多模态大模型来提升模型效果的需求，此过程中数据要素成为核心。基于此PaddleMIX提供了数据处理和分析的工具DataCopilot，使开发者可在PaddleMIX套件完成端到端的开发体验。

PP-InsCapTagger(Instance Capability Tagger) 是 DataCopilot 基于 PaddleMIX 实现的数据集能力标签模型，用于为多模态数据实例能力打标，通过实例能力分布对数据集进行优化，可以提高模型训练效率，为数据集分析和评价提供了一种高效的方案。 结合模型推理打标结果对LLaVA SFT数据集进行优化，可以**提高LLaVA模型SFT阶段50%的训练效率。**

`传送门`：[应用文档示例](paddlemix/datacopilot/readme.md)。

</details>

<details>
<summary><b> PP-InsCapTagger(点击展开)</b></summary>

| Model                           | ScienceQA                               | TextVQA                                | VQAv2                                  | GQA                                    | MMMU                                   | MME                                     |
|----------------------------------|-----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|-----------------------------------------|
| llava-1.5-7b (origin)            | 66.8                                    | 58.2                                   | 78.5                                   | 62                                     | -                                      | -                                       |
| llava-1.5-7b (rerun)             | 69.01                                   | 57.6                                   | 79                                     | 62.95                                  | 36.89                                  | 1521<br>323                             |
| llava-1.5-7b (random 50%)        | 67.31                                   | 55.6                                   | 76.89                                  | 61.01                                  | 34.67                                  | 1421<br>286                             |
| **llava-1.5-7b (our 50%)**       | **70.24** *(+2.93)*                     | **57.12** *(+1.52)*                    | **78.32** *(+1.43)*                    | **62.14** *(+1.13)*                    | **37.11** *(+2.44)*                    | **1476** *(+55)*<br>**338** *(+52)*    |


`传送门`：[应用文档示例](paddlemix/datacopilot/example/pp_inscaptagger/readme.md)。
</details>


## 🤔FAQ
关于我们项目的一些常见问题解答，请参考[FAQ](docs/FAQ.md)。如果您的问题没有得到解答，请随时在[Issues](https://github.com/PaddlePaddle/PaddleMIX/issues)中提出


## ❤️致谢

- PaddleMIX的部分模块和案例设计借鉴了 Hugging Face 的[Transformers](https://github.com/huggingface/transformers)🤗关于预训练模型使用的优秀设计，在此对 Hugging Face 作者及其开源社区表示感谢。

- PaddleMIX 的部分案例和代码由以下优秀社区开发者贡献，（完整的贡献者请参考: [Contributors](https://github.com/PaddlePaddle/PaddleMIX/graphs/contributors)）：
    [co63oc](https://github.com/co63oc)，
    [CrazyBoyM](https://github.com/CrazyBoyM)，
    [KPCOFGS](https://github.com/KPCOFGS)，
    [pkhk-1](https://github.com/pkhk-1)，
    [1649759610](https://github.com/1649759610)，
    [DrRyanHuang](https://github.com/DrRyanHuang)，
    [zhiboniu](https://github.com/zhiboniu)，
    [cocoshe](https://github.com/cocoshe)，
    [sneaxiy](https://github.com/sneaxiy)，
    [yangrongxinuser](https://github.com/yangrongxinuser)，
    [cheng221](https://github.com/cheng221)，
    [Liyulingyue](https://github.com/Liyulingyue)，
    [zhoutianzi666](https://github.com/zhoutianzi666)，
    [Birdylx](https://github.com/Birdylx)，
    [FeixLiu](https://github.com/FeixLiu)，
    [Tsaiyue](https://github.com/Tsaiyue)，
    [fightfat](https://github.com/fightfat)，
    [warrentdrew](https://github.com/warrentdrew)，
    [swagger-coder](https://github.com/swagger-coder)
    ...

- 感谢星河(AIStudio)社区的项目达人们开发了众多有趣的应用，为PaddleMIX的发展创造了更多可能性。特别感谢以下活跃的项目达人（完整名单请参见[AIStudio 项目达人推荐榜](https://aistudio.baidu.com/projectoverview)）：
    [好想成为PPDE（已成为版）](https://aistudio.baidu.com/personalcenter/thirdview/2553954)，
    [旭_1994](https://aistudio.baidu.com/personalcenter/thirdview/9044961)，
    [knoka](https://aistudio.baidu.com/personalcenter/thirdview/2258742)，
    [魔术师](https://aistudio.baidu.com/personalcenter/thirdview/710848)，
    [非鱼子焉](https://aistudio.baidu.com/personalcenter/thirdview/91451)
    ...


## 📝许可证书

本项目的发布受[Apache 2.0 license](LICENSE)许可认证。

## 📌社区交流

- 微信扫描二维码并填写问卷，即可加入交流群与众多社区开发者以及官方团队深度交流。
<div align="center">
    <img src="https://github.com/user-attachments/assets/ecf292da-9ac6-41cb-84b6-df726ef4522d" width="300" height="300" />
</div>

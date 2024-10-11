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
    <a href="./README_EN.md">English Document</a>
</p>
</div>

## 简介

PaddleMIX是基于飞桨的多模态大模型开发套件，聚合图像、文本、视频等多种模态，覆盖视觉语言预训练，文生图，文生视频等丰富的多模态任务。提供开箱即用的开发体验，同时满足开发者灵活定制需求，探索通用人工智能。

## 最新进展

<!-- 📚《飞桨多模态大模型开发套件PaddleMIX 2.1 震撼发布》，图文音视频场景全覆盖，多模态高效助力产业创新。超大规模训练支持，覆盖图文预训练、文生图、跨模态视觉任务，覆盖金融、教育、电商、医疗等产业场景。8月8日（周四）20：00 带你直播了解多模态大模型最新架构，深度解析PaddleMIX高性能模型库，手把手演示LLaVA模型训推全流程。[报名链接](https://www.wjx.top/vm/wKqysjx.aspx?udsid=449688)   -->

**🔥2024.10.11 发布PaddleMIX v2.1**
* 支持[PaddleNLP 3.0 beta](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v3.0.0-beta0)版本，抢先体验其最新功能。
* 新增[Qwen2-VL](./paddlemix/examples/qwen2_vl/)、[InternVL2](./paddlemix/examples/internvl2/)、[Stable Diffusion 3 (SD3)](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/examples/dreambooth/README_sd3.md)等前沿模型。
* 发布自研多模数据能力标签模型[PP-InsCapTagger](./paddlemix/datacopilot/example/pp_inscaptagger/)；可用于数据的分析和过滤，试验案例表明在保持模型效果的条件下可减少50%的数据量，大幅提高训练效率。
* 多模态大模型InternVL2、LLaVA、SD3、SDXL适配昇腾910B，提供国产计算芯片上的训推能力。

**2024.09.11 更新**
* 新增Qwen2-VL、InternVL2、SD3等模型

**2024.07.25 发布PaddleMIX v2.0**
* 多模态理解：新增LLaVA系列，Qwen-VL等；新增Auto模块统一SFT训练流程；新增mixtoken训练策略，SFT吞吐量提升5.6倍。
* 多模态生成：发布[PPDiffusers 0.24.1](./ppdiffusers/README.md)版本，支持视频生成能力，文生图模型新增LCM。新增飞桨版peft，accelerate后端。提供基于飞桨开发的ComfyUI插件。
* 多模态数据处理工具箱[DataCopilot](./paddlemix/datacopilot/)：支持自定义数据结构，数据转换，离线格式检查；支持基本的统计信息，数据可视化功能。

**2023.10.7 发布 PaddleMIX v1.0**
* 新增图文预训练模型分布式训练能力，BLIP-2支持千亿规模训练
* 新增跨模态应用流水线[AppFlow](./applications/README.md)，一键支持自动标注，图像编辑，音生图等11种跨模态应用
* [PPDiffusers](./ppdiffusers/README.md)发布 0.19.3 版本，新增SDXL及相关任务

## 主要特性

- **丰富的多模态功能:** 覆盖图文预训练，文生图，跨模态视觉任务，实现图像编辑、图像描述、数据标注等多样功能
- **简洁的开发体验:** 模型统一开发接口，高效实现自定义模型开发和功能实现
- **高效的训推流程:** 全量模型打通训练推理一站式开发流程，BLIP-2，Stable Diffusion等重点模型训推性能业界领先
- **超大规模训练支持:** 可训练千亿规模图文预训练模型，百亿规模文生图底座模型

## 任务展示

- 视频Demo展示（video Demo）

https://github.com/PaddlePaddle/PaddleMIX/assets/29787866/8d32722a-e307-46cb-a8c0-be8acd93d2c8


## 安装步骤
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
### 3. 安装PaddlePaddle
#### 方法 1: 一键安装（推荐）
- CUDA 11.x或12.3
- PaddlePaddle 3.0.0b1
```
sh build_paddle_env.sh
```

#### 方法 2: 手动安装
关于PaddlePaddle安装的详细教程请查看[Installation](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)。


### 4. 安装依赖

#### 方法 1: 一键安装（推荐）

运行以下命令来自动安装所有必要的依赖:
```
sh build_env.sh
```

#### 方法 2: 手动安装（请参考 build_env.sh）

> 注：ppdiffusers部分模型需要依赖 CUDA 11.2 及以上版本，如果本地机器不符合要求，建议前往 [AI Studio](https://aistudio.baidu.com/index) 进行模型训练、推理任务。

> 如果希望使用**bf16**训练推理，请使用支持**bf16**的GPU，如A100。


## 教程

- [快速开始](applications/README.md/#快速开始)
- [训练微调](paddlemix/tools/README.md)
- [推理部署](deploy/README.md)

## 特色应用

1. ComfyUI创作工作流

<div align="center">
<img src="https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/36ba7261-1744-41a4-b1cb-c9e99f6931f2" height = "300",caption='' />
<p>体验专区: https://aistudio.baidu.com/community/app/106043</p>
</div>

2. 艺术风格二维码模型

<div align="center">
<img src="https://github.com/PaddlePaddle/Paddle/assets/22989727/ba091291-a1ee-49dc-a1af-fc501c62bfc8" height = "300",caption='' />
<p>体验专区: https://aistudio.baidu.com/community/app/1339</p>
</div>

3. Mix叠图

<div align="center">
<img src="https://github.com/PaddlePaddle/Paddle/assets/22989727/a71be5a0-b0f3-4aa8-bc20-740ea8ae6785" height = "300",caption='' />
<p>体验专区: https://aistudio.baidu.com/community/app/1340</p>
</div>

## 模型库

<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>多模态理解</b>
      </td>
      <td>
        <b>多模态生成</b>
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
            <li><a href="paddlemix/examples/llava">LLaVA</a></li>
            <li><a href="paddlemix/examples/llava">LLaVA-1.5</a></li>
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
          <li><b>开放世界视觉模型</b></li>
        <ul>
            <li><a href="paddlemix/examples/groundingdino">Grounding DINO</a></li>
            <li><a href="paddlemix/examples/sam">SAM</a></li>
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
            <li><a href="ppdiffusers/examples/controlnet">ControlNet</a></li>
            <li><a href="ppdiffusers/examples/t2i-adapter">T2I-Adapter</a></li>
            <li><a href="ppdiffusers/examples/text_to_image_laion400m">LDM</a></li>
            <li><a href="ppdiffusers/ppdiffusers/pipelines/unidiffuser">Unidiffuser</a></li>
            <li><a href="ppdiffusers/examples/class_conditional_image_generation/DiT">DiT</a></li>
            <li><a href="ppdiffusers/examples/HunyuanDiT">HunyuanDiT</a></li>
        </ul>
        </ul>
          <li><b>文生视频</b></li>
        <ul>
           <li><a href="ppdiffusers/examples/lvdm">LVDM</a></li>
           <li><a href="ppdiffusers/examples/stable_video_diffusion">SVD</a></li>
           <li><a href="ppdiffusers/examples/AnimateAnyone">AnimateAnyone</a></li>
           <li><a href="ppdiffusers/examples/Open-Sora">OpenSora</a></li>
        </ul>
        </ul>
          <li><b>音频生成</b></li>
        <ul>
           <li><a href="ppdiffusers/ppdiffusers/pipelines/audioldm">AudioLDM</a></li>
           <li><a href="ppdiffusers/ppdiffusers/pipelines/audioldm2">AudioLDM2</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

更多模型能力，可参考[模型能力矩阵](./paddlemix/examples/README.md)

## 社区交流

- 微信扫描二维码并填写问卷，即可加入交流群与众多社区开发者以及官方团队深度交流。
<div align="center">
    <img src="https://github.com/user-attachments/assets/ecf292da-9ac6-41cb-84b6-df726ef4522d" width="300" height="300" />
</div>

## 许可证书

本项目的发布受Apache 2.0 license许可认证。

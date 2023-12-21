<p align="center">
  <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/22989727/2cd19298-1c52-4d73-a0f7-dcdab6a8ec90" align="middle" width = "600" />
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleMIX/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleMIX?color=ccf"></a>
</p>
</div>

## 简介

PaddleMIX是基于飞桨的跨模态大模型开发套件，聚合图像、文本、视频等多种模态，覆盖视觉语言预训练，文生图，文生视频等丰富的跨模态任务。提供开箱即用的开发体验，同时满足开发者灵活定制需求，探索通用人工智能。

## 最新进展

**2023.10.7 发布 PaddleMIX v1.0**
* 新增图文预训练模型分布式训练能力，BLIP-2支持千亿规模训练
* 新增跨模态应用流水线[AppFlow](./applications/README.md)，一键支持自动标注，图像编辑，音生图等11种跨模态应用
* [PPDiffusers](./ppdiffusers/README.md)发布 0.19.3 版本，新增SDXL及相关任务

**2023.7.31 发布 PaddleMIX v0.1**
* 首次发布PaddleMIX跨模态大模型开发套件，融合PPDiffusers多模态扩散模型工具箱，广泛支持PaddleNLP大语言模型
* 新增EVA-CLIP，BLIP-2，miniGPT-4，Stable Diffusion，ControlNet等12个跨模态大模型

## 主要特性

- **丰富的多模态功能:** 覆盖图文预训练，文生图，跨模态视觉任务，实现图像编辑、图像描述、数据标注等多样功能
- **简洁的开发体验:** 模型统一开发接口，高效实现自定义模型开发和功能实现
- **高效的训推流程:** 全量模型打通训练推理一站式开发流程，BLIP-2，Stable Diffusion等重点模型训推性能业界领先
- **超大规模训练支持:** 可训练千亿规模图文预训练模型，百亿规模文生图底座模型

## 任务展示

- 视频Demo展示（video Demo）

https://github.com/PaddlePaddle/PaddleMIX/assets/29787866/8d32722a-e307-46cb-a8c0-be8acd93d2c8



## 安装

1. 环境依赖
```
pip install -r requirements.txt
```

关于PaddlePaddle安装的详细教程请查看[Installation](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)。

> 注：ppdiffusers部分模型需要依赖 CUDA 11.2 及以上版本，如果本地机器不符合要求，建议前往 [AI Studio](https://aistudio.baidu.com/index) 进行模型训练、推理任务。

> 如果希望使用**bf16**训练推理，请使用支持**bf16**的GPU，如A100。

2. 手动安装
```
git clone https://github.com/PaddlePaddle/PaddleMIX
cd PaddleMIX
pip install -e .

#ppdiffusers 安装
cd ppdiffusers
pip install -e .
```

## 教程

- [快速开始](applications/README.md/#快速开始)
- [训练微调](docs/train_tutorial.md)
- [推理部署](deploy/README.md)

## 特色应用

1. 艺术风格二维码模型

<div align="center">
<img src="https://github.com/PaddlePaddle/Paddle/assets/22989727/ba091291-a1ee-49dc-a1af-fc501c62bfc8" height = "300",caption='' />
<p>体验专区: https://aistudio.baidu.com/community/app/1339</p>
</div>

2. Mix叠图

<div align="center">
<img src="https://github.com/PaddlePaddle/Paddle/assets/22989727/a71be5a0-b0f3-4aa8-bc20-740ea8ae6785" height = "300",caption='' />
<p>体验专区: https://aistudio.baidu.com/community/app/1340</p>
</div>


## 模型库

<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>多模态预训练</b>
      </td>
      <td>
        <b>扩散类模型</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        </ul>
          <li><b>图文预训练</b></li>
        <ul>
            <li><a href="paddlemix/examples/evaclip">EVA-CLIP</a></li>
            <li><a href="paddlemix/examples/coca">CoCa</a></li>
            <li><a href="paddlemix/examples/clip">CLIP</a></li>
            <li><a href="paddlemix/examples/blip2">BLIP-2</a></li>
            <li><a href="paddlemix/examples/minigpt4">miniGPT-4</a></li>
            <li><a href="paddlemix/examples/visualglm">VIsualGLM</a></li>
            <li><a href="paddlemix/examples/qwen_vl">qwen_vl</a></li>
      </ul>
      </ul>
          <li><b>开放世界视觉模型</b></li>
        <ul>
            <li><a href="paddlemix/examples/groundingdino">Grounding DINO</a></li>
            <li><a href="paddlemix/examples/sam">SAM</a></li>
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
            <li><a href="ppdiffusers/examples/text_to_image_laion400m">LDM</a></li>
            <li><a href="ppdiffusers/ppdiffusers/pipelines/unidiffuser">Unidiffuser</a></li>
        </ul>
        </ul>
          <li><b>文生视频</b></li>
        <ul>
           <li><a href="ppdiffusers/ppdiffusers/pipelines/lvdm">LVDM</a></li>
        </ul>
        </ul>
          <li><b>音频生成</b></li>
        <ul>
           <li><a href="ppdiffusers/ppdiffusers/pipelines/audioldm">AudioLDM</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

更多模型能力，可参考[模型能力矩阵](./paddlemix/examples/README.md)
## 许可证书

本项目的发布受Apache 2.0 license许可认证。

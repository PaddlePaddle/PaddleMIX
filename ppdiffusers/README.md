<div align="center">
  <img src="https://user-images.githubusercontent.com/11793384/215372703-4385f66a-abe4-44c7-9626-96b7b65270c8.png" width="40%" height="40%" />
</div>

<p align="center">
    <a href="https://pypi.org/project/ppdiffusers/"><img src="https://img.shields.io/pypi/pyversions/ppdiffusers"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
</p>

<h4 align="center">
  <a href=#特性> 特性 </a> |
  <a href=#安装> 安装 </a> |
  <a href=#快速开始> 快速开始 </a> |
  <a href=#模型部署> 模型部署</a>
</h4>

# PPDiffusers: Diffusers toolbox implemented based on PaddlePaddle

**PPDiffusers**是一款支持多种模态（如文本图像跨模态、图像、语音）扩散模型（Diffusion Model）训练和推理的国产化工具箱，依托于[**PaddlePaddle**](https://www.paddlepaddle.org.cn/)框架和[**PaddleNLP**](https://github.com/PaddlePaddle/PaddleNLP)自然语言处理开发库。

## News 📢
* 🔥 **2025.07.14 发布 0.30.0 版本，新增文生图算法FLUX，文生视频算法[HunyuanVideo](examples/HunyuanVideo/)和[Wan2.1](examples/Wan2.1/)。发布推理加速工具包[Fast-Diffusers](examples/Fast-Diffusers)，包含SOTA Training-Free算法：[T-gate](examples/Fast-Diffusers/Training-Free/tgate), [PAB](examples/Fast-Diffusers/Training-Free/pab), [TeaCache](examples/Fast-Diffusers/Training-Free/teacache), [TaylorSeer](examples/Fast-Diffusers/Training-Free/taylorseer), [BlockDance](examples/Fast-Diffusers/Training-Free/blockdance)等和自研算法：[SortBlock](examples/Fast-Diffusers/Training-Free/sortblock)，[TeaBlockCache](examples/Fast-Diffusers/Training-Free/teablockcache), [CG-Taylor](ppdiffusers/examples/Fast-Diffusers/Training-Free/CG-Taylor/)和[FirstBlockTaylor](examples/Fast-Diffusers/Training-Free/firstblock_taylorseer)算法，在保证生成图像质量的同时，实现2倍以上的端到端推理加速效果。新增[PCM](examples/Fast-Diffusers/diffusion-distill/phased_consistency_distillation),[DMD2](examples/Fast-Diffusers/diffusion-distill//dmd2)等扩散模型蒸馏算法，并提供了多种蒸馏loss供开发者灵活搭配。同时基于上述蒸馏算法，发布了基于FLUX-dev的4步蒸馏模型，配合飞桨深度学习编译器，推理时延降低至1.66秒。**

* 🔥 **2025.03.11 发布 0.29.1 版本，发布自研可控视频模型 [PP-VCtrl](https://github.com/westfish/PaddleMIX/tree/develop/ppdiffusers/examples/ppvctrl)，支持在多种控制条件下的视频生成，广泛适用于人物动画、场景转换和视频编辑场景；集成高质量文生视频模型 [CogVideoX](https://github.com/westfish/PaddleMIX/tree/develop/ppdiffusers/examples/cogvideo) 和 HunyuanVideo，进一步提高了文本到视频生成的效果；新增对 SD3 ControlNet 和 SD3.5 的支持，提升了Stable Diffusion 3的兼容性，进一步强化图像生成的精细化控制能力。**

* 🔥 **2024.10.18 发布 0.29.0 版本，新增图像生成模型[Stable Diffusion 3 (SD3)](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/examples/text_to_image/README_sd3.md)，支持DreamBooth训练及高性能推理；SD3、SDXL适配昇腾910B，提供国产计算芯片上的训推能力；DIT支持[高性能推理](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/examples/class_conditional_image_generation/DiT/README.md#23-paddle-inference-%E9%AB%98%E6%80%A7%E8%83%BD%E6%8E%A8%E7%90%86)；支持PaddleNLP 3.0 beta版本。**

* 🔥 **2024.07.15 发布 0.24.1 版本，新增[Open-Sora](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/Open-Sora)，支持模型训练和推理；全面支持Paddle 3.0。**

<details>
<summary>点击展开更多历史版本更新</summary>

* 🔥 **2024.04.17 发布 0.24.0 版本，支持[Sora相关技术](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/sora)，支持[DiT](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/class_conditional_image_generation/DiT)、[SiT](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/class_conditional_image_generation/DiT#exploring-flow-and-diffusion-based-generative-models-with-scalable-interpolant-transformers-sit)、[UViT](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/text_to_image_mscoco_uvit)训练推理，新增[NaViT](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/navit)、[MAGVIT-v2](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/video_tokenizer/magvit2)模型；
视频生成能力全面升级；
新增视频生成模型[SVD](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/stable_video_diffusion)，支持模型微调和推理；
新增姿态可控视频生成模型[AnimateAnyone](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/AnimateAnyone)、即插即用视频生成模型[AnimateDiff](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/examples/inference/text_to_video_generation_animediff.py)、GIF视频生成模型[Hotshot-XL](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/community/Hotshot-XL)；
新增高速推理文图生成模型[LCM](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/consistency_distillation)，支持SD/SDXL训练和推理；
[模型推理部署](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/deploy)全面升级；新增peft，accelerate后端；
权重加载/保存全面升级，支持分布式、模型切片、safetensors等场景，相关能力已集成DiT、 [IP-Adapter](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/ip_adapter)、[PhotoMaker](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/PhotoMaker)、[InstantID](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/InstantID)等。**
* 🔥 **2023.12.12 发布 0.19.4 版本，修复已知的部分 BUG，修复 0D Tensor 的 Warning，新增 SDXL 的 FastdeployPipeline。**
* 🔥 **2023.09.27 发布 0.19.3 版本，新增[SDXL](#文本图像多模)，支持Text2Image、Img2Img、Inpainting、InstructPix2Pix等任务，支持DreamBooth Lora训练；
新增[UniDiffuser](#文本图像多模)，通过统一的多模态扩散过程支持文生图、图生文等任务；
新增文本条件视频生成模型[LVDM](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/text_to_video_lvdm)，支持训练与推理；
新增文图生成模型[Kandinsky 2.2](#文本图像多模)，[Consistency models](#文本图像多模)；
Stable Diffusion支持[BF16 O2训练](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/stable_diffusion)，效果对齐FP32；
[LoRA加载升级](#加载HF-LoRA权重)，支持加载SDXL的LoRA权重；
[Controlnet](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/ppdiffusers/pipelines/controlnet)升级，支持ControlNetImg2Img、ControlNetInpaint、StableDiffusionXLControlNet等。**
</details>


## 特性
#### 📦 SOTA扩散模型Pipelines集合
我们提供**SOTA（State-of-the-Art）** 的扩散模型Pipelines集合。
目前**PPDiffusers**已经集成了**100+Pipelines**，支持文图生成（Text-to-Image Generation）、文本引导的图像编辑（Text-Guided Image Inpainting）、文本引导的图像变换（Image-to-Image Text-Guided Generation）、文本条件的视频生成（Text-to-Video Generation）、超分（Super Superresolution）、文本条件的音频生成（Text-to-Audio Generation）在内的**10余项**任务，覆盖**文本、图像、视频、音频**等多种模态。
如果想要了解当前支持的所有**Pipelines**以及对应的来源信息，可以阅读[🔥 PPDiffusers Pipelines](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/ppdiffusers/pipelines/README.md)文档。


#### 🔊 提供丰富的Noise Scheduler
我们提供了丰富的**噪声调度器（Noise Scheduler）**，可以对**速度**与**质量**进行权衡，用户可在推理时根据需求快速切换使用。
当前**PPDiffusers**已经集成了**14+Scheduler**，不仅支持 [DDPM](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/ppdiffusers/schedulers/scheduling_ddpm.py)、[DDIM](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/ppdiffusers/schedulers/scheduling_ddim.py) 和 [PNDM](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/ppdiffusers/schedulers/scheduling_pndm.py)，还支持最新的 [🔥 DPMSolver](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/ppdiffusers/schedulers/scheduling_dpmsolver_multistep.py)！

#### 🎛️ 提供多种扩散模型组件
我们提供了**多种扩散模型**组件，如[UNet1DModel](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/ppdiffusers/models/unet_1d.py)、[UNet2DModel](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/ppdiffusers/models/unet_2d.py)、[UNet2DConditionModel](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/ppdiffusers/models/unet_2d_condition.py)、[UNet3DConditionModel](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/ppdiffusers/models/unet_3d_condition.py)、[VQModel](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/ppdiffusers/models/vae.py)、[AutoencoderKL](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/ppdiffusers/models/vae.py)等。


#### 📖 提供丰富的训练和推理教程
我们提供了丰富的训练教程，不仅支持扩散模型的二次开发微调，如基于[Textual Inversion](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/textual_inversion)和[DreamBooth](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/dreambooth)使用3-5张图定制化训练生成图像的风格或物体，还支持[🔥 Latent Diffusion Model](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/text_to_image_laion400m)、[🔥 ControlNet](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/controlnet)、[🔥 T2I-Adapter](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/t2i-adapter)  等扩散模型的训练！
此外，我们还提供了丰富的[🔥 Pipelines推理样例](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/examples/inference)。

#### 🚀 支持FastDeploy高性能部署
我们提供基于[FastDeploy](https://github.com/PaddlePaddle/FastDeploy)的[🔥 高性能Stable Diffusion Pipeline](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/ppdiffusers/pipelines/stable_diffusion/pipeline_fastdeploy_stable_diffusion.py)，更多有关FastDeploy进行多推理引擎后端高性能部署的信息请参考[🔥 高性能FastDeploy推理教程](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers/deploy)。

## 安装

### 环境依赖
```
pip install -r requirements.txt
```
关于PaddlePaddle安装的详细教程请查看[Installation](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)。

### pip安装

```shell
pip install --upgrade ppdiffusers
```

### 手动安装
```shell
git clone https://github.com/PaddlePaddle/PaddleMIX
cd PaddleMIX/ppdiffusers
python setup.py install
```
### 设置代理
```shell
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_ENDPOINT=https://hf-mirror.com
```

## 快速开始
我们将以扩散模型的典型代表**Stable Diffusion**为例，带你快速了解PPDiffusers。

**Stable Diffusion**基于**潜在扩散模型（Latent Diffusion Models）**，专门用于**文图生成（Text-to-Image Generation）任务**。该模型是由来自 [CompVis](https://github.com/CompVis), [Stability AI](https://stability.ai/), [LAION](https://laion.ai/)以及[RunwayML](https://runwayml.com/)的工程师共同开发完成，目前发布了v1和v2两个版本。v1版本采用了LAION-5B数据集子集（分辨率为 512x512）进行训练，并具有以下架构设置：自动编码器下采样因子为8，UNet大小为860M，文本编码器为CLIP ViT-L/14。v2版本相较于v1版本在生成图像的质量和分辨率等进行了改善。

### Stable Diffusion重点模型权重

<details><summary>&emsp; Stable Diffusion 模型支持的权重（英文） </summary>

**我们只需要将下面的"xxxx"，替换成所需的权重名，即可快速使用！**
```python
from ppdiffusers import *

pipe_text2img = StableDiffusionPipeline.from_pretrained("xxxx")
pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained("xxxx")
pipe_inpaint_legacy = StableDiffusionInpaintPipelineLegacy.from_pretrained("xxxx")
pipe_mega = StableDiffusionMegaPipeline.from_pretrained("xxxx")

# pipe_mega.text2img() 等于 pipe_text2img()
# pipe_mega.img2img() 等于 pipe_img2img()
# pipe_mega.inpaint_legacy() 等于 pipe_inpaint_legacy()
```

| PPDiffusers支持的模型名称                     | 支持加载的Pipeline                                    | 备注 | huggingface.co地址 |
| :-------------------------------------------: | :--------------------------------------------------------------------: | --- | :-----------------------------------------: |
| CompVis/stable-diffusion-v1-4           | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | Stable-Diffusion-v1-4 使用 Stable-Diffusion-v1-2 的权重进行初始化。随后在"laion-aesthetics v2 5+"数据集上以 **512x512** 分辨率微调了 **225k** 步数，对文本使用了 **10%** 的dropout（即：训练过程中文图对中的文本有 10% 的概率会变成空文本）。模型使用了[CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14)作为文本编码器。| [地址](https://huggingface.co/CompVis/stable-diffusion-v1-4) |
| CompVis/ldm-text2im-large-256               | LDMTextToImagePipeline | [LDM论文](https://arxiv.org/pdf/2112.10752.pdf) LDM-KL-8-G* 权重。| [地址](https://huggingface.co/CompVis/ldm-text2im-large-256) |
| CompVis/ldm-super-resolution-4x-openimages  | LDMSuperResolutionPipeline | [LDM论文](https://arxiv.org/pdf/2112.10752.pdf) LDM-VQ-4 权重，[原始权重链接](https://ommer-lab.com/files/latent-diffusion/sr_bsr.zip)。| [地址](https://huggingface.co/CompVis/ldm-super-resolution-4x-openimages) |
| runwayml/stable-diffusion-v1-5              | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | Stable-Diffusion-v1-5 使用 Stable-Diffusion-v1-2 的权重进行初始化。随后在"laion-aesthetics v2 5+"数据集上以 **512x512** 分辨率微调了 **595k** 步数，对文本使用了 **10%** 的dropout（即：训练过程中文图对中的文本有 10% 的概率会变成空文本）。模型同样也使用了[CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14)作为文本编码器。| [地址](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
| runwayml/stable-diffusion-inpainting        | StableDiffusionInpaintPipeline | Stable-Diffusion-Inpainting 使用 Stable-Diffusion-v1-2 的权重进行初始化。首先进行了 **595k** 步的常规训练（实际也就是 Stable-Diffusion-v1-5 的权重），然后进行了 **440k** 步的 inpainting 修复训练。对于 inpainting 修复训练，给 UNet 额外增加了 **5** 输入通道（其中 **4** 个用于被 Mask 遮盖住的图片，**1** 个用于 Mask 本身）。在训练期间，会随机生成 Mask，并有 **25%** 概率会将原始图片全部 Mask 掉。| [地址](https://huggingface.co/runwayml/stable-diffusion-inpainting) |
| stabilityai/stable-diffusion-2-base         | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | 该模型首先在 [LAION-5B 256x256 子集上](https://laion.ai/blog/laion-5b/) （过滤条件：[punsafe = 0.1 的 LAION-NSFW 分类器](https://github.com/LAION-AI/CLIP-based-NSFW-Detector) 和 审美分数大于等于 4.5 ）从头开始训练 **550k** 步，然后又在分辨率 **>= 512x512** 的同一数据集上进一步训练 **850k** 步。| [地址](https://huggingface.co/stabilityai/stable-diffusion-2-base) |
| stabilityai/stable-diffusion-2              | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | stable-diffusion-2 使用 stable-diffusion-2-base 权重进行初始化，首先在同一数据集上（**512x512** 分辨率）使用 [v-objective](https://arxiv.org/abs/2202.00512) 训练了 **150k** 步。然后又在 **768x768** 分辨率上使用 [v-objective](https://arxiv.org/abs/2202.00512) 继续训练了 **140k** 步。| [地址](https://huggingface.co/stabilityai/stable-diffusion-2) |
| stabilityai/stable-diffusion-2-inpainting   | StableDiffusionInpaintPipeline |stable-diffusion-2-inpainting 使用 stable-diffusion-2-base 权重初始化，并且额外训练了 **200k** 步。训练过程使用了 [LAMA](https://github.com/saic-mdal/lama) 中提出的 Mask 生成策略，并且使用 Mask 图片的 Latent 表示（经过 VAE 编码）作为附加条件。| [地址](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) |
| stabilityai/stable-diffusion-x4-upscaler    | StableDiffusionUpscalePipeline | 该模型在**LAION 10M** 子集上（>2048x2048）训练了 1.25M 步。该模型还在分辨率为 **512x512** 的图像上使用 [Text-guided Latent Upscaling Diffusion Model](https://arxiv.org/abs/2112.10752) 进行了训练。除了**文本输入**之外，它还接收 **noise_level** 作为输入参数，因此我们可以使用 [预定义的 Scheduler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/blob/main/low_res_scheduler/scheduler_config.json) 向低分辨率的输入图片添加噪声。| [地址](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) |
| hakurei/waifu-diffusion    | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | waifu-diffusion-v1-2 使用 stable-diffusion-v1-4 权重初始化，并且在**高质量动漫**图像数据集上进行微调后得到的模型。用于微调的数据是 **680k** 文本图像样本，这些样本是通过 **booru 网站** 下载的。| [地址](https://huggingface.co/hakurei/waifu-diffusion) |
| hakurei/waifu-diffusion-v1-3    | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | waifu-diffusion-v1-3 是 waifu-diffusion-v1-2 基础上进一步训练得到的。他们对数据集进行了额外操作：（1）删除下划线；（2）删除括号；（3）用逗号分隔每个booru 标签；（4）随机化标签顺序。| [地址](https://huggingface.co/hakurei/waifu-diffusion) |
| naclbit/trinart_stable_diffusion_v2_60k    | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | trinart_stable_diffusion 使用 stable-diffusion-v1-4 权重初始化，在 40k **高分辨率漫画/动漫风格**的图片数据集上微调了 8 个 epoch。V2 版模型使用 **dropouts**、**10k+ 图像**和**新的标记策略**训练了**更长时间**。| [地址](https://huggingface.co/naclbit/trinart_stable_diffusion_v2) |
| naclbit/trinart_stable_diffusion_v2_95k    | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | **95k** 步数的结果，其他同上。| [地址](https://huggingface.co/naclbit/trinart_stable_diffusion_v2) |
| naclbit/trinart_stable_diffusion_v2_115k    | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | **115k** 步数的结果，其他同上。| [地址](https://huggingface.co/naclbit/trinart_stable_diffusion_v2) |
| Deltaadams/Hentai-Diffusion    | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | None| [地址](https://huggingface.co/Deltaadams/Hentai-Diffusion) |
| ringhyacinth/nail-set-diffuser    | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | 美甲领域的扩散模型，训练数据使用了 [Weekend](https://weibo.com/u/5982308498)| [地址](https://huggingface.co/ringhyacinth/nail-set-diffuser) |
| Linaqruf/anything-v3.0    | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | 该模型可通过输入几个文本提示词就能生成**高质量、高度详细的动漫风格图片**，该模型支持使用 **danbooru 标签文本** 生成图像。| [地址](https://huggingface.co/Linaqruf/anything-v3.0) |

</details>
<details><summary>&emsp; Stable Diffusion 模型支持的权重（中文和多语言） </summary>


| PPDiffusers支持的模型名称                     | 支持加载的Pipeline                                    | 备注 | huggingface.co地址 |
| :-------------------------------------------: | :--------------------------------------------------------------------: | --- | :-----------------------------------------: |
| BAAI/AltDiffusion                           | AltDiffusionPipeline、AltDiffusionImg2ImgPipeline | 该模型使用 [AltCLIP](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP/README.md) 作为文本编码器，在 Stable Diffusion 基础上训练了**双语Diffusion模型**，其中训练数据来自 [WuDao数据集](https://data.baai.ac.cn/details/WuDaoCorporaText) 和 [LAION](https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6plus) 。| [地址](https://huggingface.co/BAAI/AltDiffusion) |
| BAAI/AltDiffusion-m9                        | AltDiffusionPipeline、AltDiffusionImg2ImgPipeline |该模型使用9种语言的 [AltCLIP-m9](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/AltCLIP/README.md) 作为文本编码器，其他同上。| [地址](https://huggingface.co/BAAI/AltDiffusion-m9) |
| IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1 | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | 他们将 [Noah-Wukong](https://wukong-dataset.github.io/wukong-dataset/) 数据集 (100M) 和 [Zero](https://zero.so.com/) 数据集 (23M) 用作预训练的数据集，先用 [IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese) 对这两个数据集的图文对相似性进行打分，取 CLIP Score 大于 0.2 的图文对作为训练集。 他们使用 [IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese) 作为初始化的text encoder，冻住 [stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) ([论文](https://arxiv.org/abs/2112.10752)) 模型的其他部分，只训练 text encoder，以便保留原始模型的生成能力且实现中文概念的对齐。该模型目前在0.2亿图文对上训练了一个 epoch。 在 32 x A100 上训练了大约100小时，该版本只是一个初步的版本。| [地址](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1) |
| IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1 | StableDiffusionPipeline、StableDiffusionImg2ImgPipeline、StableDiffusionInpaintPipelineLegacy、StableDiffusionMegaPipeline、StableDiffusionPipelineAllinOne | 他们将 [Noah-Wukong](https://wukong-dataset.github.io/wukong-dataset/) 数据集 (100M) 和 [Zero](https://zero.so.com/) 数据集 (23M) 用作预训练的数据集，先用 [IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese) 对这两个数据集的图文对相似性进行打分，取 CLIP Score 大于 0.2 的图文对作为训练集。 他们使用 [stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) ([论文](https://arxiv.org/abs/2112.10752)) 模型进行继续训练，其中训练分为**两个stage**。**第一个stage** 中冻住模型的其他部分，只训练 text encoder ，以便保留原始模型的生成能力且实现中文概念的对齐。**第二个stage** 中将全部模型解冻，一起训练 text encoder 和 diffusion model ，以便 diffusion model 更好的适配中文引导。第一个 stage 他们训练了 80 小时，第二个 stage 训练了 100 小时，两个stage都是用了8 x A100，该版本是一个初步的版本。| [地址](https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1) |
</details>


### 加载HF Diffusers权重
```python
from ppdiffusers import StableDiffusionPipeline
# 设置from_hf_hub为True，表示从huggingface hub下载，from_diffusers为True表示加载的是diffusers版Pytorch权重
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", from_hf_hub=True, from_diffusers=True)
```

### 加载原库的Lightning权重
```python
from ppdiffusers import StableDiffusionPipeline
# 可输入网址 或 本地ckpt、safetensors文件
pipe = StableDiffusionPipeline.from_single_file("https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/ppdiffusers/chilloutmix_NiPrunedFp32Fix.safetensors")
```

### 加载HF LoRA权重
```python
from ppdiffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", paddle_dtype=paddle.float16)

pipe.load_lora_weights("stabilityai/stable-diffusion-xl-base-1.0",
    weight_name="sd_xl_offset_example-lora_1.0.safetensors",
    from_diffusers=True)
```

### 加载Civitai社区的LoRA权重
```python
from ppdiffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("TASUKU2023/Chilloutmix")
# 加载lora权重
pipe.load_lora_weights("./",
    weight_name="Moxin_10.safetensors",
    from_diffusers=True)
pipe.fuse_lora()
```

### XFormers加速
为了使用**XFormers加速**，我们需要安装`develop`版本的`paddle`，Linux系统的安装命令如下：
```sh
python -m pip install paddlepaddle-gpu==0.0.0.post117 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

```python
import paddle
from ppdiffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("TASUKU2023/Chilloutmix", paddle_dtype=paddle.float16)
# 开启xformers加速 默认选择"cutlass"加速
pipe.enable_xformers_memory_efficient_attention()
# flash 需要使用 A100、A10、3060、3070、3080、3090 等以上显卡。
# pipe.enable_xformers_memory_efficient_attention("flash")
```

### ToME + ControlNet
```python
# 安装develop的ppdiffusers
# pip install "ppdiffusers>=0.24.0"
import paddle
from ppdiffusers import ControlNetModel, StableDiffusionControlNetPipeline
from ppdiffusers.utils import load_image

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet, paddle_dtype=paddle.float16
)

# Apply ToMe with a 50% merging ratio
pipe.apply_tome(ratio=0.5) # Can also use pipe.unet in place of pipe here

# 我们可以开启 xformers
# pipe.enable_xformers_memory_efficient_attention()
generator = paddle.Generator().manual_seed(0)
prompt = "bird"
image = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
)

image = pipe(prompt, image, generator=generator).images[0]

image.save("bird.png")
```

### 文图生成 （Text-to-Image Generation）

```python
import paddle
from ppdiffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")

# 设置随机种子，我们可以复现下面的结果！
paddle.seed(5232132133)
prompt = "a portrait of shiba inu with a red cap growing on its head. intricate. lifelike. soft light. sony a 7 r iv 5 5 mm. cinematic post - processing "
image = pipe(prompt, guidance_scale=7.5, height=768, width=768).images[0]

image.save("shiba_dog_with_a_red_cap.png")
```
<div align="center">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/50394665/204796701-d7911f76-8670-47d5-8d1b-8368b046c5e4.png">
</div>

### 文本引导的图像变换（Image-to-Image Text-Guided Generation）

<details><summary>&emsp;Image-to-Image Text-Guided Generation Demo </summary>

```python
import paddle
from ppdiffusers import StableDiffusionImg2ImgPipeline
from ppdiffusers.utils import load_image

pipe = StableDiffusionImg2ImgPipeline.from_pretrained("Linaqruf/anything-v3.0", safety_checker=None)

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/image_Kurisu.png"
image = load_image(url).resize((512, 768))

# 设置随机种子，我们可以复现下面的结果！
paddle.seed(42)
prompt = "Kurisu Makise, looking at viewer, long hair, standing, 1girl, hair ornament, hair flower, cute, jacket, white flower, white dress"
negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=image, strength=0.75, guidance_scale=7.5).images[0]
image.save("image_Kurisu_img2img.png")
```
<div align="center">
<img width="500" alt="image" src="https://user-images.githubusercontent.com/50394665/204799529-cd89dcdb-eb1d-4247-91ac-b0f7bad777f8.png">
</div>
</details>

### 文本引导的图像编辑（Text-Guided Image Inpainting）

注意！当前有两种版本的图像编辑代码，一个是Legacy版本，一个是正式版本，下面将分别介绍两种代码如何使用！

<details><summary>&emsp;Legacy版本代码</summary>

```python
import paddle
from ppdiffusers import StableDiffusionInpaintPipelineLegacy
from ppdiffusers.utils import load_image

# 可选模型权重
# CompVis/stable-diffusion-v1-4
# runwayml/stable-diffusion-v1-5
# stabilityai/stable-diffusion-2-base （原始策略 512x512）
# stabilityai/stable-diffusion-2 （v-objective 768x768）
# Linaqruf/anything-v3.0
# ......
img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"

image = load_image(img_url).resize((512, 512))
mask_image = load_image(mask_url).resize((512, 512))

pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained("stabilityai/stable-diffusion-2-base", safety_checker=None)

# 设置随机种子，我们可以复现下面的结果！
paddle.seed(10245)
prompt = "a red cat sitting on a bench"
image = pipe(prompt=prompt, image=image, mask_image=mask_image, strength=0.75).images[0]

image.save("a_red_cat_legacy.png")
```
<div align="center">
<img width="900" alt="image" src="https://user-images.githubusercontent.com/50394665/204802186-5a6d302b-83aa-4247-a5bb-ebabfcc3abc4.png">
</div>

</details>

<details><summary>&emsp;正式版本代码</summary>

Tips: 下面的使用方法是新版本的代码，也是官方推荐的代码，注意必须配合 **runwayml/stable-diffusion-inpainting** 和 **stabilityai/stable-diffusion-2-inpainting** 才可正常使用。
```python
import paddle
from ppdiffusers import StableDiffusionInpaintPipeline
from ppdiffusers.utils import load_image

# 可选模型权重
# runwayml/stable-diffusion-inpainting
# stabilityai/stable-diffusion-2-inpainting
img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"

image = load_image(img_url).resize((512, 512))
mask_image = load_image(mask_url).resize((512, 512))

pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting")

# 设置随机种子，我们可以复现下面的结果！
paddle.seed(1024)
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]

image.save("a_yellow_cat.png")
```
<div align="center">
<img width="900" alt="image" src="https://user-images.githubusercontent.com/50394665/204801946-6cd043bc-f3db-42cf-82cd-6a6171484523.png">
</div>
</details>

### 文本引导的图像放大 & 超分（Text-Guided Image Upscaling & Super-Resolution）

<details><summary>&emsp;Text-Guided Image Upscaling Demo</summary>

```python
import paddle
from ppdiffusers import StableDiffusionUpscalePipeline
from ppdiffusers.utils import load_image

pipe = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler")

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/low_res_cat.png"
# 我们人工将原始图片缩小成 128x128 分辨率，最终保存的图片会放大4倍！
low_res_img = load_image(url).resize((128, 128))

prompt = "a white cat"
image = pipe(prompt=prompt, image=low_res_img).images[0]

image.save("upscaled_white_cat.png")
```
<div align="center">
<img width="200" alt="image" src="https://user-images.githubusercontent.com/50394665/204806180-b7f1b9cf-8a62-4577-b5c4-91adda08a13b.png">
<img width="400" alt="image" src="https://user-images.githubusercontent.com/50394665/204806202-8c110be3-5f48-4946-95ea-21ad5a9a2340.png">
</div>
</details>

<details><summary>&emsp;Super-Resolution Demo</summary>

```python
import paddle
from ppdiffusers import LDMSuperResolutionPipeline
from ppdiffusers.utils import load_image

pipe = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages")

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"

# 我们人工将原始图片缩小成 128x128 分辨率，最终保存的图片会放大4倍！
low_res_img = load_image(url).resize((128, 128))

image = pipe(image=low_res_img, num_inference_steps=100).images[0]

image.save("ldm-super-resolution-image.png")
```
<div align="center">
<img width="200" alt="image" src="https://user-images.githubusercontent.com/50394665/204804426-5e28b571-aa41-4f56-ba26-68cca75fdaae.png">
<img width="400" alt="image" src="https://user-images.githubusercontent.com/50394665/204804148-fe7c293b-6cd7-4942-ae9c-446369fe8410.png">
</div>

</details>

## 模型推理部署
除了**Paddle动态图**运行之外，很多模型还支持将模型导出并使用推理引擎运行。我们提供基于[FastDeploy](https://github.com/PaddlePaddle/FastDeploy)上的**StableDiffusion**模型部署示例，涵盖文生图、图生图、图像编辑等任务，用户可以按照我们提供[StableDiffusion模型导出教程](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/deploy/export.md)将模型导出，然后使用`FastDeployStableDiffusionMegaPipeline`进行高性能推理部署！

<details><summary>&emsp; 已预先导出的FastDeploy版Stable Diffusion权重 </summary>

**注意：当前导出的vae encoder带有随机因素！**

- CompVis/stable-diffusion-v1-4@fastdeploy
- runwayml/stable-diffusion-v1-5@fastdeploy
- runwayml/stable-diffusion-inpainting@fastdeploy
- stabilityai/stable-diffusion-2-base@fastdeploy
- stabilityai/stable-diffusion-2@fastdeploy
- stabilityai/stable-diffusion-2-inpainting@fastdeploy
- Linaqruf/anything-v3.0@fastdeploy
- hakurei/waifu-diffusion-v1-3@fastdeploy

</details>

<details><summary>&emsp; FastDeploy Demo </summary>

```python
import paddle
import fastdeploy as fd
from ppdiffusers import FastDeployStableDiffusionMegaPipeline
from ppdiffusers.utils import load_image

def create_runtime_option(device_id=0, backend="paddle", use_cuda_stream=True):
    option = fd.RuntimeOption()
    if backend == "paddle":
        option.use_paddle_backend()
    else:
        option.use_ort_backend()
    if device_id == -1:
        option.use_cpu()
    else:
        option.use_gpu(device_id)
        if use_cuda_stream:
            paddle_stream = paddle.device.cuda.current_stream(device_id).cuda_stream
            option.set_external_raw_stream(paddle_stream)
    return option

runtime_options = {
    "text_encoder": create_runtime_option(0, "paddle"),  # use gpu:0
    "vae_encoder": create_runtime_option(0, "paddle"),  # use gpu:0
    "vae_decoder": create_runtime_option(0, "paddle"),  # use gpu:0
    "unet": create_runtime_option(0, "paddle"),  # use gpu:0
}

fd_pipe = FastDeployStableDiffusionMegaPipeline.from_pretrained(
    "Linaqruf/anything-v3.0@fastdeploy", runtime_options=runtime_options
)

# text2img
prompt = "a portrait of shiba inu with a red cap growing on its head. intricate. lifelike. soft light. sony a 7 r iv 5 5 mm. cinematic post - processing "
image_text2img = fd_pipe.text2img(prompt=prompt, num_inference_steps=50).images[0]
image_text2img.save("image_text2img.png")

# img2img
url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/image_Kurisu.png"
image = load_image(url).resize((512, 512))
prompt = "Kurisu Makise, looking at viewer, long hair, standing, 1girl, hair ornament, hair flower, cute, jacket, white flower, white dress"
negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

image_img2img = fd_pipe.img2img(
    prompt=prompt, negative_prompt=negative_prompt, image=image, strength=0.75, guidance_scale=7.5
).images[0]
image_img2img.save("image_img2img.png")

# inpaint_legacy
img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"
image = load_image(img_url).resize((512, 512))
mask_image = load_image(mask_url).resize((512, 512))
prompt = "a red cat sitting on a bench"

image_inpaint_legacy = fd_pipe.inpaint_legacy(
    prompt=prompt, image=image, mask_image=mask_image, strength=0.75, num_inference_steps=50
).images[0]
image_inpaint_legacy.save("image_inpaint_legacy.png")
```
</details>
<div align="center">
<img width="900" alt="image" src="https://user-images.githubusercontent.com/50394665/205297240-46b80992-34af-40cd-91a6-ae76589d0e21.png">
</div>


## 更多任务分类展示
### 文本图像多模

<details open>
<summary>&emsp;文图生成（Text-to-Image Generation）</summary>

#### text_to_image_generation-stable_diffusion

```python
from ppdiffusers import StableDiffusionPipeline

# 加载模型和scheduler
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# 执行pipeline进行推理
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

# 保存图片
image.save("astronaut_rides_horse_sd.png")
```
<div align="center">
<img width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/209322401-6ecfeaaa-6878-4302-b592-07a31de4e590.png">
</div>

#### text_to_image_generation-stable_diffusion_xl

```python
import paddle
from ppdiffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
     "stabilityai/stable-diffusion-xl-base-1.0",
     paddle_dtype=paddle.float16,
     variant="fp16"
)
prompt = "a photo of an astronaut riding a horse on mars"
generator = paddle.Generator().manual_seed(42)
image = pipe(prompt=prompt, generator=generator, num_inference_steps=50).images[0]
image.save('sdxl_text2image.png')
```
<div align="center">
<img width="300" alt="image" src="https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/d72729f9-8685-48f9-a238-e4ddf6d264f3">
</div>

#### text_to_image_generation-sdxl_base_with_refiner

```python
from ppdiffusers import DiffusionPipeline
import paddle

# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    paddle_dtype=paddle.float16,
)
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    paddle_dtype=paddle.float16,
    variant="fp16",
)

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

prompt = "A majestic lion jumping from a big stone at night"
prompt = "a photo of an astronaut riding a horse on mars"
generator = paddle.Generator().manual_seed(42)

# run both experts
image = base(
    prompt=prompt,
    output_type="latent",
    generator=generator,
).images

image = refiner(
    prompt=prompt,
    image=image,
    generator=generator,
).images[0]
image.save('text_to_image_generation-sdxl-base-with-refiner-result.png')
```
<div align="center">
<img width="300" alt="image" src="https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/8ef36826-ed94-4856-a356-af1677f60d1b">
</div>

#### text_to_image_generation-kandinsky2_2
```python
from ppdiffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline

pipe_prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior")
prompt = "red cat, 4k photo"
out = pipe_prior(prompt)
image_emb = out.image_embeds
zero_image_emb = out.negative_image_embeds
pipe = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder")
image = pipe(
    image_embeds=image_emb,
    negative_image_embeds=zero_image_emb,
    height=768,
    width=768,
    num_inference_steps=50,
).images
image[0].save("text_to_image_generation-kandinsky2_2-result-cat.png")
```
<div align="center">
<img width="300" alt="image" src="https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/188f76dd-4bd7-4a33-8f30-b893c7a9e249">
</div>

#### text_to_image_generation-unidiffuser
```python
import paddle
from paddlenlp.trainer import set_seed

from ppdiffusers import UniDiffuserPipeline

model_id_or_path = "thu-ml/unidiffuser-v1"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, paddle_dtype=paddle.float16)
set_seed(42)

# Text variation can be performed with a text-to-image generation followed by a image-to-text generation:
# 1. Text-to-image generation
prompt = "an elephant under the sea"
sample = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=8.0)
t2i_image = sample.images[0]
t2i_image.save("t2i_image.png")
````
<div align="center">
<img width="300" alt="image" src="https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/a6eb11d2-ad27-4263-8cb4-b0d8dd42b36c">
</div>

#### text_to_image_generation-deepfloyd_if

```python
import paddle

from ppdiffusers import DiffusionPipeline, IFPipeline, IFSuperResolutionPipeline
from ppdiffusers.utils import pd_to_pil

# Stage 1: generate images
pipe = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", paddle_dtype=paddle.float16)
pipe.enable_xformers_memory_efficient_attention()
prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'
prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)
image = pipe(
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    output_type="pd",
).images

# save intermediate image
pil_image = pd_to_pil(image)
pil_image[0].save("text_to_image_generation-deepfloyd_if-result-if_stage_I.png")
# save gpu memory
pipe.to(paddle_device="cpu")

# Stage 2: super resolution stage1
super_res_1_pipe = IFSuperResolutionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", paddle_dtype=paddle.float16
)
super_res_1_pipe.enable_xformers_memory_efficient_attention()

image = super_res_1_pipe(
    image=image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    output_type="pd",
).images
# save intermediate image
pil_image = pd_to_pil(image)
pil_image[0].save("text_to_image_generation-deepfloyd_if-result-if_stage_II.png")
# save gpu memory
super_res_1_pipe.to(paddle_device="cpu")
```
<div align="center">
<img alt="image" src="https://user-images.githubusercontent.com/20476674/246785766-700dfad9-159d-4bfb-bfc7-c18df938a052.png">
</div>
<div align="center">
<center>if_stage_I</center>
</div>
<div align="center">
<img alt="image" src="https://user-images.githubusercontent.com/20476674/246785773-3359ca5f-dadf-4cc8-b318-ff1f9d4a2d35.png">
</div>
<div align="center">
<center>if_stage_II</center>
<!-- <img alt="image" src="https://user-images.githubusercontent.com/20476674/246785774-8870829a-354b-4a87-9d67-93af315f51e6.png">
<center>if_stage_III</center> -->
</div>
</details>


<details><summary>&emsp;文本引导的图像放大（Text-Guided Image Upscaling）</summary>

#### text_guided_image_upscaling-stable_diffusion_2

```python
from ppdiffusers import StableDiffusionUpscalePipeline
from ppdiffusers.utils import load_image

pipe = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler")

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/low_res_cat.png"
low_res_img = load_image(url).resize((128, 128))

prompt = "a white cat"
upscaled_image = pipe(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save("upsampled_cat_sd2.png")
```
<div align="center">
<img alt="image" src="https://user-images.githubusercontent.com/20476674/209324085-0d058b70-89b0-43c2-affe-534eedf116cf.png">
<center>原图像</center>
<img alt="image" src="https://user-images.githubusercontent.com/20476674/209323862-ce2d8658-a52b-4f35-90cb-aa7d310022e7.png">
<center>生成图像</center>
</div>
</details>

<details><summary>&emsp;文本引导的图像编辑（Text-Guided Image Inpainting）</summary>

#### text_guided_image_inpainting-stable_diffusion_2

```python
import paddle

from ppdiffusers import PaintByExamplePipeline
from ppdiffusers.utils import load_image

img_url = "https://paddlenlp.bj.bcebos.com/models/community/Fantasy-Studio/data/image_example_1.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/Fantasy-Studio/data/mask_example_1.png"
example_url = "https://paddlenlp.bj.bcebos.com/models/community/Fantasy-Studio/data/reference_example_1.jpeg"

init_image = load_image(img_url).resize((512, 512))
mask_image = load_image(mask_url).resize((512, 512))
example_image = load_image(example_url).resize((512, 512))

pipe = PaintByExamplePipeline.from_pretrained("Fantasy-Studio/Paint-by-Example")

# 使用fp16加快生成速度
with paddle.amp.auto_cast(True):
    image = pipe(image=init_image, mask_image=mask_image, example_image=example_image).images[0]
image.save("image_guided_image_inpainting-paint_by_example-result.png")
```
<div align="center">
<img alt="image" src="https://user-images.githubusercontent.com/20476674/247118364-5d91f433-f9ac-4514-b5f0-cb4599905847.png" width=300>
<center>原图像</center>
<div align="center">
<img alt="image" src="https://user-images.githubusercontent.com/20476674/247118361-0f78d6db-6896-4f8d-b1bd-8350192f7a4e.png" width=300>
<center>掩码图像</center>
<div align="center">
<img alt="image" src="https://user-images.githubusercontent.com/20476674/247118368-305a048d-ddc3-4a5f-8915-58591ef680f0.jpeg" width=300>
<center>参考图像</center>
<img alt="image" src="https://user-images.githubusercontent.com/20476674/247117963-e5b9b754-39a3-480b-a557-46a2f9310e79.png" width=300>
<center>生成图像</center>
</div>
</details>


<details><summary>&emsp;文本引导的图像变换（Image-to-Image Text-Guided Generation）</summary>

#### text_guided_image_inpainting-kandinsky2_2
```python
import numpy as np
import paddle

from ppdiffusers import KandinskyV22InpaintPipeline, KandinskyV22PriorPipeline
from ppdiffusers.utils import load_image

pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-prior", paddle_dtype=paddle.float16
)
prompt = "a hat"
image_emb, zero_image_emb = pipe_prior(prompt, return_dict=False)
pipe = KandinskyV22InpaintPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", paddle_dtype=paddle.float16
)
init_image = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png"
)
mask = np.zeros((768, 768), dtype=np.float32)
mask[:250, 250:-250] = 1
out = pipe(
    image=init_image,
    mask_image=mask,
    image_embeds=image_emb,
    negative_image_embeds=zero_image_emb,
    height=768,
    width=768,
    num_inference_steps=50,
)
image = out.images[0]
image.save("text_guided_image_inpainting-kandinsky2_2-result-cat_with_hat.png")
```
<div align="center">
<img width="300" alt="image" src="https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/64a943d5-167b-4433-91c3-3cf9279714db">
<center>原图像</center>
<img width="300" alt="image" src="https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/f469c127-52f4-4173-a693-c06b92a052aa">
<center>生成图像</center>
</div>

#### image_to_image_text_guided_generation-stable_diffusion
```python
import paddle

from ppdiffusers import StableDiffusionImg2ImgPipeline
from ppdiffusers.utils import load_image

# 加载pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# 下载初始图片
url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/sketch-mountains-input.png"

init_image = load_image(url).resize((768, 512))

prompt = "A fantasy landscape, trending on artstation"
# 使用fp16加快生成速度
with paddle.amp.auto_cast(True):
    image = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]

image.save("fantasy_landscape.png")
```
<div align="center">
<img width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/209327142-d8e1d0c7-3bf8-4a08-a0e8-b11451fc84d8.png">
<center>原图像</center>
<img width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/209325799-d9ff279b-0d57-435f-bda7-763e3323be23.png">
<center>生成图像</center>
</div>

#### image_to_image_text_guided_generation-stable_diffusion_xl
```python
import paddle
from ppdiffusers import StableDiffusionXLImg2ImgPipeline
from ppdiffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    paddle_dtype=paddle.float16,
    # from_hf_hub=True,
    # from_diffusers=True,
    variant="fp16"
)
url = "https://paddlenlp.bj.bcebos.com/models/community/westfish/develop-0-19-3/000000009.png"
init_image = load_image(url).convert("RGB")
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, image=init_image).images[0]
image.save('sdxl_image2image.png')
```
<div align="center">
<img width="300" alt="image" src="https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/41bd9381-2799-4bed-a5e2-ba312a2f8da9">
<center>原图像</center>
<img width="300" alt="image" src="https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/db672d03-2e3a-46ac-97fd-d80cca18dbbe">
<center>生成图像</center>
</div>

#### image_to_image_text_guided_generation-kandinsky2_2
```python
import paddle

from ppdiffusers import KandinskyV22Img2ImgPipeline, KandinskyV22PriorPipeline
from ppdiffusers.utils import load_image

pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-prior", paddle_dtype=paddle.float16
)
prompt = "A red cartoon frog, 4k"
image_emb, zero_image_emb = pipe_prior(prompt, return_dict=False)
pipe = KandinskyV22Img2ImgPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder", paddle_dtype=paddle.float16
)

init_image = load_image(
    "https://hf-mirror.com/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/frog.png"
)
image = pipe(
    image=init_image,
    image_embeds=image_emb,
    negative_image_embeds=zero_image_emb,
    height=768,
    width=768,
    num_inference_steps=100,
    strength=0.2,
).images
image[0].save("image_to_image_text_guided_generation-kandinsky2_2-result-red_frog.png")
```
<div align="center">
<img width="300" alt="image" src="https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/aae57109-94ad-408e-ae75-8cce650cebe5">
<center>原图像</center>
<img width="300" alt="image" src="https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/23cf2c4e-416f-4f21-82a6-e57de11b5e83">
<center>生成图像</center>
</div>

</details>
</details>

<details><summary>&emsp;文本图像双引导图像生成（Dual Text and Image Guided Generation）</summary>

#### dual_text_and_image_guided_generation-versatile_diffusion
```python
from ppdiffusers import VersatileDiffusionDualGuidedPipeline
from ppdiffusers.utils import load_image

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/benz.jpg"
image = load_image(url)
text = "a red car in the sun"

pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained("shi-labs/versatile-diffusion")
pipe.remove_unused_weights()

text_to_image_strength = 0.75
image = pipe(prompt=text, image=image, text_to_image_strength=text_to_image_strength).images[0]
image.save("versatile-diffusion-red_car.png")
```
<div align="center">
<img width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/209325965-2475e9c4-a524-4970-8498-dfe10ff9cf24.jpg" >
<center>原图像</center>
<img width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/209325293-049098d0-d591-4abc-b151-9291ac2636da.png">
<center>生成图像</center>
</div>
</details>

### 文本视频多模

<details open>
<summary>&emsp;文本条件的视频生成（Text-to-Video Generation）</summary>

#### text_to_video_generation-lvdm

```python
import paddle

from ppdiffusers import LVDMTextToVideoPipeline

# 加载模型和scheduler
pipe = LVDMTextToVideoPipeline.from_pretrained("westfish/lvdm_text2video_orig_webvid_2m")

# 执行pipeline进行推理
seed = 2013
generator = paddle.Generator().manual_seed(seed)
samples = pipe(
    prompt="cutting in kitchen",
    num_frames=16,
    height=256,
    width=256,
    num_inference_steps=50,
    generator=generator,
    guidance_scale=15,
    eta=1,
    save_dir=".",
    save_name="text_to_video_generation-lvdm-result-ddim_lvdm_text_to_video_ucf",
    encoder_type="2d",
    scale_factor=0.18215,
    shift_factor=0,
)
```
<div align="center">
<img width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/270906907-2b9d53c1-0272-4c7a-81b2-cd962d23bbee.gif">
</div>

#### text_to_video_generation-synth

```python
import imageio

from ppdiffusers import DPMSolverMultistepScheduler, TextToVideoSDPipeline

pipe = TextToVideoSDPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "An astronaut riding a horse."
video_frames = pipe(prompt, num_inference_steps=25).frames
imageio.mimsave("text_to_video_generation-synth-result-astronaut_riding_a_horse.mp4", video_frames, fps=8)
```
<div align="center">
<img width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/281259277-0ebe29a3-4eba-48ee-a98b-292e60de3c98.gif">
</div>


#### text_to_video_generation-synth with zeroscope_v2_XL

```python
import imageio

from ppdiffusers import DPMSolverMultistepScheduler, TextToVideoSDPipeline

# from ppdiffusers.utils import export_to_video

pipe = TextToVideoSDPipeline.from_pretrained("cerspense/zeroscope_v2_XL")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "An astronaut riding a horse."
video_frames = pipe(prompt, num_inference_steps=50, height=320, width=576, num_frames=24).frames
imageio.mimsave("text_to_video_generation-synth-result-astronaut_riding_a_horse.mp4", video_frames, fps=8)
```
<div align="center">
<img width="300" alt="image" src="https://github.com/PaddlePaddle/PaddleMIX/assets/35400185/43ebbca0-9f07-458b-809a-acf296a2539b">
</div>

#### text_to_video_generation-zero

```python
import imageio

# pip install imageio[ffmpeg]
import paddle

from ppdiffusers import TextToVideoZeroPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipe = TextToVideoZeroPipeline.from_pretrained(model_id, paddle_dtype=paddle.float16)

prompt = "A panda is playing guitar on times square"
result = pipe(prompt=prompt).images
result = [(r * 255).astype("uint8") for r in result]
imageio.mimsave("text_to_video_generation-zero-result-panda.mp4", result, fps=4)
```
<div align="center">
<img width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/246779321-c2b0c2b4-e383-40c7-a4d8-f417e8062b35.gif">
</div>

</details>

### 文本音频多模
<details>
<summary>&emsp;文本条件的音频生成（Text-to-Audio Generation）</summary>

#### text_to_audio_generation-audio_ldm

```python
import paddle
import scipy

from ppdiffusers import AudioLDM2Pipeline

pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2", paddle_dtype=paddle.float16)

prompt = "Musical constellations twinkling in the night sky, forming a cosmic melody."
negative_prompt = "Low quality."
audio = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=200, audio_length_in_s=10).audios[0]

output_path = f"{prompt}.wav"
# save the audio sample as a .wav file
scipy.io.wavfile.write(output_path, rate=16000, data=audio)
```
<div align = "center">
  <thead>
  </thead>
  <tbody>
   <tr>
      <td align = "center">
      <a href="https://paddlenlp.bj.bcebos.com/models/community/paddlemix/ppdiffusers/AudioLDM2-Music.wav" rel="nofollow">
            <img align="center" src="https://user-images.githubusercontent.com/20476674/209344877-edbf1c24-f08d-4e3b-88a4-a27e1fd0a858.png" width="200 style="max-width: 100%;"></a><br>
      </td>
    </tr>
  </tbody>
</div>
</details>

可以使用以下代码转换[huggingface](https://huggingface.co/docs/diffusers/api/pipelines/audioldm2)的模型，一键在paddle中使用
```python
pipe = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-music", from_hf_hub=True, from_diffusers=True).save_pretrained("cvssp/audioldm2-music")
```
### 图像

<details><summary>&emsp;无条件图像生成（Unconditional Image Generation）</summary>

#### unconditional_image_generation-latent_diffusion_uncond

```python
from ppdiffusers import LDMPipeline

# 加载模型和scheduler
pipe = LDMPipeline.from_pretrained("CompVis/ldm-celebahq-256")

# 执行pipeline进行推理
image = pipe(num_inference_steps=200).images[0]

# 保存图片
image.save("ldm_generated_image.png")
```
<div align="center">
<img width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/209327936-7fe914e0-0ea0-4e21-a433-24eaed6ee94c.png">
</div>
</details>

<details><summary>&emsp;超分（Super Superresolution）</summary>

#### super_resolution-latent_diffusion
```python
import paddle

from ppdiffusers import LDMSuperResolutionPipeline
from ppdiffusers.utils import load_image

# 加载pipeline
pipe = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages")

# 下载初始图片
url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"

init_image = load_image(url).resize((128, 128))
init_image.save("original-image.png")

# 使用fp16加快生成速度
with paddle.amp.auto_cast(True):
    image = pipe(init_image, num_inference_steps=100, eta=1).images[0]

image.save("super-resolution-image.png")
```
<div align="center">
<img  alt="image" src="https://user-images.githubusercontent.com/20476674/209328660-9700fdc3-72b3-43bd-9a00-23b370ba030b.png">
<center>原图像</center>
<img  alt="image" src="https://user-images.githubusercontent.com/20476674/209328479-4eaea5d8-aa4a-4f31-aa2a-b47e3c730f15.png">
<center>生成图像</center>
</div>
</details>


<details><summary>&emsp;图像编辑（Image Inpainting）</summary>

#### image_inpainting-repaint
```python
from ppdiffusers import RePaintPipeline, RePaintScheduler
from ppdiffusers.utils import load_image

img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/celeba_hq_256.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/mask_256.png"

# Load the original image and the mask as PIL images
original_image = load_image(img_url).resize((256, 256))
mask_image = load_image(mask_url).resize((256, 256))

scheduler = RePaintScheduler.from_pretrained("google/ddpm-ema-celebahq-256", subfolder="scheduler")
pipe = RePaintPipeline.from_pretrained("google/ddpm-ema-celebahq-256", scheduler=scheduler)

output = pipe(
    original_image=original_image,
    mask_image=mask_image,
    num_inference_steps=250,
    eta=0.0,
    jump_length=10,
    jump_n_sample=10,
)
inpainted_image = output.images[0]

inpainted_image.save("repaint-image.png")
```
<div align="center">
<img  alt="image" src="https://user-images.githubusercontent.com/20476674/209329052-b6fc2aaf-1a59-49a3-92ef-60180fdffd81.png">
<center>原图像</center>
<img  alt="image" src="https://user-images.githubusercontent.com/20476674/209329048-4fe12176-32a0-4800-98f2-49bd8d593799.png">
<center>mask图像</center>
<img  alt="image" src="https://user-images.githubusercontent.com/20476674/209329241-b7e4d99e-468a-4b95-8829-d77ee14bfe98.png">
<center>生成图像</center>
</div>
</details>



<details><summary>&emsp;图像变化（Image Variation）</summary>

#### image_variation-versatile_diffusion
```python
from ppdiffusers import VersatileDiffusionImageVariationPipeline
from ppdiffusers.utils import load_image

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/benz.jpg"
image = load_image(url)

pipe = VersatileDiffusionImageVariationPipeline.from_pretrained("shi-labs/versatile-diffusion")

image = pipe(image).images[0]
image.save("versatile-diffusion-car_variation.png")
```
<div align="center">
<img  width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/209331434-51f6cdbd-b8e4-4faa-8e49-1cc852e35603.jpg">
<center>原图像</center>
<img  width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/209331591-f6cc4cd8-8430-4627-8d22-bf404fb2bfdd.png">
<center>生成图像</center>
</div>
</details>





### 音频
<details>
<summary>&emsp;无条件音频生成（Unconditional Audio Generation）</summary>

#### unconditional_audio_generation-audio_diffusion

```python
from scipy.io.wavfile import write
from ppdiffusers import AudioDiffusionPipeline
import paddle

# 加载模型和scheduler
pipe = AudioDiffusionPipeline.from_pretrained("teticio/audio-diffusion-ddim-256")
pipe.set_progress_bar_config(disable=None)
generator = paddle.Generator().manual_seed(42)

output = pipe(generator=generator)
audio = output.audios[0]
image = output.images[0]

# 保存音频到本地
for i, audio in enumerate(audio):
    write(f"audio_diffusion_test{i}.wav", pipe.mel.config.sample_rate, audio.transpose())

# 保存图片
image.save("audio_diffusion_test.png")
```
<div align = "center">
  <thead>
  </thead>
  <tbody>
   <tr>
      <td align = "center">
      <a href="https://paddlenlp.bj.bcebos.com/models/community/teticio/data/audio_diffusion_test0.wav" rel="nofollow">
            <img align="center" src="https://user-images.githubusercontent.com/20476674/209344877-edbf1c24-f08d-4e3b-88a4-a27e1fd0a858.png" width="200 style="max-width: 100%;"></a><br>
      </td>
    </tr>
  </tbody>
</div>

<div align="center">
<img  width="300" alt="image" src="https://user-images.githubusercontent.com/20476674/209342125-93e8715e-895b-4115-9e1e-e65c6c2cd95a.png">
</div>


#### unconditional_audio_generation-spectrogram_diffusion

```python
import paddle
import scipy

from ppdiffusers import MidiProcessor, SpectrogramDiffusionPipeline
from ppdiffusers.utils.download_utils import ppdiffusers_url_download

# Download MIDI from: wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/beethoven_hammerklavier_2.mid
mid_file_path = ppdiffusers_url_download(
    "https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/beethoven_hammerklavier_2.mid", cache_dir="."
)
pipe = SpectrogramDiffusionPipeline.from_pretrained("google/music-spectrogram-diffusion", paddle_dtype=paddle.float16)
processor = MidiProcessor()
output = pipe(processor(mid_file_path))
audio = output.audios[0]

output_path = "unconditional_audio_generation-spectrogram_diffusion-result-beethoven_hammerklavier_2.wav"
# save the audio sample as a .wav file
scipy.io.wavfile.write(output_path, rate=16000, data=audio)
```
<div align = "center">
  <thead>
  </thead>
  <tbody>
   <tr>
      <td align = "center">
      <a href="https://paddlenlp.bj.bcebos.com/models/community/westfish/develop_ppdiffusers_data/beethoven_hammerklavier_2.wav" rel="nofollow">
            <img align="center" src="https://user-images.githubusercontent.com/20476674/209344877-edbf1c24-f08d-4e3b-88a4-a27e1fd0a858.png" width="200 style="max-width: 100%;"></a><br>
      </td>
    </tr>
  </tbody>
</div>
</details>



## License
PPDiffusers 遵循 [Apache-2.0开源协议](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/LICENSE)。

Stable Diffusion 遵循 [The CreativeML OpenRAIL M 开源协议](https://huggingface.co/spaces/CompVis/stable-diffusion-license)。
> The CreativeML OpenRAIL M is an [Open RAIL M license](https://www.licenses.ai/blog/2022/8/18/naming-convention-of-responsible-ai-licenses), adapted from the work that [BigScience](https://bigscience.huggingface.co/) and [the RAIL Initiative](https://www.licenses.ai/) are jointly carrying in the area of responsible AI licensing. See also [the article about the BLOOM Open RAIL license](https://bigscience.huggingface.co/blog/the-bigscience-rail-license) on which this license is based.

Stable Diffusion 3遵循 [Stability Community 开源协议](https://stability.ai/license)。
> Community License: Free for research, non-commercial, and commercial use for organisations or individuals with less than $1M annual revenue. You only need a paid Enterprise license if your yearly revenues exceed USD$1M and you use Stability AI models in commercial products or services. Read more: https://stability.ai/license

## Acknowledge
我们借鉴了🤗 Hugging Face的[Diffusers](https://github.com/huggingface/diffusers)关于预训练扩散模型使用的优秀设计，在此对Hugging Face作者及其开源社区表示感谢。

## Citation

```bibtex
@misc{ppdiffusers,
  author = {PaddlePaddle Authors},
  title = {PPDiffusers: State-of-the-art diffusion model toolkit based on PaddlePaddle},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers}}
}
```

# 版本更新信息

## 最新版本信息

### 3.0-beta(05/09/2024)

#### 多模态理解

1. 新增模型：Qwen2VL/Qwen2.5VL系列，DeepSeek-VL2, miniCPM-V 2.6, Janus系列，LLaVA-Critic, LLaVA-DenseConnector, LLaVA-OneVision, GOT-OCR2.0, mPLUG-Owl3
2. PP系列模型：发布自研PP-DocBee文档理解多模态大模型，在学术界权威的英文文档理解评测榜单上达到同参数量级别模型SOTA
3. 工具链升级：完善高性能推理部署，新增支持Qwen2.5VL系列，A800推理性能较vllm领先11.5%。LLaVA、InternVL2模型训练和推理适配昇腾910B

#### 多模态生成

1. 新增模型：Open-MAGVIT2，文生视频模型CogVideoX, HunyuanVideo
2. PP系列模型：发布自研可控视频模型PP-VCtrl，支持在多种控制条件下的视频生成
3. 工具链升级：发布ppdiffusers 0.29.1版本，新增对SD3 ControlNet和SD3.5的支持。SD3高性能推理性能打平TensorRT。SD3、SDXL模型LoRA训练和推理适配昇腾910B

### 2.0(07/26/2024)

#### 多模态理解

1. 新增模型：LLaVA: v1.5-7b, v1.5-13b, v1,6-7b，CogAgent, CogVLM, Qwen-VL, InternLM-XComposer2
2. 数据集增强：新增chatml_dataset图文对话数据读取方案，可自定义chat_template文件适配，支持混合数据集
3. 工具链升级：新增Auto模块，统一SFT训练流程，兼容全参数、lora训练。新增mixtoken训练策略，SFT吞吐量提升5.6倍。支持Qwen-VL，LLaVA推理部署，较torch推理性能提升2.38倍

#### 多模态生成

1. 视频生成能力：支持Sora相关技术，支持DiT、SiT、UViT训练推理，新增NaViT、MAGVIT-v2模型； 新增视频生成模型SVD、Open Sora，支持模型微调和推理； 新增姿态可控视频生成模型AnimateAnyone、即插即用视频生成模型AnimateDiff、GIF视频生成模型Hotshot-XL；
2. 文生图模型库：新增高速推理文图生成模型LCM，适配SD/SDXL训练和推理；
3. 工具链升级：发布ppdiffusers 0.24.1版本，新增peft，accelerate后端； 权重加载/保存全面升级，支持分布式、模型切片、safetensors等场景。
4. 生态兼容：提供基于ppdiffusers开发的ComfyUI插件，支持了常见的模型加载转换、文生图、图生图、图像局部修改等任务。新增Stable Diffusion 1.5系列节点；新增Stable Diffusion XL系列节点。新增4个图像生成的workflow案例。

#### DataCopilot（多模态数据处理工具箱）

1. 多模态数据集类型MMDataset，支持加载和导出Json、H5、Jsonl等多种数据存储格式，内置并发（map, filter）数据处理接口等
2. 多模态数据格式工具，支持自定义数据结构，数据转换，离线格式检查
3. 多模态数据分析工具，支持基本的统计信息，数据可视化功能，以及注册自定义功能

### 1.0(11/15/2023)

#### 核心能力

1. 大规模预训练: BLIP-2支持数据并行、sharding、模型并行，流水线并行训练；支持千亿参数规模训练; EVA-CLIP支持数据并行、sharding、模型并行训练; Stable Diffusion支持数据并行、sharding、BF16 O2训练; CLIP，Coca支持数据并行训练
2. 有监督精调: Stable Diffusion，SDXL 支持LoRA精调
3. 推理部署: 支持BLIP-2，miniGPT-4，Grounding DINO, SAM，Stable Diffusion动转静导出部署

#### 前沿模型
1. 新增CLIP系列跨模态大模型：CLIP，EVA-CLIP，Coca
2. 新增图生文跨模态大模型：BLIP-2，miniGPT-4，VisualGLM
3. 新增跨模态视觉模型：Grounding DINO， SAM
4. 新增融合更多模态大模型：ImageBind
5. 新增文生图模型：SDXL，支持Text2Image、Img2Img、Inpainting、InstructPix2Pix等任务，支持DreamBooth Lora训练； 新增UniDiffuser，通过统一的多模态扩散过程支持文生图、图生文等任务； 新增文本条件视频生成模型LVDM，支持训练与推理； 新增文图生成模型Kandinsky 2.2，Consistency models； Controlnet升级，支持ControlNetImg2Img、ControlNetInpaint、 StableDiffusionXLControlNet等。

#### 特色应用
1. 新增跨模态大模型应用流水线AppFlow
2. 新增基于chat的图像编辑应用
3. 新增自动标注应用

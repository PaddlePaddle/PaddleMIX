# Aria

## 1. 模型介绍

[rhymes-ai/Aria](https://github.com/rhymes-ai/Aria) 是一款原生多模态 MoE（混合专家）模型，具有在各种多模态和语言任务上表现出色的能力，特别是在视频和文档理解方面优于其他模型。它支持64K tokens的长多模态上下文窗口，并且每个token激活3.9B参数，能够实现快速的推理速度和低的微调成本。

- ✨**模型架构**✨：Aria的核心是一个细粒度的混合专家（MoE）解码器，通过专家特化实现更高效的参数利用率，从而在训练和推理速度上优于密集解码器。Aria MoE每个文本token激活3.5B参数，总参数量为24.9B。具有可变长度、大小和比例的视觉输入通过一个轻量级的视觉编码器（438M参数）转化为视觉token。Aria支持64k tokens的长多模态上下文窗口。

- ✨**数据**✨：Aria在6.4T语言tokens和400B多模态tokens上进行了预训练。Aria开发了一套严格的流程，从多样化的来源筛选高质量数据。多模态预训练数据包括四大类：来自公共爬虫的图文交替序列、合成图像标题、文档转录和问答对等。

- ✨**训练Pipeline**✨：Aria设计了一个四阶段的训练管道，包括语言预训练、多模态预训练、多模态长上下文预训练和多模态后训练。每个阶段旨在逐步增强Aria的某些能力，同时保持在早期阶段获得的能力。Aria的训练管道高效且有效地利用数据和计算资源，以最大化模型性能。


**本仓库支持的模型权重:**

| Model              |
|--------------------|
| rhymes-ai/Aria  |


注意：与huggingface权重同名，但权重为paddle框架的Tensor，使用`xxx.from_pretrained("rhymes-ai/Aria")`即可自动下载该权重文件夹到缓存目录。


## 2 环境准备

1）[安装 PaddleMIX 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/develop?tab=readme-ov-file#%E5%AE%89%E8%A3%85)


注意：Python版本最好为3.10及以上版本。

## 3 快速开始

### 推理
> 注：在V100上运行以下代码需要指定dtype="float16"
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
--dtype bfloat16 \
--base_model_path path_to_Aria \
--tokenizer_path path_to_Aria  \
--image_path ../../demo_images/examples_image1.jpg \
--prompt "what is the image?"
```
!['LLaVa-1.5'](../../demo_images/examples_image1.jpg)
**Prompt:** what is the image?

**Aria output:** The image shows a red panda resting on a wooden structure. The red panda has a distinctive reddish-brown fur with a white face and black markings around its eyes. It appears to be in a natural or semi-natural environment, possibly a zoo or wildlife sanctuary, with greenery in the background.


### 参考文献
```BibTeX
@article{aria,
  title={Aria: An Open Multimodal Native Mixture-of-Experts Model},
  author={Dongxu Li and Yudong Liu and Haoning Wu and Yue Wang and Zhiqi Shen and Bowen Qu and Xinyao Niu and Guoyin Wang and Bei Chen and Junnan Li},
  year={2024},
  journal={arXiv preprint arXiv:2410.05993},
}
```

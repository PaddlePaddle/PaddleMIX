# InternLM-XComposer2.5

## 1. 模型介绍

**浦语·灵笔2.5**是基于[书生·浦语2](https://github.com/InternLM/InternLM/tree/main)大语言模型研发的突破性的图文多模态大模型，仅使用 7B LLM 后端就达到了 GPT-4V 级别的能力。浦语·灵笔2.5使用24K交错的图像-文本上下文进行训练，通过RoPE外推可以无缝扩展到96K长的上下文。这种长上下文能力使浦语·灵笔2.5在需要广泛输入和输出上下文的任务中表现出色。

- **超高分辨率理解**：浦语·灵笔2.5使用560×560分辨率的ViT视觉编码器增强了IXC2-4KHD中提出的动态分辨率解决方案，支持具有任意纵横比的高分辨率图像。
- **细粒度视频理解**：浦语·灵笔2.5将视频视为由数十到数千帧组成的超高分辨率复合图像，从而通过密集采样和每帧更高的分辨率捕捉细节。
- **多轮多图像对话**：浦语·灵笔2.5支持自由形式的多轮多图像对话，使其能够在多轮对话中与人类自然互动。
- **网页制作**：浦语·灵笔2.5可以通过遵循文本-图像指令来创建网页，包括源代码（HTML、CSS和JavaScript）的组合。
- **高质量文本-图像文章创作**：浦语·灵笔2.5利用特别设计的“思维链”（CoT）和“直接偏好优化”（DPO）技术，显著提高了其创作内容的质量。
- **出色的性能**：浦语·灵笔2.5在28个基准测试中进行了评估，在16个基准测试上优于现有的开源先进模型。它还在16个关键任务上超越或与GPT-4V和Gemini Pro表现相近。

---
## 2 环境准备
1）[安装PaddlePaddle](https://github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
- **python >= 3.10**
- **paddlepaddle-gpu 要求是3.0.0b2或develop版本**
- **硬件配置至少A100**
```bash
conda create -n internlm python=3.11 -y
conda activate internlm
```
2） [安装PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

3）[安装 PaddleMix 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/b4f97ff859e1964c839fc5fab94f7ba63b1e5959?tab=readme-ov-file#%E5%AE%89%E8%A3%85)
# 安装 PaddleMIX
```bash
pip install -e .
```
# 安装ppdiffusers
```bash
cd ppdiffusers
pip install -e .
cd ..
```
# 安装flash_attn
```bash
pip install --no-index  /home/aistudio/PaddleMIX/ppdiffusers/flash-attention/flash_attn-2.7.3-cp311-cp311-linux_x86_64.whl.whl
```
## 3 模型转换

将torch模型转换成paddle模型，请采用下述命令。

```bash

python torch2paddle.py

```

生成权重文件model_state.pdparams、tokenizer.model、tokenizer_config.json、special_tokens_map.json等
![image](https://github.com/user-attachments/assets/d71f6ea0-c2c0-44fd-ac43-d6b762b2042f)

## 4 模型推理


```bash
python paddlemix/examples/internlm_xcomposer2/chat_demo.py \
--model_name_or_path "/home/aistudio/internlm-xcomposer2d5-7b-paddle" \
--image_path "/home/aistudio/internlm-xcomposer2-4bit/panda.jpg" \
--text "Please describe this image in detail."
```
![图片](https://github.com/user-attachments/assets/bc63f0d6-841e-488d-9557-31b6a88bb294)

**Prompt:**

>please describe the image in detail

**Result:**

>The image presents a panda bear in its natural habitat, which is characterized by a lush environment with green grass and various plants. The panda's fur is predominantly black with distinctive white patches around its eyes, ears, and shoulders. It stands on all fours, suggesting a moment of rest or observation. The panda's posture is upright, and it appears to be looking downwards, possibly at something of interest on the ground. The blurred background indicates movement, either from the panda itself or from the camera's perspective, adding a dynamic element to the scene. The overall composition of the image captures the essence of the panda's serene yet curious nature within its natural setting.

### 参考文献
```BibTeX
@article{internlmxcomposer2,
      title={InternLM-XComposer2: Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Model},
      author={Xiaoyi Dong and Pan Zhang and Yuhang Zang and Yuhang Cao and Bin Wang and Linke Ouyang and Xilin Wei and Songyang Zhang and Haodong Duan and Maosong Cao and Wenwei Zhang and Yining Li and Hang Yan and Yang Gao and Xinyue Zhang and Wei Li and Jingwen Li and Kai Chen and Conghui He and Xingcheng Zhang and Yu Qiao and Dahua Lin and Jiaqi Wang},
      journal={arXiv preprint arXiv:2401.16420},
      year={2024}
}
```

# Janus/JanusFlow

## 1. 模型介绍

[Janus/JanusFlow](https://github.com/deepseek-ai/Janus) 是一种统一多模态理解和生成任务的自回归模型，之前的方法主要依赖于一个单一的视觉编码器用于理解和生成，而忽略了多模态理解和生成需要不同级别信息粒度。

Janus/JanusFlow 将视觉编码器解耦为理解和生成编码器，同时仍然使用统一的自回归 transformer 进行处理。对于理解任务，使用 LLM 中的预测头进行文本预测。对于生成任务，使用中的 VQVAE 或 VAE 的解码器用于图像生成。这种解耦的设计使得Janus模型在生成任务上超过了 SDv1.5和 SDXL 等文生图模型。在多个多模态理解评测中超过以往统一的模型，性能接近为特定任务训练的模型。

![Overview of Janus](https://ai-studio-static-online.cdn.bcebos.com/ea0703505b3b40ad923981dbddda20973c81da7a36194e3abc75ad1d9b870ab4)
注：以上为 Janus 的整体架构图

**本仓库支持的模型权重:**

| Model              |
|--------------------|
| deepseek-ai/Janus-1.3B  |
| deepseek-ai/JanusFlow-1.3B  |

注意：与huggingface权重同名，但权重为paddle框架的Tensor，使用`xxx.from_pretrained("deepseek-ai/Janus-1.3B")`即可自动下载该权重文件夹到缓存目录。


## 2 环境准备

1）[安装 PaddleMIX 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/develop?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

2）pip install pillow tqdm paddlenlp==3.0.0b2

注意：Python版本最好为3.10及以上版本。

## 3 快速开始

### 推理
> 注：在V100上运行以下代码需要指定dtype="float16"
```bash
# Janus understanding
python paddlemix/examples/janus/run_understanding_inference.py \
    --model_path="deepseek-ai/Janus-1.3B" \
    --image_file="paddlemix/demo_images/examples_image1.jpg" \
    --question="What is shown in this image?" \
    --dtype="bfloat16"

# Janus generation
python paddlemix/examples/janus/run_generation_inference.py \
    --model_path="deepseek-ai/Janus-1.3B" \
    --prompt="A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair" \
    --dtype="bfloat16"

# JanusFlow generation
python paddlemix/examples/janus/run_generation_inference_janusflow.py \
    --model_path="deepseek-ai/JanusFlow-1.3B" \
    --inference_step=30 \
    --prompt="A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair" \
    --dtype="bfloat16"

# Janus interactivechat
python paddlemix/examples/janus/run_interactivechat.py \
    --model_path="deepseek-ai/Janus-1.3B" \
    --dtype="bfloat16"
```

### 效果展示

1）Janus understanding:

You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.

User: <image_placeholder>
What is shown in this image?

!['LLaVa-1.5'](https://ai-studio-static-online.cdn.bcebos.com/41e8d021ea7e4efd801fb197690d704a325c490ee17547c29f61a5d0075137d5)

Assistant: The image shows a radar chart comparing the performance of different models across various metrics. The chart includes the following metrics: VQA-2, GOA, LQA, V2VWiz, LLaVA-Bench, LLaVA-Bench-CN, MM-Vet, LLaVA-1.5, and POPE. The metrics are plotted on a polar scale, with different colors representing different models: BLIP 2 (blue), InstructBLIP (green), and Owen-Vi-Chat (orange). Each model's performance is represented by a line on the chart.

2）Janus/JanusFlow generation:

Prompt:A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair

![Janus/JanusFlow generation](https://ai-studio-static-online.cdn.bcebos.com/c453a3536ab84c30ae416e0cea9c139abe3f233b5c0748b28eac72d89e6759f4)




### 参考文献
```BibTeX
@article{wu2024janus,
  title={Janus: Decoupling visual encoding for unified multimodal understanding and generation},
  author={Wu, Chengyue and Chen, Xiaokang and Wu, Zhiyu and Ma, Yiyang and Liu, Xingchao and Pan, Zizheng and Liu, Wen and Xie, Zhenda and Yu, Xingkai and Ruan, Chong and others},
  journal={arXiv preprint arXiv:2410.13848},
  year={2024}
}

@misc{ma2024janusflow,
      title={JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation},
      author={Yiyang Ma and Xingchao Liu and Xiaokang Chen and Wen Liu and Chengyue Wu and Zhiyu Wu and Zizheng Pan and Zhenda Xie and Haowei Zhang and Xingkai yu and Liang Zhao and Yisong Wang and Jiaying Liu and Chong Ruan},
      journal={arXiv preprint arXiv:2411.07975},
      year={2024}
}
```

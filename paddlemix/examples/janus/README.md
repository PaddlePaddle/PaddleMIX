# Janus/JanusFlow

## 1. 模型介绍

[Janus/JanusFlow](https://github.com/deepseek-ai/Janus) 将视觉编码解耦到单独的路径中，同时仍然使用单个统一的转换器架构进行处理，解决了以前方法的局限性。解耦不仅缓解了视觉编码器在理解和生成中的角色冲突，还增强了框架的灵活性。


**本仓库支持的模型权重:**

| Model              |
|--------------------|
| deepseek-ai/Janus-1.3B  |
| deepseek-ai/JanusFlow-1.3B  |

注意：与huggingface权重同名，但权重为paddle框架的Tensor，使用`xxx.from_pretrained("deepseek-ai/Janus-1.3B")`即可自动下载该权重文件夹到缓存目录。


## 2 环境准备

1）[安装 PaddleMIX 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/b4f97ff859e1964c839fc5fab94f7ba63b1e5959?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

2) pip install pillow tqdm paddlenlp==3.0.0b2
注意：Python版本最好为3.10及以上版本。

## 3 快速开始

### 推理
```bash
# Janus understanding
python paddlemix/examples/janus/run_understanding_inference.py \
    --model_path="deepseek-ai/Janus-1.3B" \
    --image_file="paddlemix/demo_images/examples_image1.jpg" \
    --question="What is shown in this image?" \

# Janus generation
python paddlemix/examples/janus/run_generation_inference.py \
    --model_path="deepseek-ai/Janus-1.3B" \
    --prompt="A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair"

# JanusFlow generation
python paddlemix/examples/janus/run_generation_inference_janusflow.py \
    --model_path="deepseek-ai/JanusFlow-1.3B" \
    --inference_step=30 \
    --prompt="A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair"

# Janus interactivechat
python paddlemix/examples/janus/run_interactivechat.py \
    --model_path="deepseek-ai/Janus-1.3B" \

```

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

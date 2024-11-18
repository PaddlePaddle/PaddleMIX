# InternVL2 模型

## 1. 模型介绍

[InternVL2](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/)是 InternVL 系列多模态大模型的最新成员。InternVL2 包含多个经过指令微调的模型，参数量从 1B 到 76B 不等。在开源模型中，InternVL2 在文档和图表理解、信息图表问答、场景文本理解和 OCR 任务、科学和数学问题解决等方面表现出色。
[InternVL2-MPO](https://internvl.github.io/blog/2024-11-14-InternVL-2.0-MPO/)是混合偏好优化后的InternVL2模型，基于InternVL2在多个基准测试中表现出了改进的性能，特别是在多模态推理方面。


**本仓库支持的模型权重:**

| Model              |
|--------------------|
| OpenGVLab/InternVL2-1B  |
| OpenGVLab/InternVL2-2B  |
| OpenGVLab/InternVL2-8B  |
| OpenGVLab/InternVL2-26B |
| OpenGVLab/InternVL2-40B |
| OpenGVLab/InternVL2-8B-MPO |

注意：与huggingface权重同名，但权重为paddle框架的Tensor，使用`xxx.from_pretrained("OpenGVLab/InternVL2-2B")`即可自动下载该权重文件夹到缓存目录。


## 2 环境准备

1）[安装PaddleNLP develop分支](https://github.com/PaddlePaddle/PaddleNLP?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

2）[安装 PaddleMIX 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/b4f97ff859e1964c839fc5fab94f7ba63b1e5959?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

注意：Python版本最好为3.10及以上版本。

## 3. 模型推理预测

### 3.1. 图片预测
```bash
python paddlemix/examples/internvl2/chat_demo.py \
    --model_name_or_path "OpenGVLab/InternVL2-8B" \
    --image_path 'paddlemix/demo_images/examples_image1.jpg' \
    --text "Please describe this image in detail."
```
可配置参数说明：
  * `model_name_or_path`: 指定 internvl2 的模型名字或权重路径以及tokenizer组件，默认 OpenGVLab/InternVL2-8B，也可选择 OpenGVLab/InternVL2-2B
  * `image_path`: 指定图片路径
  * `text`: 用户指令, 例如 "Please describe this image in detail."

### 3.2. 视频预测
```bash
python paddlemix/examples/internvl2/chat_demo_video.py \
    --model_name_or_path "OpenGVLab/InternVL2-8B" \
    --video_path 'paddlemix/demo_images/red-panda.mp4' \
    --text "Please describe this video in detail."
```
可配置参数说明：
  * `model_name_or_path`: 指定 internvl2 的模型名字或权重路径以及tokenizer组件，默认 OpenGVLab/InternVL2-8B，也可选择 OpenGVLab/InternVL2-2B
  * `video_path`: 指定视频路径
  * `text`: 用户指令, 例如 "Please describe this video in detail."


## 4 模型微调

### 4.1 微调数据准备

SFT数据集采用 InternVL2 官方公布的1.3M的SFT数据集，包括了`sharegpt4v`、`llava_instruct_150k_zh`、`dvqa`、`chartqa`、`ai2d`、`docvqa`、`geoqa+`、`synthdog_en`等。

PaddleMIX团队整理后的下载链接为：
```
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground.tar # 50G
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/LLaVA/LLaVA-SFT.tar # 116G
```

下载后可解压或软链接在 PaddleMIX/ 目录下。

PaddleMIX团队也提供了其中单独的`chartqa`数据集的下载链接，作为训练示例：
```
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground/data/chartqa.tar
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground/opensource.tar
```
chartqa.tar需下载解压在playground/data/目录下，opensource.tar需下载解压在playground/目录下，opensource里是数据标注的jsonl文件。

### 4.2 微调命令

注意：此微调训练为全参数微调，冻结视觉编码器而放开LLM训练，2B模型微调训练的显存大小约为40G，8B模型微调训练的显存大小约为80G。

```bash
# 1B
sh paddlemix/examples/internvl2/shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_full.sh

# 2B
sh paddlemix/examples/internvl2/shell/internvl2.0/2nd_finetune/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_full.sh

# 8B
sh paddlemix/examples/internvl2/shell/internvl2.0/2nd_finetune/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_full.sh
```

### 4.3 微调后使用

同按步骤3中的模型推理预测，只需将`model_name_or_path`参数修改为微调后的模型路径即可。

```bash
python paddlemix/examples/internvl2/chat_demo.py \
    --model_name_or_path "your_checkpoints" \
    --image_path 'paddlemix/demo_images/examples_image1.jpg' \
    --text "Please describe this image in detail."
```

### 4.4 MiniMonkey 模型

[MiniMonkey](https://github.com/Yuliang-Liu/Monkey/blob/main/project/mini_monkey/) 是基于 InternVL2 的专用于OCR文档理解的多模态大模型。
具体使用请参照[minimonkey](../minimonkey/)


## 5 NPU硬件训练
请参照[tools](../../tools/README.md)进行NPU硬件Paddle安装和环境变量设置。
配置完成后可直接按步骤4中的微调命令进行训练。


### 参考文献
```BibTeX
@article{chen2023internvl,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and Li, Bin and Luo, Ping and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2312.14238},
  year={2023}
}

@article{chen2024far,
  title={How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites},
  author={Chen, Zhe and Wang, Weiyun and Tian, Hao and Ye, Shenglong and Gao, Zhangwei and Cui, Erfei and Tong, Wenwen and Hu, Kongzhi and Luo, Jiapeng and Ma, Zheng and others},
  journal={arXiv preprint arXiv:2404.16821},
  year={2024}
}
```

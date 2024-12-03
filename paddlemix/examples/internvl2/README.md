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


## 2 环境准备(如符合则跳过)
* 通过 `git clone` 命令拉取 PaddleMIX 源码，并安装必要的依赖库。
* Python版本最好为3.10及以上版本。
* PaddleNLP版本最好为3.0及以上。
> 注：本模型训练与推理需要依赖 CUDA 11.2 及以上版本，如果本地机器不符合要求，建议前往 [AI Studio](https://aistudio.baidu.com/index) 进行模型训练、推理任务。推荐使用Linux系统，Windows系统未经过系统测试。

## 3. 模型推理预测

### 3.1. 图片预测
<div style="width: 300px; height: 300px; margin: 0 auto; text-align: center;">
    <img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleMIX/develop/paddlemix/demo_images/examples_image1.jpg" 
         alt="examples_image1" 
         style="max-width: 100%; height: auto;"/>
</div>

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

<div style="display: flex; justify-content: center; align-items: center;">
    <video width="300" height="300" controls>
        <source src="https://raw.githubusercontent.com/PaddlePaddle/PaddleMIX/develop/paddlemix/demo_images/red-panda.mp4" type="video/mp4">
    </video>
</div>

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

#### 数据集下载
SFT数据集采用 InternVL2 官方公布的1.3M的SFT数据集，总共包含约 120 万个完全开源的视觉指令调优样本。从宏观角度来看，在 ShareGPT-4V 的基础上，还整合了 LLaVA-ZH、DVQA、ChartQA、AI2D、DocVQA、GeoQA+ 和 SynthDoG-EN。大部分数据与 LLaVA-NeXT 保持一致。

PaddleMIX团队整理后的下载链接为：
```
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground.tar # 50G
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/LLaVA/LLaVA-SFT.tar # 116G
```
下载后可解压或软链接在 PaddleMIX/ 目录下。

**PaddleMIX团队也提供了其中单独的`chartqa`数据集的下载链接，作为训练示例（入门推荐）：**
```
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground/data/chartqa.tar # 1.1G
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground/opensource.tar # 5.2G
```

chartqa.tar需下载解压在playground/data/目录下，opensource.tar需下载解压在playground/目录下，opensource里是数据标注的jsonl文件。

----
#### 数据集介绍
数据集包含以下数据集：
* AI2D：ai2d_images（由 InternLM-XComposer 提供）
* ChartQA：ChartQA Dataset
* COCO：train2017
* DocVQA：train、val、test
* DVQA：images
* GQA：images
* LLaVA-Pretrain：images
* OCR-VQA：下载脚本。我们将所有文件保存为 `.jpg`
* SAM：目前我们仅使用 000000~000050.tar。您可以从这里快速下载 9K 张图像。
* TextVQA：trainvalimages
* SynthDoG-EN：目前我们仅使用 00000~00004 parquet 文件，共 3 万张图像。我们提供了转换后的图像。
* VisualGenome：part1、part2
* WebData：images。仅供学术使用。
* GeoQA+：images。我们已转换数据格式并重新发布。


#### 数据集组织结构
按照以下方式在 `playground/data` 中组织数据：

```
playground/
├── opensource
│   ├── ai2d_train_12k.jsonl
│   ├── chartqa_train_18k.jsonl
│   ├── docvqa_train_10k.jsonl
│   ├── dvqa_train_200k.jsonl
│   ├── geoqa+.jsonl
│   ├── llava_instruct_150k_zh.jsonl
│   ├── sharegpt4v_instruct_gpt4-vision_cap100k.jsonl
│   ├── sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.jsonl
│   └── synthdog_en.jsonl
├── data
│   ├── ai2d
│   │   ├── abc_images
│   │   └── images
│   ├── chartqa
│   │   ├── test
│   │   ├── train
│   │   └── val
│   ├── coco
│   │   └── train2017
│   ├── docvqa
│   │   ├── test
│   │   ├── train
│   │   └── val
│   ├── dvqa
│   │   └── images
│   ├── gqa
│   │   └── images
│   ├── llava
│   │   └── llava_pretrain
│   │       └── images
│   ├── ocr_vqa
│   │   └── images
│   ├── sam
│   │   └── images
│   ├── share_textvqa
│   │   └── images
│   ├── synthdog-en
│   │   └── images
│   ├── textvqa
│   │   └── train_images
│   ├── vg
│   │   ├── VG_100K
│   │   └── VG_100K_2
│   ├── web-celebrity
│   │   └── images
│   ├── web-landmark
│   │   └── images
│   ├── wikiart
│   │   └── images
│   ├── geoqa+
│   │   └── images
```

### 4.2 微调命令

注意：此微调训练为全参数微调，冻结视觉编码器而放开LLM训练，1B V100 32G可跑。
2B模型微调训练的显存大小约为40G，8B模型微调训练的显存大小约为80G。

```bash
# 单卡
# 1B
sh paddlemix/examples/internvl2/shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_full.sh

## 多卡
# 2B
sh paddlemix/examples/internvl2/shell/internvl2.0/2nd_finetune/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_full.sh 

## 多卡
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

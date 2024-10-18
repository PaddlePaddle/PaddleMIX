# MiniMonkey 模型

## 1. 模型介绍

[MiniMonkey](https://github.com/Yuliang-Liu/Monkey/blob/main/project/mini_monkey/) 是基于 InternVL2 的专用于OCR文档理解的多模态大模型。


## 2 环境准备

1）[安装PaddleNLP develop分支](https://github.com/PaddlePaddle/PaddleNLP?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

2）[安装 PaddleMIX 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/b4f97ff859e1964c839fc5fab94f7ba63b1e5959?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

注意：Python版本最好为3.10及以上版本。

## 3. 模型推理预测

## 3.1. 图片预测
```bash
python paddlemix/examples/minimonkey/chat_demo_minimonkey.py \
    --model_name_or_path "HUST-VLRLab/Mini-Monkey" \
    --image_path 'path/to/image.jpg' \
    --text "Read the all text in the image."
```
可配置参数说明：
  * `model_name_or_path`: 指定 minimonkey 的模型名字或权重路径以及tokenizer组件，默认 HUST-VLRLab/Mini-Monkey
  * `image_path`: 指定图片路径
  * `text`: 用户指令, 例如 "Read the all text in the image."

## 4 模型微调

### 4.1 微调数据准备

SFT数据集采用 InternVL2 官方公布的1.3M的SFT数据集中的`dvqa`、`chartqa`、`ai2d`、`docvqa`、`geoqa+`、`synthdog_en`共6个。

PaddleMIX团队整理后的下载链接为：
```
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground.tar # 50G
```

下载后可解压或软链接在 PaddleMIX/ 目录下。

PaddleMIX团队也提供了其中单独的`chartqa`数据集的下载链接，作为训练示例：
```
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground/data/chartqa.tar
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground/opensource.tar
```
chartqa.tar需下载解压在playground/data/目录下，opensource.tar需下载解压在playground/目录下，opensource里是数据标注的jsonl文件。

### 4.2 微调命令

注意：此微调训练为全参数微调，冻结视觉编码器而放开LLM训练，2B模型微调训练的显存大小约为40G。

```bash
sh paddlemix/examples/minimonkey/shell/internvl2.0/2nd_finetune/minimonkey_2b_internlm2_1_8b_dynamic_res_2nd_finetune_full.sh
```


### 参考文献
```BibTeX
@article{huang2024mini,
  title={Mini-Monkey: Multi-Scale Adaptive Cropping for Multimodal Large Language Models},
  author={Huang, Mingxin and Liu, Yuliang and Liang, Dingkang and Jin, Lianwen and Bai, Xiang},
  journal={arXiv preprint arXiv:2408.02034},
  year={2024}
}
```

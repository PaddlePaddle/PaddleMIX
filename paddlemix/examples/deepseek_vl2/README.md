# DeepSeek-VL2

## 1. 模型介绍
[DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2)是一种基于大型混合专家（Mixture-of-Experts，MoE）视觉语言模型，相较于其前身DeepSeek-VL有了显著提升。DeepSeek-VL2在各种任务中展现出了卓越的能力，包括但不限于视觉问答、光学字符识别、文档/表格/图表理解以及视觉定位。我们的模型系列包含三种变体：DeepSeek-VL2-Tiny、DeepSeek-VL2-Small和DeepSeek-VL2，分别拥有10亿、28亿和45亿个激活参数。与现有的开源密集型和基于MoE的模型相比，DeepSeek-VL2在激活参数相似或更少的情况下，实现了具有竞争力甚至最先进的性能。
![Overview of DeepSeek-VL2](https://github.com/user-attachments/assets/926928a3-bad2-4c5b-8f45-d0c9a7661f34)
注：以上为 DeepSeek-VL2 的整体架构图引用自论文。

**本仓库支持的模型权重:**

| Model              |
|--------------------|
| deepseek-ai/deepseek-vl2-tiny  |
| deepseek-ai/deepseek-vl2-small  |
| deepseek-ai/deepseek-vl2  |

注意：与huggingface权重同名，但权重为paddle框架的Tensor，使用`xxx.from_pretrained("deepseek-ai/deepseek-vl2-tiny")`即可自动下载该权重文件夹到缓存目录。

## 2 环境准备

1）[安装PaddlePaddle](https://github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
- **python >= 3.10**
- **paddlepaddle-gpu 要求是3.0.0b2或develop版本**
```bash
# 提供三种 PaddlePaddle 安装命令示例，也可参考PaddleMIX主页的安装教程进行安装

# 3.0.0b2版本安装示例 (CUDA 11.8)
python -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Develop 版本安装示例
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

# sh 脚本快速安装
sh build_paddle_env.sh
```

2）[安装PaddleMIX环境依赖包](https://github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
- **paddlenlp >= 3.0.0b3**

```bash
# 提供两种 PaddleMIX 依赖安装命令示例

# pip 安装示例，安装paddlemix、ppdiffusers、项目依赖、paddlenlp
python -m pip install -e . --user
python -m pip install -e ppdiffusers --user
python -m pip install -r requirements.txt --user
python -m pip install paddlenlp==3.0.0b3 --user

# sh 脚本快速安装
sh build_env.sh
```

> 注：
* 请确保安装了以上依赖，否则无法运行。同时，需要安装 paddlemix/external_ops 下的自定义OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置PYTHONPATH
* (默认开启flash_attn)使用flash_attn 要求A100/A800显卡或者H20显卡。V100请用float16推理。

## 3 快速开始

### 推理
> 注：在V100上运行以下代码需要指定dtype="float16", 如果需要使用deepseek-vl2-small模型，需要修改model_path为"deepseek-ai/deepseek-vl2-small"

```bash
# Deepseek-vl2-tiny single image understanding
python paddlemix/examples/deepseek_vl2/single_image_infer.py \
    --model_path="deepseek-ai/deepseek-vl2-tiny" \
    --image_file="paddlemix/demo_images/examples_image2.jpg" \
    --question="The Panda" \
    --dtype="bfloat16"

# Deepseek-vl2-tiny multi image understanding
python paddlemix/examples/deepseek_vl2/multi_image_infer.py \
    --model_path="deepseek-ai/deepseek-vl2-tiny" \
    --image_file_1="paddlemix/demo_images/examples_image1.jpg" \
    --image_file_2="paddlemix/demo_images/examples_image2.jpg" \
    --image_file_3="paddlemix/demo_images/twitter3.jpeg" \
    --question="Can you tell me what are in the images?" \
    --dtype="bfloat16"

# Deepseek-vl2-tiny increment prefilling kv cache inference
python paddlemix/examples/deepseek_vl2/increment_prefilling_infer.py \
    --model_path="deepseek-ai/deepseek-vl2-tiny" \
    --image_file_1="paddlemix/demo_images/examples_image1.jpg" \
    --image_file_2="paddlemix/demo_images/examples_image2.jpg" \
    --image_file_3="paddlemix/demo_images/twitter3.jpeg" \
    --question="Can you tell me what are in the images?" \
    --dtype="bfloat16"
```

### 结果展示
1） DeepSeek-VL2-tiny Single Image Understanding

![panda](https://github.com/user-attachments/assets/6f66021c-c2fe-4231-a466-6b3747c26f7c)
```
<|User|>: <image>
<|ref|>The Panda<|/ref|>.
<|Assistant|>: <|ref|>The Panda<|/ref|><|det|>[[100, 192, 998, 998]]<|/det|><｜end▁of▁sentence｜>
```

2) DeepSeek-VL2-tiny Multi Image Understanding
```
<|User|>: This is image_1: <image>
This is image_2: <image>
This is image_3: <image>
 Can you tell me what are in the images?

<|Assistant|>: The first image shows a red panda resting on a wooden platform. The second image features a giant panda sitting among bamboo plants. The third image captures a rocket launch at night, with the bright trail of the rocket illuminating the sky.<｜end▁of▁sentence｜>
```
![mutli-infer](https://github.com/user-attachments/assets/4a1ade41-90ed-4d04-949a-90c3b54bdf78)


## 4 训练微调

### 数据准备

PaddleMIX团队整理了`chartqa`数据集作为小型的示例数据集，下载链接为：

```bash
wget https://paddlenlp.bj.bcebos.com/models/community/paddlemix/benchmark/playground.tar # 1.0G
```

playground/目录下包括了图片目录`data/chartqa/`和标注目录`opensource_json/`，详见`paddlemix/examples/qwen2_5_vl/configs/demo_chartqa_500.json`。

### 训练命令

```bash
# DeepSeek-VL2-tiny LoRA Training
sh paddlemix/examples/deepseek_vl2/shell/deepseek_vl2_tiny_lora_bs16_1e5.sh

# DeepSeek-VL2-tiny SFT Training
sh paddlemix/examples/deepseek_vl2/shell/deepseek_vl2_tiny_sft_bs16_1e5.sh
```

### LoRA参数合并

```bash
# tiny
python paddlemix/examples/deepseek_vl2/lora_merge.py \
    --model_name_or_path deepseek-ai/deepseek-vl2-tiny \
    --lora_path work_dirs/deepseekvl2_tiny_lora_bs16_1e5/checkpoint-xx \
    --merge_model_path work_dirs/lora_merge_deepseekvl2_tiny_lora_bs16_1e5 \
    --device "gpu"
```

## 参考文献
```BibTeX
@misc{wu2024deepseekvl2mixtureofexpertsvisionlanguagemodels,
      title={DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding},
      author={Zhiyu Wu and Xiaokang Chen and Zizheng Pan and Xingchao Liu and Wen Liu and Damai Dai and Huazuo Gao and Yiyang Ma and Chengyue Wu and Bingxuan Wang and Zhenda Xie and Yu Wu and Kai Hu and Jiawei Wang and Yaofeng Sun and Yukun Li and Yishi Piao and Kang Guan and Aixin Liu and Xin Xie and Yuxiang You and Kai Dong and Xingkai Yu and Haowei Zhang and Liang Zhao and Yisong Wang and Chong Ruan},
      year={2024},
      eprint={2412.10302},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.10302},
}
```

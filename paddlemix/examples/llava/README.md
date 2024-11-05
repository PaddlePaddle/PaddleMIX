# LLaVA

## 1. 模型介绍

[LLaVA](https://arxiv.org/pdf/2310.03744.pdf) 是基于大规模语言模型 llama 的视觉语言模型。支持多个多模态任务，包括零样本图像描述生成（Zero-shot Image Caption）、视觉问答（VQA）、细粒度视觉定位（Referring Expression Comprehension）等任务。

其性能优于其他模型，在多个任务上取得了更好的效果。

<p align="center">
  <img src="https://github.com/haotian-liu/LLaVA/blob/main/images/llava_v1_5_radar.jpg" align="middle" width = "600" />
</p>

注：图片引用自[LLaVA](https://github.com/haotian-liu/LLaVA).

本仓库提供paddle版本的Llava-v1.5-7b、Llava-v1.5-13b、Llava-v1.6-7b以及预训练所用的vicuna-13b-v1.5模型。


## 2 环境准备
- **python >= 3.8**
- **paddlenlp >= 2.7**

## 3 快速开始
完成环境准备后，我们提供多轮对话示例：

### 多轮对话启动
```bash
# llava
python paddlemix/examples/llava/run_predict_multiround.py \
--model-path "paddlemix/llava/llava-v1.5-7b" \
--image-file "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg" \
```
可配置参数说明：
  * `model-path`: 指定llava系列的模型名字或权重路径 ，支持 'paddlemix/llava/llava-v1.5-7b','paddlemix/llava/llava-v1.5-13b','paddlemix/llava/llava-v1.6-vicuna-7b'
  * `image-flie` :输入图片路径或url，默认None。



输入图片：<center><img src="https://github.com/LokeZhou/PaddleMIX/assets/13300429/95f73037-097e-4712-95be-17d5ca489f11" /></center>

```
USER: 描述这张照片
ASSISTANT: 这是一个照片，展示了一辆红色公交车在街道上行驶。车辆正在行驶在一个狭窄的道路上，周围有一些汽车和树木。车辆的前部有一个路灯，并且还有一个路灯在车辆的右侧。
USER: 给出公交车位置的坐标
ASSISTANT: 0.23, 0.33, 0.79, 0.78
```

## 4 预训练
我们提供`pretrain.py`脚本，用于预训练llava模型。

### 4.1 数据准备
将自己的数据放到一个列表中并存入json文件中，示例如下,或参考[llava_train_part_examples](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/llava/llava_train_examples.json)：
```json
[
    {
        "image": "http://ecx.images-amazon.com/images/I/51ntbts0gmL.jpg",
        "conversations": [
            [
                "<image>\nWhat is the genre of this book?",
                "Literature & Fiction"
            ]

        ]
    },
    {
        "image": "http://ecx.images-amazon.com/images/I/51cc3XrLevL.jpg",
        "conversations": [
            [
                "<image>\nWhat is the title of this book?",
                "Beyond Bigger Leaner Stronger: The Advanced Guide to Building Muscle, Staying Lean, and Getting Strong (The Build Muscle, Get Lean, and Stay Healthy Series)"
            ]
        ]
    },
    {
        "image": "http://ecx.images-amazon.com/images/I/517lfifp%2BqL.jpg",
        "conversations": [
            [
                "<image>\nIs this a romantic book?",
                "No"
            ]
        ]
    }
]

```
其中，"image"可以是本地的图片或网络地址；“conversations”是对话列表，每个对话包含两个元素，第一个为用户输入，第二个为系统回复，用户输入中的`<image>`表示输入图片，在预处理时会被替换为空。


### 4.2 预训练
预训练时使用`paddlemix/examples/llava/pretrain.py`程序进行训练，并使用`paddlemix/config/llava/pretrain.json`进行参数配置，**训练前请先检查数据集路径,如果使用url，请确保环境网络正常**。

预训练命令：
```bash
python paddlemix/examples/llava/pretrain.py paddlemix/config/llava/pretrain.json
```

## 5 模型微调
Llava 基于 PaddleMIX tool 统一微调工具链，支持全参数、lora微调，数据准备及参数配置等可参考 [tools](../../tools/README.md)

```bash
# llava lora微调
python paddlemix/tools/supervised_finetune.py paddlemix/config/llava/v1_5/lora_sft_argument.json

# llava full参数微调
python paddlemix/tools/supervised_finetune.py paddlemix/config/llava/v1_5/sft_argument.json
```

## 6 NPU硬件训练
请参照[tools](../../tools/README.md)进行NPU硬件Paddle安装和环境变量设置。执行预测和训练前需要设置如下环境变量：
```shell
export ASCEND_RT_VISIBLE_DEVICES=8
export FLAGS_npu_storage_format=0
export FLAGS_use_stride_kernel=0
export FLAGS_npu_jit_compile=0
export FLAGS_npu_scale_aclnn=True
export FLAGS_npu_split_aclnn=True
export FLAGS_allocator_strategy=auto_growth
export CUSTOM_DEVICE_BLACK_LIST=set_value,set_value_with_tensor
```

预测:
```shell
python paddlemix/examples/llava/run_predict_multiround.py \
    --model-path "paddlemix/llava/llava-v1.6-7b" \
    --image-file "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg" \
    --fp16
```
微调:
```shell
# llava lora微调
python paddlemix/tools/supervised_finetune.py paddlemix/config/llava/v1_5/lora_sft_argument.json

# llava full参数微调
python paddlemix/tools/supervised_finetune.py paddlemix/config/llava/v1_5/sft_argument.json
```


### 参考文献
```BibTeX
@misc{liu2024llavanext,
    title={LLaVA-NeXT: Improved reasoning, OCR, and world knowledge},
    url={https://llava-vl.github.io/blog/2024-01-30-llava-next/},
    author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Li, Bo and Zhang, Yuanhan and Shen, Sheng and Lee, Yong Jae},
    month={January},
    year={2024}
}

@misc{liu2023improvedllava,
      title={Improved Baselines with Visual Instruction Tuning},
      author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Lee, Yong Jae},
      publisher={arXiv:2310.03744},
      year={2023},
}

@misc{liu2023llava,
      title={Visual Instruction Tuning},
      author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
      publisher={NeurIPS},
      year={2023},
}
```

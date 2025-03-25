# Qwen-VL

## 1. 模型介绍

[Qwen-VL](https://arxiv.org/pdf/2308.12966.pdf) 是大规模视觉语言模型。可以以图像、文本、检测框作为输入，并以文本和检测框作为输出。Qwen-VL 系列模型的特点包括：

- **功能强大丰富**：支持多个多模态任务，包括零样本图像描述生成（Zero-shot Image Caption)、视觉问答（VQA）、细粒度视觉定位（Referring Expression Comprehension）等；
- **多语言对话模型**：支持英文、中文等多语言对话，端到端支持图片里中英双语的长文本识别；
- **多图多轮交错对话**：支持多图输入和比较，指定图片问答等；
- **细粒度识别和理解**：细粒度的文字识别、文档问答和检测框标注。

本仓库提供paddle版本的Qwen-VL-7b和Qwen-VL-Chat-7b模型。


## 2 环境准备
- **python >= 3.8**
- tiktoken
> 注：tiktoken 要求python >= 3.8
- paddlepaddle-gpu >= 2.5.1
- paddlenlp >= 2.6.1

> 注：请确保安装了以上依赖，否则无法运行。同时，需要安装 paddlemix/external_ops 下的自定义OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置PYTHONPATH

## 3 快速开始
完成环境准备后，我们提供三种使用方式：

## a. 单轮预测
```bash
# qwen-vl
python paddlemix/examples/qwen_vl/run_predict.py \
--model_name_or_path "qwen-vl/qwen-vl-7b" \
--input_image "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg" \
--prompt "Generate the caption in English with grounding:" \
--dtype "bfloat16"
```
可配置参数说明：
  * `model_name_or_path`: 指定qwen_vl系列的模型名字或权重路径，默认 qwen-vl/qwen-vl-7b
  * `seed` :指定随机种子，默认1234。
  * `visual:` :设置是否可视化结果，默认True。
  * `output_dir` :指定可视化图片保存路径。
  * `dtype` :设置数据类型，默认bfloat16,支持float32、bfloat16、float16。
  * `input_image` :输入图片路径或url，默认None。
  * `prompt` :输入prompt。

## b. 多轮对话
```bash
python paddlemix/examples/qwen_vl/chat_demo.py
```

## c. 通过[Appflow](../../../applications/README.md/)调用
> 注：使用Appflow前，需要完成Appflow环境配置，请参考[依赖安装](../../../applications/README.md/#1-appflow-依赖安装)。
```python

import paddle
from paddlemix.appflow import Appflow
paddle.seed(1234)
task = Appflow(app="image2text_generation",
                   models=["qwen-vl/qwen-vl-chat-7b"])
image= "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg"
prompt = "这是什么？"
result = task(image=image,prompt=prompt)

print(result["result"])

prompt2 = "框出图中公交车的位置"
result = task(prompt=prompt2)
print(result["result"])

```

输入图片：<center><img src="https://github.com/LokeZhou/PaddleMIX/assets/13300429/95f73037-097e-4712-95be-17d5ca489f11" /></center>

prompt：“这是什么？”

输出:
```
这是一张红色城市公交车的图片，它正在道路上行驶，穿越城市。该区域似乎是一个住宅区，因为可以在背景中看到一些房屋。除了公交车之外，还有其他车辆，包括一辆汽车和一辆卡车，共同构成了交通场景。此外，图片中还显示了一一个人，他站在路边，可能是在等待公交车或进行其他活动。
```
prompt2：“框出图中公交车的位置”

输出:
```
<ref>公交车</ref><box>(178,280),(803,894)</box>
```
<center><img src="https://github.com/LokeZhou/PaddleMIX/assets/13300429/2ff2ebcf-b7d7-48ed-af42-ead9d2befeb4" /></center>


## 4 模型微调
我们提供 `finetune.py` 脚本，用于 stage3 微调模型。
### 4.1 数据准备
将自己的数据放到一个列表中并存入json文件中，示例如下,或参考[sft_examples](https://bj.bcebos.com/v1/paddlenlp/models/community/qwen-vl/sft_examples.json)：
```json
[
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "user",
        "value": "你好"
      },
      {
        "from": "assistant",
        "value": "我是Qwen-VL,一个支持视觉输入的大模型。"
      }
    ]
  },
  {
    "id": "identity_1",
    "conversations": [
      {
        "from": "user",
        "value": "Picture 1: <img>https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg<img>\n图中的巴士是什么颜色的？"
      },
      {
        "from": "assistant",
        "value": "红色的。"
      },
      {
        "from": "user",
        "value": "框出图中的巴士的位置"
      },
      {
        "from": "assistant",
        "value": "<ref>巴士</ref><box>(178,279),(806,884)</box>"
      }
    ]
  },
  {
    "id": "identity_2",
    "conversations": [
      {
        "from": "user",
        "value": "Picture 1: <img>Chongqing.jpeg</img>\nPicture 2: <img>Beijing.jpeg</img>\n图中都是哪"
      },
      {
        "from": "assistant",
        "value": "第一张图片是重庆的城市天际线，第二张图片是北京的天际线。"
      }
    ]
  }
]
```

对于带图像输入的内容可表示为
`Picture id: <img>img_path</img>\n{your prompt}`，其中`id`表示对话中的第几张图片。"img_path"可以是本地的图片或网络地址。

对话中的检测框可以表示为`<box>(x1,y1),(x2,y2)</box>`，其中 `(x1, y1)` 和`(x2, y2)`分别对应左上角和右下角的坐标，并且被归一化到`[0, 1000)`的范围内. 检测框对应的文本描述也可以通过`<ref>text_caption</ref>`表示。

### 4.2 训练
训练时使用`paddlemix/examples/qwen_vl/finetune.py`程序进行训练，**训练前请先检查数据集路径,如果使用url，请确保环境网络正常**。推荐使用A100训练。

单卡训练
命令及参数配置示例：
```
export FLAGS_use_cuda_managed_memory=true #若显存不够，可设置环境变量，但会影响训练速度
MODEL_NAME="qwen-vl/qwen-vl-chat-7b"
DATA="train.json"

python paddlemix/examples/qwen_vl/finetune.py \
    --model_name_or_path ${MODEL_NAME} \
    --data_path ${DATA} \
    --dtype 'bfloat16' \
    --fix_vit True \
    --output_dir output_qwen_vl \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 1000 \
    --save_strategy "steps" \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 2048 \
    --lazy_preprocess True
```

多卡训练命令及参数配置示例：
```
MODEL_NAME="qwen-vl/qwen-vl-chat-7b"
MASTER='127.0.0.1:8080'
DATA="train.json"

python -m paddle.distributed.launch --master ${MASTER} --nnodes 1 --nproc_per_node 8 \
paddlemix/examples/qwen_vl/finetune.py \
    --model_name_or_path ${MODEL_NAME} \
    --data_path ${DATA} \
    --dtype 'bfloat16' \
    --fix_vit True \
    --output_dir output_qwen_vl \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 1000 \
    --save_strategy "steps" \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 2048 \
    --lazy_preprocess True
```


```
# 参数说明

--model_name_or_path #设置实际使用的模型，默认‘model_name_or_path’

--data_path #数据 json文件路径

--dtype    #数据类型，默认‘bfloat16’

--fix_vit #训练时是否固定visual vit的参数，默认True

--output_dir #模型存储路径

--num_train_epochs #训练epoch次数

--per_device_train_batch_size   #训练batch大小

--gradient_accumulation_steps   #在执行backward更新过程之前，用于累积梯度的更新步骤数。默认16，即执行16个step后，更新一次参数

--save_strategy   #训练期间要采用保存模型策略。可选择：
                  #“no”：在训练期间不进行任何保存。
                  #“epoch”`：每个epoch后保存。
                  #“steps”`：每“Save_steps”保存一次。

--save_steps  #每多少个steps保存一次模型

--save_total_limit  #最多保存多少个模型

--learning_rate  #学习率

--adam_beta2   #optimizer中beta2参数

--warmup_ratio  #学习率warm up比例

--weight_decay  #权重衰减

--lr_scheduler_type 1 #学习率衰减策略，可选cosine、linear

--logging_steps #日志打印间隔

--report_to  #日志集成，‘none’表示不集成，‘visualdl’表示集成到visualdl中

--model_max_length  #模型最大长度，默认2048

--lazy_preprocess #lazy 数据加载


```


### 参考文献
```BibTeX
@article{Qwen-VL,
  title={Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
  author={Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2308.12966},
  year={2023}
}
```

# Qwen-VL

## 1. 模型简介

该模型是 [Qwen-VL](https://arxiv.org/pdf/2308.12966.pdf) 的 paddle 实现。


## 2. Demo

## 2.1 环境准备
- **python >= 3.8**
- tiktoken
> 注：tiktoken 要求python >= 3.8
- paddlepaddle-gpu >= 2.5.1
- paddlenlp >= 2.6.1

> 注：请确保安装了以上依赖，否则无法运行。同时，需要安装[此目录](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/gpt-3/external_ops)下的自定义OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置PYTHONPATH

## 2.2 动态图推理
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


# qwen-vl-chat demo
```bash
python paddlemix/examples/qwen_vl/chat_demo.py
```

## 2.3 stage3 微调
我们提供 `finetune.py` 脚本，用于 stage3 微调模型。
### 2.3.1 数据准备
将自己的数据放到一个列表中并存入json文件中，示例如下：
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
        "value": "Picture 1: <img>https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg\n图中的巴士是什么颜色的？"
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

对于带图像输入的内容可表示为 `Picture id: <img>img_path</img>\n{your prompt}`，其中`id`表示对话中的第几张图片。"img_path"可以是本地的图片或网络地址。

对话中的检测框可以表示为`<box>(x1,y1),(x2,y2)</box>`，其中 `(x1, y1)` 和`(x2, y2)`分别对应左上角和右下角的坐标，并且被归一化到`[0, 1000)`的范围内. 检测框对应的文本描述也可以通过`<ref>text_caption</ref>`表示。

### 2.3.2 训练
训练时使用`paddlemix/examples/qwen_vl/finetune.py`程序进行训练，**训练前请先检查数据集路径,如果使用url，请确保环境网络正常**。

训练命令及参数配置示例：
```
MODEL_NAME="qwen-vl/qwen-vl-chat-7b"
MASTER='127.0.0.1:8080'
DATA="train.json"

python3.8 -m paddle.distributed.launch --master ${MASTER} --nnodes 1 --nproc_per_node 8 \
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
    --lazy_preprocess True \
    --sharding "stage2" \
    --tensor_parallel_degree 1 \
    --sharding_parallel_degree 8 \
    --pipeline_parallel_degree 1
```


```
# 参数说明

--model_name_or_path #设置实际使用的模型，默认‘model_name_or_path’

--data_path #数据 json文件路径

--bf16.    #是否使用bf16,默认True

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

--tensor_parallel_degree  # 模型并行系数，设置为N则进行N卡间模型并行。可选参数。

--sharding_parallel_degree  #显存优化策略，详情参考 [《ZeRO: Memory Optimizations Toward Training Trillion Parameter Models》]（https://arxiv.org/abs/1910.02054）可选参数。

--sharding  #显存优化策略stage选择，目前支持stage1、stage2。可选参数。

--pipeline_parallel_degree #流水线并行。详情参考[飞桨大语言模型工具链]（https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/README.md）可选参数。

```

> 注：若不需要 sharding 策略，则无需指定tensor_parallel_degree、sharding_parallel_degree、sharding、pipeline_parallel_degree参数

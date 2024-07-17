# InternLM-XComposer2

## 1. 模型介绍

[InternVL2](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/) InternVL 2.0，这是 InternVL 系列多模态大型语言模型的最新成员。InternVL 2.0 包含多种经过指令微调的模型，参数数量从 20 亿到 1080 亿不等。本仓库包含的是经过指令微调的 InternVL2-8B 模型。

与当前最先进的开源多模态大型语言模型相比，InternVL 2.0 超越了大多数开源模型。在多种能力方面，它表现出与专有商业模型相媲美的竞争力，包括文档和图表理解、信息图表问答、场景文本理解和 OCR 任务、科学和数学问题解决以及文化理解和综合多模态能力。

本仓库提供paddle版本的 InternVL2-8B 模型。


## 2 环境准备

1） [安装PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

2）[安装 PaddleMix 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/b4f97ff859e1964c839fc5fab94f7ba63b1e5959?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

## 3. 快速开始
完成环境准备后，我们目前提供单轮对话方式使用：


## a. 单轮预测
```bash
python paddlemix/examples/internvl2/chat_demo.py \
--model_name_or_path "OpenGVLab/InternVL2-8B"
--image_path "path/to/image.jpg"
--text "Please describe this image in detail."
```
可配置参数说明：
  * `model_name_or_path`: 指定 internvl2 的模型名字或权重路径以及tokenizer, processor 组件，默认 OpenGVLab/InternVL2-8B
  * `image_path`: 指定图片路径
  * `text`: 用户指令, 例如 "Please describe this image in detail."

## 4 模型微调
我们提供 `supervised_finetune.py` 脚本，用于模型微调。模型微调支持全参数微调，以及lora微调。
全参数微调需要A100 80G显存，lora微调支持V100 32G显存。

### 4.1 数据准备
将自己的数据放到一个列表中并存入json文件中，示例如下：
```json
[
    {
        "id": "identity_1651",
        "conversations": [
            [
                "<img>train/3304/image.png</img>\nWhich animal's feet are also adapted for sticking to smooth surfaces?\nmonitor lizard\nCosta Rica brook frog\n",
                "Costa Rica brook frog"
            ]
        ]
    },
    {
        "id": "identity_4090",
        "conversations": [
            [
                "<img>train/8416/image.png</img>\nWhat is the probability that a Labrador retriever produced by this cross will be homozygous dominant for the fur color gene?\n1/4\n0/4\n4/4\n2/4\n3/4\n",
                "0/4"
            ]
        ]
    },
    {
        "id": "identity_8662",
        "conversations": [
            [
                "<img>train/17895/image.png</img>\nWhich is this organism's scientific name?\nCastor canadensis\nNorth American beaver\n",
                "Castor canadensis"
            ]
        ]
    }
]
```

对于带图像输入的内容可表示为 `<img>img_path</img>\n{your prompt}`,"img_path"可以是本地的图片或url。
其中问题为`conversations`列表中的第一个元素，即`conversations[0]`；答案为第二个元素，即`conversations[1]`。

### 4.2 全参数训练

训练时使用统一sft脚本`paddlemix/tools/supervised_finetune.py`与配置文件`paddlemix/config/internvl2/sft_argument.json`进行训练。

训练命令：
```bash
paddlemix/tools/supervised_finetune.py paddlemix/config/internvl2/sft_argument.json
```

参数配置示例：
```json
{
    "model_name_or_path": "OpenGVLab/InternVL2-8B",
    "freeze_exclude": ["*vit*"],
    "dataset": {
        "train":[{"name": "chatml_dataset", "data_files": "path/to/train.json"}],
        "eval": [{"name": "chatml_dataset", "data_files": "path/to/eval.json"}]
    },
    "mixtoken": false,
    "output_dir": "./checkpoints/internvl_sft_ckpts",
    "overwrite_output_dir": true,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps":8,
    "eval_accumulation_steps":8,
    "num_train_epochs": 1,
    "learning_rate": 1e-05,
    "weight_decay": 0.1,
    "adam_beta2": 0.95,
    "warmup_ratio": 0.0,
    "lr_scheduler_type": "cosine",
    "logging_steps": 1,
    "save_steps": 100,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "max_length": 4096,
    "bf16": true,
    "fp16_opt_level": "O1",
    "do_train": true,
    "do_eval": false,
    "disable_tqdm": true,
    "load_best_model_at_end": false,
    "eval_with_do_generation": false,
    "skip_memory_metrics": false,
    "save_total_limit": 1
  }
```

参数说明 (大部分参数可从 [PaddleNLP 文档](https://paddlenlp.readthedocs.io/zh/latest/llm/finetune.html) 中查看相关含义)

```
--model_name_or_path # 设置实际使用的模型，默认 "internlm/internvl2"

--dataset # 数据集类型，以及路径

--per_device_train_batch_size  # 训练时每个设备的 batchsize

--per_device_eval_batch_size  # 验证时每个设备的 batchsize

--gradient_accumulation_steps  # 设置每隔若干个 step 更新一次梯度

--eval_accumulation_steps  # 验证时每隔若干个 step 将验证过程中得到的inputs, labels, logits, loss 放到cpu上

--num_train_epochs  # 训练的 epoch 数

--learning_rate  # 学习率

--weight_decay  # 优化器递减率

--adam_beta2  # adam 优化器相关参数

--warmup_ratio  # warm-up 率

--lr_scheduler_type  # 学习率调度类型

--logging_steps  # 日志配置

--save_steps  # 每多少个steps保存一次模型

--evaluation_strategy   # 评估策略。可选择：
                        # “no”：在训练期间不进行任何评估。
                        # “epoch”`：每个epoch后评估。
                        # “steps”`：每“eval_steps”评估一次。

--save_strategy   # 训练期间要采用保存模型策略。可选择：
                  # “no”：在训练期间不进行任何保存。
                  # “epoch”`：每个epoch后保存。
                  # “steps”`：每“Save_steps”保存一次。

--max_length  # 模型最大长度

--bf16  # 是否选用 bf16 精度

--fp16_opt_level  # fp16 混合精度配置

--do_train  # 是否进行训练

--do_eval  # 是否进行验证

--disable_tqdm  # 是否使用 tqdm

--load_best_model_at_end  # 是否在训练结束时加载最优模型

--eval_with_do_generation  # 在模型效果评估的时候是否调用model.generate,默认为False

--skip_memory_metrics  # 是否跳过将内存探查器报告添加到度量中。默认情况下会跳过此操作，因为它会降低训练和评估速度

--save_total_limit  # 保留checkpoint的个数，老的checkpoint会被删除
```





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
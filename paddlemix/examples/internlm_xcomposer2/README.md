# InternLM-XComposer2

## 1. 模型介绍

[InternLM-XComposer2](https://arxiv.org/abs/2401.16420) 是基于 InternLM2-7B 大语言模型研发的突破性的图文多模态大模型，具有非凡的图文写作和图像理解能力，在多种应用场景表现出色：

+ 自由指令输入的图文写作： InternLM-XComposer2 可以理解自由形式的图文指令输入，包括大纲、文章细节要求、参考图片等，为用户打造图文并貌的专属文章。生成的文章文采斐然，图文相得益彰，提供沉浸式的阅读体验。

+ 准确的图文问题解答： InternLM-XComposer2 具有海量图文知识，可以准确的回复各种图文问答难题，在识别、感知、细节描述、视觉推理等能力上表现惊人。

+ 杰出性能： InternLM-XComposer2 在13项多模态评测中大幅领先同量级多模态模型，在其中6项评测中超过 GPT-4V 和 Gemini Pro。

本仓库提供paddle版本的 InternLM-XComposer2-7b 模型。


## 2 环境准备

1） [安装PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

2）[安装 PaddleMix 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/b4f97ff859e1964c839fc5fab94f7ba63b1e5959?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

## 3. 快速开始
完成环境准备后，我们目前提供单轮对话方式使用：


## a. 单轮预测
```bash
python paddlemix/examples/internlm_xcomposer2/chat_demo.py \
--from_pretrained "internlm/internlm-xcomposer2-7b"
```
可配置参数说明：
  * `from_pretrained`: 指定 internlm_xcomposer2 的模型名字或权重路径以及tokenizer, processor 组件，默认 internlm/internlm-xcomposer2-7b


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

对于带图像输入的内容可表示为 `<img>img_path</img>\n{your prompt}`,"img_path"可以是本地的图片或网络地址。
其中问题为`conversations`列表中的第一个元素，即`conversations[0]`；答案为第二个元素，即`conversations[1]`。

### 4.2 全参数训练

训练时使用统一sft脚本`paddlemix/tools/supervised_finetune.py`与配置文件`paddlemix/config/internlm_xcomposer2/sft_argument.json`进行训练。

训练命令：
```bash
paddlemix/tools/supervised_finetune.py paddlemix/config/internlm_xcomposer2/sft_argument.json
```

参数配置示例：
```json
{
    "model_name_or_path": "internlm/internlm-xcomposer2-7b",
    "dataset": {
        "train":[{"name": "chatml_dataset", "data_files": "path/to/train.json"}],
        "eval": [{"name": "chatml_dataset", "data_files": "path/to/eval.json"}]
    },
    "mixtoken": false,
    "output_dir": "./checkpoints/internlm_xcomposer_sft_ckpts",
    "overwrite_output_dir": true,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps":8,
    "eval_accumulation_steps":8,
    "num_train_epochs": 1,
    "learning_rate": 1e-05,
    "weight_decay": 0.1,
    "adam_beta2": 0.95,
    "warmup_ratio": 0.01,
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
--model_name_or_path # 设置实际使用的模型，默认 "internlm/internlm-xcomposer2-7b"

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
@article{internlmxcomposer2,
      title={InternLM-XComposer2: Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Model},
      author={Xiaoyi Dong and Pan Zhang and Yuhang Zang and Yuhang Cao and Bin Wang and Linke Ouyang and Xilin Wei and Songyang Zhang and Haodong Duan and Maosong Cao and Wenwei Zhang and Yining Li and Hang Yan and Yang Gao and Xinyue Zhang and Wei Li and Jingwen Li and Kai Chen and Conghui He and Xingcheng Zhang and Yu Qiao and Dahua Lin and Jiaqi Wang},
      journal={arXiv preprint arXiv:2401.16420},
      year={2024}
}
```

## 📦 PaddleMIX Tools Introduction📦
The PaddleMIX toolkit embodies the design philosophy of one-stop experience, ultimate performance, and ecosystem compatibility upheld by the PaddlePaddle suite. It aims to provide developers with a unified set of tools for the entire process of cross-modal large-model development, aligning with industry standards. This facilitates low-cost, low-threshold, and rapid customization of cross-modal large models.

[[中文文档](README.md)]

##  🛠️ Supported Model List 🛠️
| Model | Inference |Pretrain | SFT | LoRA | Deploy |
| --- | --- | --- | --- | --- | --- |
| [qwen_vl](../examples/qwen_vl/) | ✅  | ❌  | ✅  | ✅  |  ✅ |
| [blip2](../examples/blip2/) | ✅  | ✅ | ✅  | ✅ | ✅  |
| [visualglm](../examples/visualglm/) | ✅ | ❌ | ✅ | ✅ | ❌ |
| [llava](../examples/llava/) | ✅  | ✅   | ✅  | ✅  | 🚧  |

* ✅: Supported
* 🚧: In Progress
* ❌: Not Supported

Note:
1. 开始前请先按照[环境依赖](../../README_EN.md#environment-dependencies)安装环境，不同模型请参考 [examples](../examples/README.md) 下对应的模型目录安装依赖；
2. 当前**tools**统一接口只支持部分模型的精调能力，其他模型及其他能力后续陆续上线。


##  🚀 Quick Start 🚀

### 1. Fine-tuning
PaddleMIX fine-tuning supports various mainstream large multi-modal fine-tuning strategies such as SFT and LoRA, providing a unified and efficient fine-tuning solution:
- **Unified Training Entry**: The PaddleMIX fine-tuning solution is adaptable to various mainstream large multi-modal models. Users only need to modify the configuration file in [config](../config/) to perform fine-tuning on single or multiple GPUs for different large models.
- **Multiple and Mixed Datasets**: Supports fine-tuning with multiple datasets and mixed datasets simultaneously, including a mixture of datasets such as VQA, Caption, Chatml, etc.
- **Powerful Data Flow and Distributed Strategies**: The MIXToken strategy effectively increases data throughput, significantly improving model training efficiency. Adaptive Trainer and customizable Trainer configurations seamlessly integrate with Paddle's distributed parallelism strategies, greatly reducing the hardware threshold for fine-tuning large models.

**Data Preparation**：

We support the use of multiple datasets and mixed datasets simultaneously for fine-tuning. This is achieved through **MixDataset** scheduling. Users only need to specify in the configuration file a collection of datasets supported by [dataset](../datasets/). For example:
```
# config.json
...

"dataset": {
      "train":[
        {"name": "chatml_dataset", "data_files": "train.json"，"chat_template":"chat_template.json"},
        {"name": "coco_caption", "data_files": "train.json"},
        ......],
      "eval": [
        {"name": "chatml_dataset", "data_files": "val.json"，"chat_template":"chat_template.json"},
        {"name": "coco_caption", "data_files": "val.json"},
        ......],
    },
....

```

For each sub-dataset, such as the coco_caption dataset mentioned above, you can refer to the documentation inside the corresponding model directory under [examples](../examples/).

Additionally, we support mainstream LLM dialogue custom templates through the chatml_dataset dataset. For information on customizing dialogue templates, please refer to [Custom Dialogue Templates](https://github.com/PaddlePaddle/PaddleNLP/blob/16d3c49d2b8d0c7e56d1be8d7f6f2ca20aac80cb/docs/get_started/chat_template.md#自定义对话模板).

For convenience in testing, we also provide a dataset in the chatml_dataset format and the corresponding chat_template.json file for fine-tuning the qwen-vl model. You can directly [download](https://bj.bcebos.com/v1/paddlenlp/datasets/examples/ScienceQA.tar) and use it.

**config** Configuration File Parameter Explanation：
```
{
    “model_name_or_path”  #设置实际使用的模型名称，如"qwen-vl/qwen-vl-chat-7b"

    "freeze_include"  #设置需要冻结的层，如["*visual*"]，默认None

    "freeze_exclude"  #设置不需要冻结的层，如["*visual.attn_pool*"]，默认None

    "dataset": {
        "train":[{"name": "chatml_dataset", "data_files": "train.json"}],
        "eval": [{"name": "chatml_dataset", "data_files": "val.json"}]
    },  #数据集配置

    “mixtoken” : #是否使用mixtoken策略，默认False,

    "output_dir":  #模型存储路径

    "overwrite_output_dir": # 覆盖输出目录，默认False

    "per_device_train_batch_size":  #训练batch大小

    “gradient_accumulation_steps”: #在执行backward更新过程之前，用于累积梯度的更新步骤数。如设置为16，即执行16个step后，更新一次参数

    "per_device_eval_batch_size"  #评估batch大小

    "eval_accumulation_steps" : 评估累积步数

    “save_strategy”:  #训练期间要采用保存模型策略。可选择：
                    #“no”：在训练期间不进行任何保存。
                    #“epoch”`：每个epoch后保存。
                    #“steps”`：每“Save_steps”保存一次。

    "save_steps":  #每多少个steps保存一次模型

    "save_total_limit":  #最多保存多少个模型

    "evaluation_strategy":   #评估策略。可选择：
                            #“no”：在训练期间不进行任何评估。
                            #“epoch”`：每个epoch后评估。
                            #“steps”`：每“eval_steps”评估一次。

    "do_train":  #是否进行训练

    "bf16": #是否使用bf16训练，默认False，仅支持a100

    "fp16": #是使用fp16训练，默认False

    “fp16_opt_level”: #混合精度训练等级，可选O1,O2

    "learning_rate":  #学习率

    "adam_beta2":   #optimizer中beta2参数

    "warmup_ratio":  #学习率warm up比例

    "weight_decay":  #权重衰减

    "lr_scheduler_type":  #学习率衰减策略，可选cosine、linear

    "logging_steps": #日志打印间隔

    "max_length":  #模型最大长度，默认2048

    "benchmark":  #是否开启benchmark模式，默认False

    "skip_memory_metrics":  #是否跳过内存指标，默认False

    “lora”: #是否使用LoRA策略，默认False

    "lora_rank": #LoRA rank

    "lora_alpha": #LoRA alpha

    "lora_dropout": #LoRA dropout

    "lora_target_modules": #LoRA target modules,如
    [ ".*attn.c_attn.*",
        ".*attn.c_proj.*",
        ".*mlp.w1.*",
        ".*mlp.w2.*"]

    "tensor_parallel_degree":  # 模型并行系数，设置为N则进行N卡间模型并行。可选参数。

    "sharding_parallel_degree":  #显存优化策略，详情参考 [《ZeRO: Memory Optimizations Toward Training Trillion Parameter Models》]（https://arxiv.org/abs/1910.02054）可选参数。

    "sharding":  #显存优化策略stage选择，目前支持stage1、stage2。可选参数。

    "pipeline_parallel_degree": #流水线并行。详情参考[飞桨大语言模型工具链]（https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/README.md）可选参数。

}

```

> Note: If you do not need the sharding strategy, you do not need to specify the tensor_parallel_degree, sharding_parallel_degree, sharding, or pipeline_parallel_degree parameters.

> For more parameters, please refer to [argument](../trainer/argument.py).

**Full-parameter Fine-tuning: SFT**
```bash
# 单卡Qwen-vl SFT启动命令参考
python paddlemix/tools/supervised_finetune.py paddlemix/config/qwen_vl/sft_argument.json

# 多卡Qwen-vl SFT启动命令参考
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" paddlemix/tools/supervised_finetune.py paddlemix/config/qwen_vl/sft_argument.json
```

**LoRA**
```bash
# 单卡Qwen-vl LoRA启动命令参考
python  paddlemix/tools/supervised_finetune.py paddlemix/config/qwen_vl/lora_sft_argument.json
```

Note: After training with LoRA, it's necessary to merge the LoRA parameters. We provide a script for merging LoRA parameters, which combines the LoRA parameters into the main model and saves the corresponding weights. The command is as follows:

```bash
python paddlemix/paddlemix/tools/merge_lora_params.py \
--model_name_or_path qwen-vl/qwen-vl-chat-7b \
--lora_path output_qwen_vl\
--merge_model_path qwen_vl_merge
```

## 📦 PaddleMIX工具箱介绍 📦
PaddleMIX工具箱秉承了飞桨套件一站式体验、性能极致、生态兼容的设计理念，旨在提供业界主流跨模态大模型全流程统一工具，帮助开发者低成本、低门槛、快速实现跨模态大模型定制化。

[[English](README_en.md)]

##  🛠️ Unified Fine-tuning Tool for Multimodal Understanding 🛠️

| Model |  SFT | LoRA | Deploy | NPU training |
| --- |  --- | --- | --- | --- | 
| [YOLO-World](./YOLO-World/) | ❌  | ❌  | ❌ | ❌ |
| [audioldm2](./audioldm2/) | ❌ | ❌ | ❌ | ❌ |
| [blip2](./blip2/) | ✅  | ✅ |  ❌ | ❌ |
| [clip](./clip) |❌ | ❌ | ❌ | ❌ |
| [coca](./coca/) |  ❌ | ❌ | ❌ | ❌ |
| [CogVLM && CogAgent](./cogvlm/) |❌ | ❌ | ❌ | ❌ |
| [eva02](./eva02/)|   ✅  |  ❌   | ❌   | ❌ |
| [evaclip](./evaclip/) | ❌ | ❌ |  ❌ | ❌ |
| [groundingdino](./groundingdino/) |  🚧   | ❌  | ✅  | ❌ |
| [imagebind](./imagebind/) |  ❌  | ❌ | ❌ | ❌ |
| [InternLM-XComposer2](./internlm_xcomposer2/) | ✅ | ❌ | ❌ | ❌ |
| [Internvl2](./internvl2/)| ✅ | ❌ | ❌ | ✅ |
| [llava](./llava/)  | ✅  | ✅  | 🚧  | ✅ |
| [llava-next](./llava_next_interleave/) | ❌ | ❌ | ❌ | ❌ |
| [minigpt4](./minigpt4) | ✅   |  ❌  | ✅  | ❌ |
| [minimonkey](./minimonkey/) | ✅ | ❌ | ❌ | ❌ |
| [qwen2_vl](./qwen2_vl/)| ✅ | ❌ | ❌ | ❌ |
| [qwen_vl](./qwen_vl/)  | ✅  | ✅  | ✅  | ❌ |
| [sam](./sam/) | ❌ | ❌ | ✅  | ❌ |
| [visualglm](./visualglm/) | ✅ | ✅ | ❌ | ❌ |

* ✅: Supported
* 🚧: In Progress
* ❌: Not Supported

注意：
1. 开始前请先按照[环境依赖](../../README.md#安装)安装环境，不同模型请参考 [examples](../examples/README.md) 下对应的模型目录安装依赖；
2. 当前**tools**统一接口只支持部分模型的精调能力，其他模型及其他能力后续陆续上线。


##  🚀 快速开始 🚀

### 1. 精调
PaddleMIX 精调支持多个主流跨模态大模型的SFT、LoRA等精调策略，提供统一、高效精调方案：
- **统一训练入口** PaddleMIX 精调方案可适配业界主流跨模态大模型，用户只需修改[config](../config/) 配置文件，即能在单卡或多卡进行多种大模型精调；
- **多数据集和混合数据集** 支持多种数据集和混合数据集同时精调，包括：VQA、Caption、Chatml等数据集混合使用；
- **强大数据流和分布式策略** MIXToken策略有效增加数据吞吐量，大幅度提高模型训练效率。自适应Trainer和定制化Trainer灵活配置，无缝链接飞桨分布式并行策略，大幅降低大模型精调硬件门槛。


**数据准备**：

我们支持多数据集、混合数据集同时用于精调，通过**MixDataset**统一调度，用户只需在配置文件指定 [dataset](../datasets/) 支持的数据集组成集合，即可使用。如：

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

而对于每个子数据集，如上述的 coco_caption 数据格式，可参考[examples](../examples/) 下对应的模型目录里面的文档介绍。

同时，我们通过 chatml_dataset 数据集支持主流的LLM对话自定义模版。关于自定义对话模板，请参考[自定义对话模版](https://github.com/PaddlePaddle/PaddleNLP/blob/16d3c49d2b8d0c7e56d1be8d7f6f2ca20aac80cb/docs/get_started/chat_template.md#自定义对话模板
)

为了方便测试，我们也提供了 chatml_dataset 格式的数据集和对应的 chat_template.json,用于 qwen-vl 模型精调，可以直接[下载](https://bj.bcebos.com/v1/paddlenlp/datasets/examples/ScienceQA.tar)使用。

**config** 配置文件参数说明：
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

    "device": #训练硬件，npu、gpu

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

> 注：若不需要 sharding 策略，则无需指定tensor_parallel_degree、sharding_parallel_degree、sharding、pipeline_parallel_degree参数

> 更多参数，可参考 [argument](../trainer/argument.py)

**全参精调：SFT**
```bash
# 单卡Qwen-vl SFT启动命令参考
export FLAGS_use_cuda_managed_memory=true #若显存不够，可设置环境变量
python paddlemix/tools/supervised_finetune.py paddlemix/config/qwen_vl/sft_argument.json

# 多卡Qwen-vl SFT启动命令参考
export FLAGS_use_cuda_managed_memory=true #若显存不够，可设置环境变量
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" paddlemix/tools/supervised_finetune.py paddlemix/config/qwen_vl/sft_argument.json

# 或者使用统一启动脚本
sh paddlemix/tools/train.sh paddlemix/config/qwen_vl/sft_argument.json
```

**LoRA**
```bash
# 单卡Qwen-vl LoRA启动命令参考
python  paddlemix/tools/supervised_finetune.py paddlemix/config/qwen_vl/lora_sft_argument.json

# 多卡Qwen-vl LoRA启动命令参考
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" paddlemix/tools/supervised_finetune.py paddlemix/config/qwen_vl/lora_sft_argument.json

```

注：使用lora训练后，需要合并lora参数，我们提供LoRA参数合并脚本，可以将LoRA参数合并到主干模型并保存相应的权重。命令如下：

```bash
python paddlemix/tools/merge_lora_params.py \
--model_name_or_path qwen-vl/qwen-vl-chat-7b \
--lora_path output_qwen_vl\
--merge_model_path qwen_vl_merge
```

**NPU硬件训练**

PaddleMIX支持在NPU硬件上进行训练：
1. 请先参照[PaddleCustomDevice](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/README_cn.md)安装NPU硬件Paddle
2. 在config配置文件中增加`device`字段指定设备：
```json
{
    ...
    "model_name_or_path": "paddlemix/llava/llava-v1.5-7b",
    "device": "npu",
    "output_dir": "./checkpoints/llava_sft_ckpts",
    ...
}
```
3. 启动训练前请设置如下环境变量用于性能加速和精度对齐
```shell
export FLAGS_use_stride_kernel=0
export FLAGS_npu_storage_format=0 # 关闭私有格式
export FLAGS_npu_jit_compile=0 # 关闭即时编译
export FLAGS_npu_scale_aclnn=True # aclnn加速
export FLAGS_npu_split_aclnn=True # aclnn加速
export CUSTOM_DEVICE_BLACK_LIST=set_value,set_value_with_tensor # set_value加入黑名单
```
目前支持NPU训练的模型可以参考此[文档](../examples/README.md)
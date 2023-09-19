# BLIP-2

## 1. 模型简介

[BLIP-2](https://arxiv.org/abs/2301.12597): Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models, Paddle实现版本.

BLIP-2：使用冻结图像编码器和大型语言模型的语言图像预训练，在VQA,Caption等多图文任务上表现出性能优势。

<p align="center">
  <img src="https://github.com/salesforce/LAVIS/blob/main/projects/blip2/blip2_illustration.png" align="middle" width = "600" />
</p>

注：图片引用自[BLIP-2](https://github.com/salesforce/LAVIS/blob/main/projects/blip2).

### BLIP-2 Series

> visual encoder: ``eva_vit_g`.

| model name | weight |
|:-----|:------:|
| `blip2-stage1` | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/blip2-stage1/model_state.pdparams) |
| `blip2-stage2` | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/blip2-stage2/model_state.pdparams) |
| `blip2-pretrained-opt2.7b` | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/blip2-pretrained-opt2.7b/model_state.pdparams) |
| `blip2-pretrained-opt6.7b` | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/blip2-pretrained-opt2.7b/model_state.pdparams) |
| `blip2_pretrained_flant5xl` | To be released |
| `blip2_pretrained_flant5xxl` | To be released|
| `blip2-caption-opt2.7b` | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/blip2-pretrained-opt2.7b/model_state.pdparams) |
| `blip2-caption-opt6.7b` | [weight](https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/blip2-pretrained-opt6.7b/model_state.pdparams) |
| `blip2_caption_flant5xl` | To be released |


  * `blip2-stage1` :blip2第一阶段预训练模型,可用于开启第二阶段预训练
  * `blip2-stage2` :使用论文中数据训练好的第一阶段模型,可用于开启第二阶段预训练
  * `blip2-pretrained-opt2.7b` :对应论文精度Blip2第二阶段训练完成的模型,语言模型使用`opt-2.7b`,可用于模型微调任务或进行`zeroshot vqa`推理
  * `blip2-pretrained-opt6.7b` :对应论文精度Blip2第二阶段训练完成的模型,语言模型使用`opt-6.7b`,可用于模型微调任务或进行`zeroshot vqa`推理
  * `blip2_pretrained_flant5xl` :对应论文精度Blip2第二阶段训练完成的模型,语言模型使用`flant5-xl`,可用于模型微调任务或进行`zeroshot vqa`推理
  * `blip2_pretrained_flant5xxl` :对应论文精度Blip2第二阶段训练完成的模型,语言模型使用`flant5-xxl`,可用于模型微调任务或进行`zeroshot vqa`推理
  * `blip2-caption-opt2.7b` :对应论文精度Blip2第二阶段训练完成并在caption数据集进行微调的模型,语言模型使用`opt-2.7b`,可用于`image caption`推理
  * `blip2-caption-opt6.7b` :对应论文精度Blip2第二阶段训练完成并在caption数据集进行微调的模型,语言模型使用`opt-6.7b`,可用于`image caption`推理
  * `blip2_caption_flant5xl` :对应论文精度Blip2第二阶段训练完成并在caption数据集进行微调的模型,语言模型使用`flant5-xl`,可用于`image caption`推理

</div>

## 2. 环境准备
  ```
  cd PaddleMIX
  pip install -r requirements.txt
  ```

## 3. 数据准备


1. coco数据

  >数据部分，默认使用`coco_karpathy`数据，使用该数据不需另外配置，会自动下载。 目前已支持 "coco_caption","vg_caption"等数据集训练

2. 自定义数据

  >如果需要自定义数据，推荐沿用`coco_karpathy`数据格式处理自己的数据。其中每条数据标注格式示例为:
  ```
  {'caption': 'A woman wearing a net on her head cutting a cake. ', 'image': 'val2014/COCO_val2014_000000522418.jpg', 'image_id': 'coco_522418'}
  ```
  >更多可参考数据集中的`annotations/coco_karpathy_train.json`文件。

  >在准备好自定义数据集以后, 我们可以使用 ``load_dataset()`` 来加载数据.
  ```python
  from lavis.datasets.builders import load_dataset
  coco_dataset = load_dataset("coco_caption", data_files=[[TRAIN_IMAGE_LOCAL_PATH,TRAIN_ANN_LOCAL_PATH,MODE]])
  '''
  for example:
  lcoco_dataset = oad_dataset("coco_caption", data_files=[['/root/.paddlemix/datasets/coco/images/','/root/.paddlemix/datasets/coco/annotations/coco_karpathy_train.json',"train"]])[0]
  print(coco_dataset[0])
  '''
  # {'image':
  # '/root/.paddlemix/datasets/coco/images/val2014/COCO_val2014_000000522418.jpg',
  # 'image_id': 0,
  # 'text_input':'A woman wearing a net on her head cutting a cake. '}
  ```

## 4. 使用说明

我们在Paddle中实现了`BLIP-2`系列模型，目前包括`BLIP-2-OPT`、`BLIP-2-FlanT5`

### 4.1 训练

无需更改参数即可开始训练BLIP-2
如需调整参数请见以下参数配置示例：
```python
MODEL_NAME="paddlemix/blip2-stage2"
fleetrun --master '127.0.0.1' --nnodes 1 --nproc_per_node 8 --ips '127.0.0.1:8080' run_pretrain_stage2.py \
    --per_device_train_batch_size 128 \
    --model_name_or_path ${MODEL_NAME}  \
    --warmup_steps 2000 \
    --eta_min 1e-5 \
    --learning_rate 0.0001 \
    --weight_decay 0.05 \
    --num_train_epochs 10 \
    --tensor_parallel_degree 1 \
    --sharding_parallel_degree 1 \
    --output_dir "./output" \
    --logging_steps 50 \
    --do_train \
    --save_strategy epoch \
#MODEL_NAME 路径配置等价于 paddlemix/ + 已支持的`model name` 例如 `blip2-pretrained-opt2.7b`,`paddlemix/blip2-stage1`,`paddlemix/blip2-stage1` 等
```
  可配置参数说明(具体请参考`paddlemix/examples/blip2/run_pretrain_stage2.py`注释说明)：
  * `model_name_or_path`: 指定blip2模型的config和权重路径。
  * `text_model_name_or_path` :指定blip2 语言模型部分的tokenizer类型,通常与语言模型路径同名例如`facebook/opt-2.7b`。
  * `gradient_checkpointing` :指定是否开启recompute以节省运行显存。
  * `tensor_parallel_degree:` :设置张量模型的并行数。
  * `sharding_parallel_degree` :设置分片数量，启用分片并行。
  * `sharding` :设置分片并行类型。
  * `resume_from_checkpoint` :恢复训练中断的模型有效检查点的文件夹路径。
  * `load_model_path` :从指定路径加载权重的权重路径
  model_name_or_path目前支持,模型下载后默认保存在本地路径 `/root/.paddlemix`:
  * blip2-stage1预训练模型: `paddlemix/blip2-stage1`
  * blip2-stage2预训练模型: `paddlemix/blip2-stage2`
  * blip2-vqa模型/微调预训练模型: `paddlemix/blip2-pretrained-opt2.7b`, `paddlemix/blip2-pretrained-opt6.7b`, `paddlemix/blip2_pretrained_flant5xl`, `paddlemix/blip2_pretrained_flant5xxl`
  * blip2-caption模型: `paddlemix/blip2-caption-opt2.7b`, `paddlemix/blip2-caption-opt6.7b`, `paddlemix/blip2_caption_flant5xl.7b`


#### stage1
```
# 单卡训练
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/blip2/run_pretrain_stage1.py
# 多卡训练
fleetrun --gpus=0,1,2,3 paddlemix/examples/blip2/run_pretrain_stage1.py
```
#### stage2
```
# 单卡训练
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/blip2/run_pretrain_stage2.py
# 多卡训练
fleetrun --gpus=0,1,2,3 paddlemix/examples/blip2/run_pretrain_stage2.py
```
### 4.2 评估

#### task_vqa
```
fleetrun --gpus=0,1,2,3 paddlemix/examples/blip2/run_eval_vqav2_zeroshot.py
```
#### task_caption
```
fleetrun --gpus=0,1,2,3 paddlemix/examples/blip2/run_eval_caption.py
```

### 4.3 预测
```
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/blip2/run_predict.py
```

### 4.4  resume 以stage2为例
  ```
  # 单卡训练
  CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/blip2/run_pretrain_stage2.py --resume_from_checkpoint outdir/checkpoint-1
  # 多卡训练
  fleetrun --gpus=0,1,2,3 paddlemix/examples/blip2/run_pretrain_stage2.py  --resume_from_checkpoint outdir/checkpoint-1
  ```
### 4.4  指定加载本地模型权重 以stage2为例
  ```
  # 单卡训练
  CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/blip2/run_pretrain_stage2.py --load_model_path model_dir_path/
  # 多卡训练
  fleetrun --gpus=0,1,2,3 paddlemix/examples/blip2/run_pretrain_stage2.py  --load_model_path model_dir_path/
  ```
### 4.5 配置文件说明
  1. 以blip2-stage2为例,运行:
  ```python
  fleetrun --gpus=0,1,2,3 paddlemix/examples/blip2/run_pretrain_stage2.py
  ```
  2. 开启stage2训练 会自行下载相关配置文件和权重至/root/.paddlenlp/models/paddlemix/blip2-stage2
  3. 打开目录下的config.json 如下:
  ```python
  {
    "architectures": [
      "Blip2ForConditionalGeneration"
    ],
    "vision_name_or_path":"paddlemix/blip2-stage2/eva_vit_g",
    "bridge_name_or_path":"paddlemix/blip2-stage2/Qformer",
    "vision_and_bridge_name_or_path":"paddlemix/blip2-caption-opt2.7b",
    "text_config": "facebook/opt-2.7b",
    "freeze_vit": true,
    "initializer_factor": 1.0,
    "initializer_range": 0.02,
    "use_decoder_only_language_model": true,
    "model_type": "blip-2",
    "paddlenlp_version": null,
    "qformer_config": {
      "add_cross_attention": true,
      "attention_probs_dropout_prob": 0.1,
      "cross_attention_freq": 2,
      "embed_dim": 256,
      "fuse": false,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "layer_norm_eps": 1e-12,
      "max_position_embeddings": 512,
      "model_type": "bert",
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "num_query_tokens": 32,
      "pad_token_id": 0,
      "paddlenlp_version": null,
      "pool_act": "tanh",
      "type_vocab_size": 2,
      "vocab_size": 30522,
      "dropout":  null

    },
    "vision_config": {
      "depth": 39,
      "drop_rate": 0,
      "embed_dim": 1408,
      "epsilon": 1e-06,
      "gradient_checkpointing": false,
      "img_size": 224,
      "mlp_ratio": 4.3637,
      "model_type": "blip_2_vision_model",
      "num_heads": 16,
      "paddlenlp_version": null,
      "patch_size": 14,
      "qkv_bias": true,
      "return_dict": true
    }
  }
  ```
  可配置参数说明：
  * `vision_name_or_path`: 指定visual encoder的模型路径,默认已经提供
  * `bridge_name_or_path` : 指定Qformer的模型路径,默认已经提供
  * `vision_and_bridge_name_or_path` : 指定visual encoder和Qformer拼接好后的模型路径，如果已经指定该路径，可不配置`vision_name_or_path`,`bridge_name_or_path`
  * `freeze_vit` :设置是否冻结visual encoder参数。
  * `qformer_config` :指定Qformer的config配置。
  * `vision_config` :指定visual encoder的config配置。
  * `text_config` :指定语言模型的加载路径通常从paddlenlp中直接加在权重和配置文件,只需给出语言模型在paddlenlp中的路径即可。

  paddlemix 支持用户自行拼接visual encoder和Qformer 运行命令如下：
  ```python
  python paddlemix/examples/blip2/merge_weight.py --vision_name_or_path --bridge_name_or_path --save_path
  ```

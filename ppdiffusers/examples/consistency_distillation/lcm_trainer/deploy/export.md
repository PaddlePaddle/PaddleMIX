# Latent Consistency Models 导出教程


[PPDiffusers](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers) 是一款支持跨模态（如图像与语音）训练和推理的扩散模型（Diffusion Model）工具箱，其借鉴了🤗 Huggingface 团队的 [Diffusers](https://github.com/huggingface/diffusers) 的优秀设计，并且依托 [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) 框架和 [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) 自然语言处理库。下面将介绍如何将 PPDiffusers 提供的预训练模型进行模型导出。

### 模型导出
可执行以下命令行完成模型导出。

#### 导出带LCM-LoRA的权重
```shell
# 关闭ppxformers，否则会导致模型导出失败
export USE_PPXFORMERS=False
python export_model.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --output_path lcm-stable-diffusion-v1-5 --lcm_lora_path ./outputs/checkpoint-10000/lora/lcm_lora.safetensors
```

#### 导出LCM的权重
```shell
# 关闭ppxformers，否则会导致模型导出失败
export USE_PPXFORMERS=False
python export_model.py --pretrained_model_name_or_path LCM-MODEL-NAME-OR-PATH --output_path lcm-stable-diffusion-v1-5 --unet_path ./outputs/checkpoint-10000/unet
```

注: 上述指令没有导出固定尺寸的模型，固定尺寸的导出模型有利于优化模型推理性能，但会牺牲一定灵活性。若要导出固定尺寸的模型，可指定`--height`和`--width`参数。

输出的模型目录结构如下：

```shell
lcm-stable-diffusion-v1-5/
├── model_index.json
├── scheduler
│   └── scheduler_config.json
├── tokenizer
│   ├── tokenizer_config.json
│   ├── merges.txt
│   ├── vocab.json
│   └── special_tokens_map.json
├── text_encoder
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
├── unet
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
├── vae_decoder
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
└── vae_encoder
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.pdmodel
```


#### 参数说明

`export_model.py` 各命令行参数的说明。

| 参数 |参数说明 |
|----------|--------------|
| <span style="display:inline-block;width: 230pt"> --pretrained_model_name_or_path </span> | ppdiffuers提供的Diffusion预训练模型，默认为，"runwayml/stable-diffusion-v1-5"。|
| --output_path | 导出的模型目录。 |
| --unet_path | 需要导出的Unet的模型权重，如果指定后我们将会优先使用本参数设置路劲下的文件，默认值为`None`，当我们训练了非lora版本LCM的时候，我们需要设置这个参数。|
| --lcm_lora_path | 需要合并的lcm lora权重的地址，注意当前仅仅支持`kohya`的`safetensors`权重，默认值为`None`。|
| --sample | vae encoder 的输出是否调整为 sample 模式，注意：sample模式会引入随机因素，默认是 False。|

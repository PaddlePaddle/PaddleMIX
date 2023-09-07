## Latent Diffusion Model for U-ViT 从零训练代码

本教程带领大家如何开启 **U-ViT**为底座的**Latent Diffusion Model**的训练。

___注意___:
___官方32层`CompVis/ldm-text2im-large-256`的Latent Diffusion Model使用的是vae，而不是vqvae！而Huggingface团队在设计目录结构的时候把文件夹名字错误的设置成了vqvae！为了与Huggingface团队保持一致，我们同样使用了vqvae文件夹命名！___

## 1 本地运行
### 1.1 安装依赖

在运行这个训练代码前，我们需要安装下面的训练依赖。
```bash
# paddlepaddle-gpu>=2.6.0
python -m pip install paddlepaddle-gpu==2.6.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install -r requirements.txt
```

### 1.2 准备数据

#### COCO数据集:
存放路径：
```
├── data/coco256_features
    ├── train
    ├── val
    ├── empty_context.npy
```

```
wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/coco256_features/train.tar
wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/coco256_features/val.tar
wget https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/coco256_features/empty_context.npy
```


### 1.3 使用trainner开启训练
#### 1.3.1 硬件要求
Tips：
- FP32 在 40GB 的显卡上可正常训练。

#### 1.3.2 单机多卡训练 (多机多卡训练，仅需在 paddle.distributed.launch 后加个 --ips IP1,IP2,IP3,IP4)


```bash
sh run_train_ldm_uvit_small_deep.sh

或者

python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_txt2img_uvit_coco_trainer.py \
    --do_train \
    --output_dir ./output_dir/uvit_t2i_small_deep \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 0.0002 \
    --weight_decay 0.03 \
    --adam_beta1 0.9 \
    --adam_beta2 0.9 \
    --max_steps 1000000 \
    --lr_scheduler_type "constant_with_warmup" \
    --warmup_steps 5000 \
    --image_logging_steps 5000 \
    --logging_steps 5000 \
    --save_steps 10000 \
    --seed 1234 \
    --dataloader_num_workers 8 \
    --max_grad_norm -1 \
    --unet_config_file config/uvit_t2i_small_deep.json \
    --num_inference_steps 50 \
    --model_max_length 77 \
    --use_ema True \
    --overwrite_output_dir \
    --disable_tqdm True \
    --recompute True \
    --fp16 False \
    --fp16_opt_level "O1" \
    --enable_xformers_memory_efficient_attention True \
```

## 2 模型推理

待模型训练完毕，会在`output_dir`保存训练好的模型权重，我们可以使用`generate_pipelines.py`生成推理所使用的`Pipeline`。
```bash
python generate_pipelines.py \
    --model_file ./output_dir/uvit_t2i_small_deep/checkpoint-200000/model_state.pdparams \
    --output_path ./ldm_uvit_pipelines \
    --vae_name_or_path runwayml/stable-diffusion-v1-5/vae \
    --unet_config_file ./config/uvit_t2i_small_deep.json \
    --text_encoder_name_or_path runwayml/stable-diffusion-v1-5/text_encoder \
    --tokenizer_name_or_path runwayml/stable-diffusion-v1-5/tokenizer \
    --model_max_length 77
```
`generate_pipelines.py`代码可传入的参数解释如下：
> * `--model_file`: 我们使用`train_txt2img_laion400m_trainer.py`代码，训练好所得到的`model_state.pdparams`文件。
> * `--output_path`: 生成的pipeline所要保存的路径。
> * `--vae_name_or_path`: 使用的`vae`的名字或者本地路径，注意我们需要里面的`config.json`文件。
> * `--text_encoder_name_or_path`: 使用的`text_encoder`的名字或者本地路径
> * `--unet_config_file`: `unet`的`config`配置文件。
> * `--tokenizer_name_or_path`: 所使用的`tokenizer`名称或者本地路径。
> * `--model_max_length`: `tokenizer`中的`model_max_length`参数，超过该长度将会被截断。


输出的模型目录结构如下：
```shell
├── ldm_uvit_pipelines  # 我们指定的输出文件路径
    ├── model_index.json # 模型index文件
    ├── vqvae # vae权重文件夹！实际是vae模型，文件夹名字与HF保持了一致！
        ├── model_state.pdparams
        ├── config.json
    ├── bert # text_encoder权重文件夹，仍然保持命名为bert，不参与训练
        ├── model_config.json
        ├── model_state.pdparams
    ├── unet # uvit权重文件夹
        ├── model_state.pdparams
        ├── config.json
    ├── scheduler # ddim scheduler文件夹
        ├── scheduler_config.json
    ├── tokenizer # tokenizer文件夹
        ├── tokenizer_config.json
        ├── special_tokens_map.json
        ├── vocab.txt
```

在生成`Pipeline`的权重后，我们可以使用如下的代码进行推理。

```python
from ppdiffusers import LDMTextToImagePipeline
model_name_or_path = "./ldm_uvit_pipelines"
dtype=paddle.float16
pipe = LDMTextToImagePipeline.from_pretrained(model_name_or_path, paddle_dtype=dtype)
prompt = "an elephant under the sea"

pipe.enable_xformers_memory_efficient_attention()
pipe.apply_tome(ratio=0.5) # Can also use pipe.unet in place of pipe here

image = pipe(prompt, guidance_scale=7.5).images[0]
image.save("an_elephant_under_the_sea.png")
```

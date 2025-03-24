# DeepSeek-VL2

[DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2) 基于大型混合专家（Mixture-of-Experts，MoE）视觉语言模型，相较于其前身DeepSeek-VL有了显著提升。DeepSeek-VL2在各种任务中展现出了卓越的能力。本仓库提供了DeepSeek-VL2高性能推理。
支持的权重如下：

|             Model               |
|---------------------------------|
| deepseek-ai/deepseek-vl2-small  |
| deepseek-ai/deepseek-vl2        |

## 环境安装
[安装PaddlePaddle](https://github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
- **python >= 3.10**
- **paddlepaddle-gpu 要求develop版本**
```bash
# Develop 版本安装示例，请确保使用的Paddle版本为develop版本
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu123/
```

2） [安装PaddleMIX环境依赖包](https://github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
```bash
# pip 安装示例，安装paddlemix、ppdiffusers、项目依赖
python -m pip install -e . 
python -m pip install -e ppdiffusers
python -m pip install -r requirements.txt

# 安装PaddleNLP
pip uninstall paddlenlp && rm -rf PaddleNLP
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
pip install -e .
cd csrc
python setup_cuda.py install
```

## 3 高性能推理

### a. fp16 高性能推理

```
export CUDA_VISIBLE_DEVICES=0
export FLAGS_cascade_attention_max_partition_size=163840
export FLAGS_mla_use_tensorcore=1
python deploy/deepseek_vl2/deepseek_vl2_infer.py \
    --model_name_or_path deepseek-ai/deepseek-vl2-small \
    --question "Describe this image." \
    --image_file paddlemix/demo_images/examples_image1.jpg \
    --min_length 128 \
    --max_length 128 \
    --top_k 1 \
    --top_p 0.001 \
    --temperature 0.1 \
    --repetition_penalty 1.05 \
    --block_attn True \
    --inference_model True \
    --append_attn True \
    --mode dynamic \
    --dtype bfloat16 \
    --mla_use_matrix_absorption
```

### b. wint8 高性能推理
```
export CUDA_VISIBLE_DEVICES=0
export FLAGS_cascade_attention_max_partition_size=163840
export FLAGS_mla_use_tensorcore=1
python deploy/deepseek_vl2/deepseek_vl2_infer.py \
    --model_name_or_path deepseek-ai/deepseek-vl2-small \
    --question "Describe this image." \
    --image_file paddlemix/demo_images/examples_image1.jpg \
    --min_length 128 \
    --max_length 128 \
    --top_k 1 \
    --top_p 0.001 \
    --temperature 0.1 \
    --repetition_penalty 1.05 \
    --block_attn True \
    --inference_model True \
    --append_attn True \
    --mode dynamic \
    --dtype bfloat16 \
    --mla_use_matrix_absorption \
    --quant_type "weight_only_int8" 
```

## 4 一键推理 & 推理说明
进入PaddleMIX目录运行
```bash
cd PaddleMIX
sh deploy/deepseek_vl2/shell/run.sh
```
#### 参数设定
|     parameter      |      Value     |
| ------------------ | -------------- |
|       Top-K        |       1        |
|       Top-P        |     0.001      |
|    temperature     |      0.1       |
| repetition_penalty |      1.05      |

#### 单一测试demo执行时，指定max_length=min_length=128，固定输出长度。
|     parameter      |      Value     |
| ------------------ | -------------- |
|     min_length     |       128      |
|     min_length     |       128      |

以下为单张图片的测速情况

|             model              |    Paddle高性能推理    |    Paddle     |
| ------------------------------ | ---------------------| ------------- |
| deepseek-ai/deepseek-vl2-small |          9.3 s       |     12.8 s    |
| deepseek-ai/deepseek-vl2       |           -          |     17.2 s    | 

# DeepSeek-VL2

[DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2) 基于大型混合专家（Mixture-of-Experts，MoE）视觉语言模型，相较于其前身DeepSeek-VL有了显著提升。DeepSeek-VL2在各种任务中展现出了卓越的能力。本仓库提供了DeepSeek-VL2高性能推理。
支持的权重如下：

|             Model               |
|---------------------------------|
| deepseek-ai/deepseek-vl2-small  |

## 环境安装
[安装PaddlePaddle](https://github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
- **python >= 3.10**
- **paddlepaddle-gpu 要求develop版本**
```bash
# Develop 版本安装示例
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu123/
```

2） [安装PaddleMIX环境依赖包](https://github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
```bash
# pip 安装示例，安装paddlemix、ppdiffusers、项目依赖
python -m pip install -e .
python -m pip install -e ppdiffusers
python -m pip install -r requirements.txt

# 安装PaddleNLP
pip uninstall -y paddlenlp && rm -rf PaddleNLP
git clone --depth=1 https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
pip install -e .

# 安装paddlenlp_ops pre-build 
pip install https://paddlenlp.bj.bcebos.com/ops/cu118/paddlenlp_ops-3.0.0b4-py3-none-any.whl
```

## 3 高性能推理

### a. fp16 高性能推理

```
export CUDA_VISIBLE_DEVICES=0
export FLAGS_mla_use_tensorcore=0
export FLAGS_cascade_attention_max_partition_size=128
export FLAGS_cascade_attention_deal_each_time=16
python deploy/deepseek_vl2/deepseek_vl2_infer.py \
    --model_name_or_path deepseek-ai/deepseek-vl2-small \
    --question "Describe this image." \
    --image_file paddlemix/demo_images/examples_image1.jpg \
    --min_length 128 \
    --max_length 128 \
    --inference_model True \
    --append_attn True \
    --mode dynamic \
    --dtype bfloat16 \
    --top_k 1 \
    --top_p 0.001 \
    --temperature 0.1 \
    --repetition_penalty 1.05 \
    --output_via_mq False \
    --benchmark

# 多图推理
python deploy/deepseek_vl2/deepseek_vl2_infer_multi_image.py \
    --model_name_or_path deepseek-ai/deepseek-vl2-small \
    --question "What are in these images." \
    --image_file_1 paddlemix/demo_images/examples_image1.jpg \
    --image_file_2 paddlemix/demo_images/examples_image2.jpg \
    --image_file_3 paddlemix/demo_images/examples_image1.jpg \
    --min_length 128 \
    --max_length 128 \
    --inference_model True \
    --append_attn True \
    --mode dynamic \
    --dtype bfloat16 \
    --top_k 1 \
    --top_p 0.001 \
    --temperature 0.1 \
    --repetition_penalty 1.05 \
    --output_via_mq False \
    --benchmark
```

### b. wint8 高性能推理
```
export CUDA_VISIBLE_DEVICES=0
export FLAGS_mla_use_tensorcore=0
export FLAGS_cascade_attention_max_partition_size=128
export FLAGS_cascade_attention_deal_each_time=16
python deploy/deepseek_vl2/deepseek_vl2_infer.py \
    --model_name_or_path deepseek-ai/deepseek-vl2-small \
    --question "Describe this image." \
    --image_file paddlemix/demo_images/examples_image1.jpg \
    --min_length 128 \
    --max_length 128 \
    --inference_model True \
    --append_attn True \
    --mode dynamic \
    --dtype bfloat16 \
    --top_k 1 \
    --top_p 0.001 \
    --temperature 0.1 \
    --repetition_penalty 1.05 \
    --quant_type "weight_only_int8" \
    --output_via_mq False \
    --benchmark True
```

## 4 一键推理 & 推理说明
cd PaddleMIX
sh deploy/deepseek_vl2/shell/run.sh
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

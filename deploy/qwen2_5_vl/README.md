# Qwen2.5-VL

## 1. 模型介绍

[Qwen2.5-VL
](https: //github.com/QwenLM/Qwen2.5-VL) 是 Qwen 团队推出的一个专注于视觉与语言（Vision-Language, VL）任务的多模态大模型。它旨在通过结合图像和文本信息，提供强大的跨模态理解能力，可以处理涉及图像描述、视觉问答（VQA）、图文检索等多种任务。

| Model              |
|--------------------|
| Qwen/Qwen2.5-VL-3B-Instruct  |
| Qwen/Qwen2.5-VL-7B-Instruct  |
| Qwen/Qwen2.5-VL-72B-Instruct  |

注意：与huggingface权重同名，但权重为paddle框架的Tensor，使用`xxx.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")`即可自动下载该权重文件夹到缓存目录。


## 2 环境准备
1）[安装PaddlePaddle
](https: //github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
- **python >= 3.10**
- **paddlepaddle-gpu 要求develop版本**
```bash
# Develop 版本安装示例
python -m pip install --pre paddlepaddle-gpu -i https: //www.paddlepaddle.org.cn/packages/nightly/cu123/

```

2）[安装PaddleMIX环境依赖包
](https: //github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
```bash
# pip 安装示例，安装paddlemix、ppdiffusers、项目依赖
python -m pip install -e . --user
python -m pip install -e ppdiffusers --user
python -m pip install -r requirements.txt --user

# pip 安装示例，安装develop版本的PaddleNLP、自定义算子
git clone https: //github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
python setup.py install
cd csrc
python setup_cuda.py install
```



## 3 高性能推理

### a. fp16 高性能推理
```bash
cd PaddleMIX
export CUDA_VISIBLE_DEVICES=0
python deploy/qwen2_5_vl/qwen2_5_vl_infer.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --question "Describe this image." \
    --image_file paddlemix/demo_images/examples_image1.jpg \
    --min_length 128 \
    --max_length 128 \
    --top_k 0 \
    --top_p 0.01 \
    --temperature 0.95 \
    --dtype bfloat16 \
    --benchmark True
```


### b. wint8 高性能推理
```bash
export CUDA_VISIBLE_DEVICES=0
python deploy/qwen2_5_vl/qwen2_5_vl_infer.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --question "Describe this image." \
    --image_file paddlemix/demo_images/examples_image1.jpg \
    --min_length 128 \
    --max_length 128 \
    --top_k 0 \
    --top_p 0.01 \
    --temperature 0.95 \
    --dtype bfloat16 \
    --quant_type weight_only_int8 \
    --benchmark True
```

### 一键推理
```bash
cd PaddleMIX
sh deploy/qwen2_5_vl/scripts/qwen2_5_vl.sh
```

## 在 NVIDIA A800-SXM4-80GB 上测试的性能如下：

|             model           | Paddle Inference wint8 | Paddle Inference|    PyTorch     | Paddle 动态图   |
| --------------------------- | ---------------------  | --------------- | -------------- | -------------- |
| Qwen/Qwen2.5-VL-3B-Instruct |      86.96 token/s     |  73.65 token/s  | 24.62 token/s  |  8.72 token/s  |
| Qwen/Qwen2.5-VL-7B-Instruct |      93.97 token/s     |  73.35 token/s  | 22.2  token/s  | 14.57 token/s  |

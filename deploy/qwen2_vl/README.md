# Qwen2-VL

## 1. 模型介绍

[Qwen2-VL
](https: //qwenlm.github.io/blog/qwen2-vl/) 是大规模视觉语言模型。可以以图像、文本、检测框、视频作为输入，并以文本和检测框作为输出。本仓库提供paddle版本的`Qwen2-VL-2B-Instruct`和`Qwen2-VL-7B-Instruct`模型。

## 2 环境准备
- **python >= 3.10**
- **paddlepaddle-gpu 要求版本develop**
```
# 安装示例
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https: //www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```
- **paddlenlp
```
# 安装示例
git submodule update --init --recursive
cd PaddleNLP
git reset --hard e91c2d3d634b12769c30aa419ddf931c20b7ca9f
pip install -e .
cd csrc
python setup_cuda.py install
```

> 注：
* 请确保安装了以上依赖，否则无法运行。同时，需要安装 paddlemix/external_ops 下的自定义OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置PYTHONPATH
* (默认开启flash_attn)使用flash_attn 要求A100/A800显卡或者H20显卡

## 3 高性能推理
# 在Qwen2-vl的推理优化中，我们在视觉模型部分继续使用paddlemix中的模型组网；
  但是在语言模型部分，我们调用Paddlenlp中高性能的qwen2语言模型，以得到高性能的Qwen2-vl推理版本。

### a. 文本&单张图像输入高性能推理
```bash
python deploy/qwen2_vl/single_image_infer.py \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dtype bfloat16 \
    --benchmark 1
```

- 在 NVIDIA A100-SXM4-80GB 上测试的性能如下：


- Qwen2-VL-2B-Instruct
| Paddle Inference|    PyTorch   | Paddle 动态图 |
| --------------- | ------------ | ------------ |
|      1.44 s     |     2.35 s   |    5.215 s   |


- Qwen2-VL-7B-Instruct
| Paddle Inference|    PyTorch   | Paddle 动态图 |
| --------------- | ------------ | ------------ |
|      1.73 s     |      4.4s    |    6.339 s   |

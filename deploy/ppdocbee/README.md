# PP-DocBee

## 1. 模型介绍

PP-DocBee 是PaddleMIX团队自研的一款专注于文档理解的多模态大模型，在中文文档理解任务上具有卓越表现。该模型是基于Qwen2-VL-2B架构针对文档理解场景进行优化的，通过近 500 万条文档理解类多模态数据集进行微调优化，各种数据集包括了通用VQA类、OCR类、图表类、text-rich文档类、数学和复杂推理类、合成数据类、纯文本数据等，并设置了不同训练数据配比。在学术界权威的几个英文文档理解评测榜单上，PP-DocBee基本都达到了同参数量级别模型的SOTA。在内部业务中文场景类的指标上，PP-DocBee也高于目前的热门开源和闭源模型。

## 2 环境准备

- **python >= 3.10**
- **paddlepaddle-gpu 要求>=3.0.0b2或版本develop**
```
# paddlepaddle-gpu develop版安装示例
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

- **paddlenlp 需要特定版本**

在PaddleMIX/代码目录下执行以下命令安装特定版本的paddlenlp：
```bash
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

在PP-DocBee的高性能推理优化中，**视觉模型部分继续使用PaddleMIX中的模型组网；但是语言模型部分调用PaddleNLP中高性能的Qwen2语言模型**，以得到高性能的PP-DocBee推理版本。

### 3.1. 文本&单张图像输入高性能推理
```bash
python deploy/ppdocbee/single_image_infer.py \
    --model_name_or_path PaddleMIX/PPDocBee-2B-1129 \
    --dtype bfloat16 \
    --benchmark True \
```

- 在 NVIDIA A100-SXM4-80GB 上测试的内部业务中文场景平均端到端速度性能如下：

| model                  | Paddle Inference|    PyTorch   | Paddle 动态图 |
| ---------------------- | --------------- | ------------ | ------------ |
| PPDocBee-2B   |      0.9267 s     |     1.7114 s   |    1.5935 s   |

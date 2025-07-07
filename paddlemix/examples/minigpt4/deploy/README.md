# MiniGPT4 推理加速

本项目提供了基于 MiniGPT4 的推理加速功能，基本的解决思路是将 MiniGPT4 动态图转为静态图，然后基于 PaddleInference 库进行推理加速。

下图展示了 MiniGPT4 的整体模型结构， 可以看到整体上，MiniGPT4的主要部分由 VIT， QFormer 和 Vicuna 模型组成，其中 Vicuna 模型是基于 Llama 训练的，在代码实现中调用的也是Llama代码，为方便描述，忽略不必要的分歧，所以在后续中将语言模型这部分默认描述为Llama。

在本方案中，我们将MiniGPT4 导出为两个子图：VIT 和 QFormer部分导出为一个静态子图， Llama 部分导出为一个子图。后续会结合这两个子图统一做 MiniGPT4 的推理功能。

<center><img src="https://github.com/PaddlePaddle/Paddle/assets/35913314/f0306cb6-4837-4f52-8f57-a0e7e35238f6" /></center>




## 1. 环境准备
### 1.1 基础环境准备：
本项目在以下基础环境进行了验证：
- CUDA: 11.7
- python: 3.11
- paddle: develop版

其中CUDA版本需要>=11.2， 具体Paddle版本可以点击[这里](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)按需下载。


### 1.2 安装项目库
1. 本项目需要用到 PaddleMIX 和 PaddleNLP 两个库，并且需要下载最新的 develop 版本：

```shell
git clone https://github.com/PaddlePaddle/PaddleNLP.git
git clone https://github.com/PaddlePaddle/PaddleMIX.git
```

2. 安装paddlenlp_ops：
```shell
cd PaddleNLP/csrc
python setup_cuda.py install
```

3. 最后设置相应的环境变量：
```shell
export PYTHONPATH= yourpath/PaddleNLP:yourpath/PaddleMIX
```

### 1.3 特别说明
目前需要修复PaddleNLP和Paddle的部分代码，从而进行MiniGPT4推理加速。这部分功能后续逐步会逐步完善到PaddleNLP和Paddle，但目前如果想使用的话需要手动修改一下。
1. 修改PaddleNLP代码:
参考该[分支代码](https://github.com/1649759610/PaddleNLP/tree/bugfix_minigpt4)，依次替换以下文件：
- PaddleNLP/paddlenlp/experimental/transformers/generation_utils.py
- PaddleNLP/paddlenlp/experimental/transformers/llama/modeling.py
- PaddleNLP/llm/export_model.py

2. 修改Paddle代码
进入到Paddle安装目录，打开文件：paddle/static/io.py, 注释第284-287行代码：
```python
if not skip_prune_program:
    copy_program = copy_program._prune_with_input(
        feeded_var_names=feed_var_names, targets=fetch_vars
    )
```

## 2. MiniGPT4 分阶段导出

### 2.1 导出前一部分子图：
请确保在该目录下：PaddleMIX/paddlemix/examples/minigpt4/inference，按照以下命令进行导出：
```
python export_image_encoder.py \
    --minigpt4_13b_path "you minigpt4 dir path" \
    --save_path "./checkpoints/encode_image/encode_image"
```

**参数说明**:
- minigpt4_13b_path: 存放MiniGPT4的目录名
- save_path: 前一部分模型的导出路径和名称


### 2.2 导出后一部分子图
请进入到目录： PaddleNLP/llm, 按照以下命令进行导出：
```
python export_model.py \
    --model_name_or_path "your llama dir path" \
    --output_path "your output path" \
    --dtype float16 \
    --inference_model \
    --model_prefix llama \
    --model_type llama-img2txt

```

**参数说明**:
- model_name_or_path: 存放Llama模型的目录名
- output_path: 语言模型部分的导出路径和名称
- dtype: 模型权重数据类型
- inference_model: 表示是推理模型
- model_prefix: 指明模型前缀
- model_type: 指明模型类型

**备注**： 当前导出Llama部分需要转移到PaddleNLP下进行手动导出，后续将支持在PaddleMIX下一键转出。

## 3. MiniGPT4 静态图推理
请进入到目录PaddleMIX/paddlemix/examples/minigpt4/inference，执行以下命令：
```python
python run_static_predict.py \
    --first_model_path "The dir name of image encoder model" \
    --second_model_path "The dir name of language model" \
    --minigpt4_path "The minigpt4 dir name of saving tokenizer"
```

**参数说明**:
- first_model_path: 存放前一部分（即vit和qformer）的静态图模型目录名
- second_model_path: 存放后一部分（即语言模型）的静态图模型目录名
- minigpt4_path: 存放 MiniGPT4 tokenizer的目录名

以下展示了针对以下这个图片，MiniGPT4静态图推理的输出：

<center><img src="https://paddlenlp.bj.bcebos.com/data/images/mugs.png" /></center>

```text
Reference: The image shows two black and white cats sitting next to each other on a blue background. The cats have black fur and white fur with black noses, eyes, and paws. They are both looking at the camera with a curious expression. The mugs are also blue with the same design of the cats on them. There is a small white flower on the left side of the mug. The background is a light blue color.

Outputs:  ['The image shows two black and white cats sitting next to each other on a blue background. The cats have black fur and white fur with black noses, eyes, and paws. They are both looking at the camera with a curious expression. The mugs are also blue with the same design of the cats on them. There is a small white flower on the left side of the mug. The background is a light blue color.##']
```

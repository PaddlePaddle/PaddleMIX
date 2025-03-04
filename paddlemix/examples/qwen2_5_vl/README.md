# Qwen2.5-VL

## 1. 模型介绍

[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) 是 Qwen 团队推出的一个专注于视觉与语言（Vision-Language, VL）任务的多模态大模型。它旨在通过结合图像和文本信息，提供强大的跨模态理解能力，可以处理涉及图像描述、视觉问答（VQA）、图文检索等多种任务。

**主要增强功能：**

**强大的文档解析能力：** 将文本识别升级为全文档解析，能够出色地处理多场景、多语言、各类内置（手写、表格、图表、化学式、乐谱）文档。

**跨格式的精确物体定位：** 提高检测、指向和计数物体的准确度，适应绝对坐标和 JSON 格式，实现高级空间推理。

**超长视频理解和细粒度视频解析：** 将原生动态分辨率扩展到时间维度，增强理解数小时视频的能力，同时在数秒内提取事件片段。

**增强计算机和移动设备的代理功能：** 利用先进的基础、推理和决策能力，通过智能手机和计算机上的卓越代理功能增强模型。

**本仓库支持的模型权重:**

| Model              |
|--------------------|
| Qwen/Qwen2.5-VL-3B-Instruct  |
| Qwen/Qwen2.5-VL-7B-Instruct  |
| Qwen/Qwen2.5-VL-72B-Instruct  |

注意：与huggingface权重同名，但权重为paddle框架的Tensor，使用`xxx.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")`即可自动下载该权重文件夹到缓存目录。


## 2 环境准备
1）[安装PaddlePaddle](https://github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
- **python >= 3.10**
- **paddlepaddle-gpu 要求是3.0.0b2或develop版本**
```bash
# 提供三种 PaddlePaddle 安装命令示例，也可参考PaddleMIX主页的安装教程进行安装

# 3.0.0b2版本安装示例 (CUDA 11.8)
python -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Develop 版本安装示例
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

# sh 脚本快速安装
sh build_paddle_env.sh
```

2）[安装PaddleMIX环境依赖包](https://github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
- **paddlenlp >= 3.0.0b3**

```bash
# 提供两种 PaddleMIX 依赖安装命令示例

# pip 安装示例，安装paddlemix、ppdiffusers、项目依赖、paddlenlp
python -m pip install -e . --user
python -m pip install -e ppdiffusers --user
python -m pip install -r requirements.txt --user
python -m pip install paddlenlp==3.0.0b3 --user

# sh 脚本快速安装
sh build_env.sh
```

> 注：
* 请确保安装了以上依赖，否则无法运行。同时，需要安装 paddlemix/external_ops 下的自定义OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置PYTHONPATH
* (默认开启flash_attn)使用flash_attn 要求A100/A800显卡或者H20显卡。V100请用float16推理。

## 3 推理预测

### a. 单图预测 (单卡 32G A卡V卡 显存可运行3B模型)
```bash
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/qwen2_5_vl/single_image_infer.py
```

### b. 多图预测 (单卡 32G A卡V卡 显存可运行3B模型)
```bash
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/qwen2_5_vl/multi_image_infer.py
```

### c. 视频预测 (单卡 32G A卡V卡 显存可运行3B模型)
```bash
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/qwen2_5_vl/video_infer.py
```

### d. batch推理 (单卡 32G A卡V卡 显存可运行3B模型)
```bash
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/qwen2_5_vl/batch_infer.py
```
### 模型推理支持分布式推理

```bash
# 3B (多卡 32G A卡V卡 显存可运行3B模型)
sh paddlemix/examples/qwen2_5_vl/shell/distributed_qwen2_5_vl_infer_3B.sh
# 7B (多卡 40G A卡 显存可运行7B模型)
sh paddlemix/examples/qwen2_5_vl/shell/distributed_qwen2_5_vl_infer_7B.sh
# 72B (多卡 40G A卡 显存可运行72B模型)
sh paddlemix/examples/qwen2_5_vl/shell/distributed_qwen2_5_vl_infer_72B.sh
```
> ⚠️注意："mp_degree"需要根据显卡数量"gpus"进行调整，例如2卡推理，则设置为2。

## 4 模型微调 (40G A卡 显存可运行3B模型)

### 4.1 小型示例数据集

PaddleMIX团队整理了`chartqa`数据集作为小型的示例数据集，下载链接为：

```bash
wget https://paddlenlp.bj.bcebos.com/models/community/paddlemix/benchmark/playground.tar # 1.0G
```
playground/目录下包括了图片目录`data/chartqa/`和标注目录`opensource_json/`，详见`paddlemix/examples/qwen2_5_vl/configs/demo_chartqa_500.json`。


### 4.2 大型公开数据集

大型的数据集选择6个公开的数据集组合，包括`dvqa`、`chartqa`、`ai2d`、`docvqa`、`geoqa+`、`synthdog_en`，详见`paddlemix/examples/qwen2_5_vl/configs/baseline_6data_330k.json`

PaddleMIX团队整理后的下载链接为：
```bash
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground.tar # 50G
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground/opensource_json.tar
```

注意：若先下载了示例数据集的`playground.tar`解压了，此处需删除后，再下载公开数据集的`playground.tar`并解压，opensource_json.tar需下载解压在playground/目录下，opensource_json 里是数据标注的json格式文件。

### 4.3 微调命令

注意：此微调训练为语言模型微调，冻结视觉编码器而放开LLM训练，2B模型全量微调训练的显存大小约为30G，7B模型全量微调训练的显存大小约为75G。

```bash
# 3B (单张40G A卡 显存可运行3B模型)
sh paddlemix/examples/qwen2_5_vl/shell/baseline_3b_lora_bs32_1e8_1mp.sh

# 3B (多张40G A卡 显存可运行3B模型)
sh paddlemix/examples/qwen2_5_vl/shell/baseline_3b_bs32_1e8.sh

# 3B lora (多张40G A卡 显存可运行3B模型)
sh paddlemix/examples/qwen2_5_vl/shell/baseline_3b_lora_bs32_1e8.sh

# 7B (多张80G A卡 显存可运行7B模型)
sh paddlemix/examples/qwen2_5_vl/shell/baseline_7b_bs32_1e8.sh

# 7B lora (多张80G A卡 显存可运行7B模型)
sh paddlemix/examples/qwen2_5_vl/shell/baseline_7b_lora_bs32_1e8.sh
```

### 4.4 微调后使用

同按步骤3中的模型推理预测，只需将`paddlemix/examples/qwen2_5_vl/single_image_infer.py`中的`--model_path`参数修改为微调后的模型路径即可。

```bash
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/qwen2_5_vl/single_image_infer.py
```

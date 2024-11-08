# Qwen2-VL

## 1. 模型介绍

[Qwen2-VL](https://qwenlm.github.io/blog/qwen2-vl/) 是大规模视觉语言模型。可以以图像、文本、检测框、视频作为输入，并以文本和检测框作为输出。本仓库提供paddle版本的`Qwen2-VL-2B-Instruct`和`Qwen2-VL-7B-Instruct`模型。


## 2 环境准备
- **python >= 3.10**
- **paddlepaddle-gpu 要求版本develop**
```
# 安装示例
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

- **paddlenlp == 3.0.0b2**

> 注：
* 请确保安装了以上依赖，否则无法运行。同时，需要安装 paddlemix/external_ops 下的自定义OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置PYTHONPATH
* (默认开启flash_attn)使用flash_attn 要求A100/A800显卡或者H20显卡

## 3 推理预测

### a. 单图预测
```bash
python paddlemix/examples/qwen2_vl/single_image_infer.py
```

### b. 多图预测
```bash
python paddlemix/examples/qwen2_vl/multi_image_infer.py
```

### c. 视频预测
```bash
python paddlemix/examples/qwen2_vl/video_infer.py
```

## 4 模型微调

### 4.1 微调数据准备

SFT数据集选择6个公开的数据集，包括`dvqa`、`chartqa`、`ai2d`、`docvqa`、`geoqa+`、`synthdog_en`，详见`paddlemix/examples/qwen2_vl/configs/baseline_6data_330k.json`

PaddleMIX团队整理后的下载链接为：
```
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground.tar # 50G
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground/opensource_json.tar
```
opensource_json.tar需下载解压在playground/目录下，opensource_json 里是数据标注的json格式文件。

### 4.2 微调命令

注意：此微调训练为全参数微调，冻结视觉编码器而放开LLM训练，2B模型微调训练的显存大小约为30G，7B模型微调训练的显存大小约为75G。

```bash
# 2B
sh paddlemix/examples/qwen2_vl/shell/basline_2b_bs32_1e8.sh

# 7B
sh paddlemix/examples/qwen2_vl/shell/basline_7b_bs32_1e8.sh
```

### 4.3 微调后使用

同按步骤3中的模型推理预测，只需将`paddlemix/examples/qwen2_vl/single_image_infer.py`中的`MODEL_NAME`参数修改为微调后的模型路径即可。

```bash
python paddlemix/examples/qwen2_vl/single_image_infer.py
```


## 参考文献
```BibTeX
@article{Qwen2-VL,
  title={Qwen2-VL},
  author={Qwen team},
  year={2024}
}
```

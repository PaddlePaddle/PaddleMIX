# Qwen2-VL

## 1. 模型介绍

[Qwen2-VL](https://qwenlm.github.io/blog/qwen2-vl/) 是大规模视觉语言模型。可以以图像、文本、检测框、视频作为输入，并以文本和检测框作为输出。
本仓库提供paddle版本的Qwen2-VL-2B-Instruct和Qwen2-VL-7B-Instruct模型。


## 2 环境准备
- **python >= 3.10**
- **paddlepaddle-gpu 要求版本develop**
```
# 安装示例
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```


- paddlenlp >= 3.0.0(默认开启flash_attn，推荐源码编译安装)

> 注：
* 请确保安装了以上依赖，否则无法运行。同时，需要安装 paddlemix/external_ops 下的自定义OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置PYTHONPATH
* 使用flash_attn 要求H或者A卡，开启后显存变化如下：2B模型: 49G -> 13G ｜ 7B模型: 61G -> 25G

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

数据集选择6个公开数据集，包括，详见`paddlemix/examples/qwen2_vl/configs/baseline_6data_330k.json`

下载链接：

```bash
sh paddlemix/examples/qwen2_vl/shell/basline_2b_bs32_1e8.sh
```


## 参考文献
```BibTeX
@article{Qwen2-VL,
  title={Qwen2-VL},
  author={Qwen team},
  year={2024}
}
```

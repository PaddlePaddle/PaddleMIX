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

- **paddlenlp == 3.0.0b2**

> 注：
* 请确保安装了以上依赖，否则无法运行。同时，需要安装 paddlemix/external_ops 下的自定义OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置PYTHONPATH
* (默认开启flash_attn)使用flash_attn 要求A100/A800显卡或者H20显卡

## 3 推理预测

### a. 文本&单张图像输入高性能推理
```bash
python /root/paddlejob/workspace/env_run/output/changwenbin/tmp_qwen2vl/PaddleMIX/deploy/qwen2_vl/single_image_infer.py \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dtype bfloat16 \
    --mode dynamic \
    --inference_model 1 \
    --append_attn 1 \
    --benchmark 1
```



## 参考文献
```BibTeX
@article{Qwen2-VL,
  title={Qwen2-VL
  },
  author={Qwen team
  },
  year={
    2024
  }
}
```

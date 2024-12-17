# GOT-OCR2.0

## 1. 模型介绍

[GOT-OCR2.0](https://arxiv.org/abs/2409.01704)是一款极具突破性的通用OCR模型，旨在解决传统OCR系统（OCR-1.0）和当前大规模视觉语言模型（LVLMs）在OCR任务中的局限性。本仓库提供paddle版本的`GOT-OCR2.0`模型。


## 2. 环境要求
- **python >= 3.10**
- **paddlepaddle-gpu 要求3.0.0b2或版本develop**
```
# develop版安装示例
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

- **paddlenlp == 3.0.0b2**

> 注：(默认开启flash_attn)使用flash_attn 要求A100/A800显卡或者H20显卡。V100请用float16推理。


## 3 推理预测

### 3.1. plain texts OCR:
```bash
python paddlemix/examples/GOT_OCR_2_0/got_ocr2_0_infer.py \
  --model_name_or_path stepfun-ai/GOT-OCR2_0 \
  --image_file paddlemix/demo_images/hospital.jpeg \
  --ocr_type ocr \
```

### 3.2. format texts OCR:
```bash
python paddlemix/examples/GOT_OCR_2_0/got_ocr2_0_infer.py \
  --model_name_or_path stepfun-ai/GOT-OCR2_0 \
  --image_file paddlemix/demo_images/hospital.jpeg \
  --ocr_type format \
```

## 4 训练
```bash
sh paddlemix/examples/GOT_OCR_2_0/run_train.sh
```


## 参考文献
```BibTeX
@article{wei2024general,
  title={General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model},
  author={Wei, Haoran and Liu, Chenglong and Chen, Jinyue and Wang, Jia and Kong, Lingyu and Xu, Yanming and Ge, Zheng and Zhao, Liang and Sun, Jianjian and Peng, Yuang and others},
  journal={arXiv preprint arXiv:2409.01704},
  year={2024}
}
```

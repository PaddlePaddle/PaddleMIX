# GOT-OCR2.0

## 1. 模型介绍

[GOT-OCR2.0](https://arxiv.org/abs/2409.01704)是由 StepFun 和中国科学院大学推出的专用于通用 OCR 任务的多模态大模型，参数量 0.6B，是一款极具突破性的通用OCR多模态模型，旨在解决传统OCR系统（OCR-1.0）和当前大规模视觉语言模型（LVLMs）在OCR任务中的局限性。

**本仓库支持的模型权重:**

| Model              |
|--------------------|
| stepfun-ai/GOT-OCR2_0  |

注意：与huggingface权重同名，但权重为paddle框架的Tensor，使用`xxx.from_pretrained("stepfun-ai/GOT-OCR2_0")`即可自动下载该权重文件夹到缓存目录。


## 2. 环境要求
- **python >= 3.10**
- **paddlepaddle-gpu 要求3.0.0b2版本或develop版本**
```
# 安装示例
python -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

- **paddlenlp == 3.0.0b3**
- **paddlenlp要求是3.0.0b3版本**
```
# 安装示例
python -m pip install paddlenlp==3.0.0b3
```

- **其他环境要求**
```
pip install -r requirements.txt
```

## 3 推理预测

注意：GOT-OCR2.0 模型推理显存约需4G，不支持数据类型为"float16"进行推理。

### 3.1. plain texts OCR:
```bash
python paddlemix/examples/GOT_OCR_2_0/got_ocr2_0_infer.py \
  --model_name_or_path stepfun-ai/GOT-OCR2_0 \
  --image_file paddlemix/demo_images/hospital.jpeg \
  --ocr_type ocr \
  --dtype "bfloat16" \
```

### 3.2. format texts OCR:
```bash
python paddlemix/examples/GOT_OCR_2_0/got_ocr2_0_infer.py \
  --model_name_or_path stepfun-ai/GOT-OCR2_0 \
  --image_file paddlemix/demo_images/hospital.jpeg \
  --ocr_type format \
  --dtype "bfloat16" \
```

### 3.3. multi_crop plain texts OCR:
```bash
python paddlemix/examples/GOT_OCR_2_0/got_ocr2_0_infer.py \
  --model_name_or_path stepfun-ai/GOT-OCR2_0 \
  --image_file paddlemix/demo_images/hospital.jpeg \
  --ocr_type ocr \
  --multi_crop \
  --dtype "bfloat16" \
```

## 4 训练

与[官方github代码库](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/?tab=readme-ov-file#train)一样，目前仅支持基于GOT权重的post-training(stage-2/stage-3)，其中stage2是全参数微调，stage3是冻结vision encoder后微调，默认训练方式是stage2全参数微调，训练显存约10GB每卡。

### 数据集下载
PaddleMIX团队提供了一个改版的SynthDoG-EN数据集，统一修改了其原先的question为```<image>\nOCR:```，下载链接为：
```
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground/synthdog_en.tar # 2.4G
```
synthdog_en.tar包括了图片images文件夹和标注json文件，需下载解压或软链接在PaddleMIX/目录下。

### 数据集格式

同[官方例子](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/blob/main/assets/train_sample.jpg)，其中question统一为```<image>\nOCR:```，answer是其OCR结果。


### 训练命令

```bash
sh paddlemix/examples/GOT_OCR_2_0/run_train.sh
```

注意：默认训练方式是stage2全参数微调，训练显存约10GB每卡。也可通过设置```--freeze_vision_tower True```冻结vision encoder后微调。

### 训完后推理

```bash
python paddlemix/examples/GOT_OCR_2_0/got_ocr2_0_infer.py \
  --model_name_or_path work_dirs/got_ocr_20/ \
  --image_file paddlemix/demo_images/hospital.jpeg \
  --ocr_type ocr \
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

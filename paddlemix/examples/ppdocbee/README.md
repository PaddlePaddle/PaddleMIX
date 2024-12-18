# PP-DocBee


# PP-DocBee

## 1. 简介

PP-DocBee 是一款专注于文档理解的多模态大模型，在中文文档理解任务上具有卓越表现。该模型基于```Qwen/Qwen2-VL-2BInstruct```架构，通过近 500 万条文档理解类多模态数据和精选的纯文本数据进行微调优化。

## 2. 环境要求
- **python >= 3.10**
- **paddlepaddle-gpu 要求3.0.0b2或版本develop**
```
# develop版安装示例
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

- **paddlenlp == 3.0.0b2**

> 注：(默认开启flash_attn)使用flash_attn 要求A100/A800显卡或者H20显卡。V100请用float16推理。


## 3. 在线体验和部署

### 3.1 在线体验
https://github.com/user-attachments/assets/8e74c364-6d65-4930-b873-6fd5df263d9a

我们提供了在线体验环境，您可以通过[AI Studio](https://aistudio.baidu.com/application/detail/60135)快速体验 PP-DocBee 的功能。

### 3.2 本地gradio部署
```bash
# 安装gradio
pip install gradio==5.6.0
# 运行gradio
python paddlemix/examples/ppdocbee/app.py
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/f6961b29-c168-4e61-b005-032f010dc2ee" width="90%" alt="示例图片"/>
</p>

### 3.3 OpenAI服务部署
我们提供了基于OpenAI服务部署的代码，您可以通过阅读[服务部署文档](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/paddlemix/examples/qwen2_vl/README_SERVER.md)快速搭建服务。



## 4. 使用指南

### 4.1 模型推理

下面展示了一个表格识别的示例：

<p align="center">
  <img src="https://github.com/user-attachments/assets/6a03a848-c396-4b2f-a7f3-47ff1441c750" width="50%" alt="示例图片"/>
</p>

```bash
python paddlemix/examples/ppdocbee/single_image_infer.py \
  --model_path "PaddleMIX/PPDocBee-2B-1129" \
  --image_file "paddlemix/demo_images/medal_table.png" \
  --question "识别这份表格的内容"
```

输出示例：
```
| 名次 | 国家/地区 | 金牌 | 银牌 | 铜牌 | 奖牌总数 |
| --- | --- | --- | --- | --- | --- |
| 1 | 中国（CHN） | 48 | 22 | 30 | 100 |
| 2 | 美国（USA） | 36 | 39 | 37 | 112 |
| 3 | 俄罗斯（RUS） | 24 | 13 | 23 | 60 |
| 4 | 英国（GBR） | 19 | 13 | 19 | 51 |
| 5 | 德国（GER） | 16 | 11 | 14 | 41 |
| 6 | 澳大利亚（AUS） | 14 | 15 | 17 | 46 |
| 7 | 韩国（KOR） | 13 | 11 | 8 | 32 |
| 8 | 日本（JPN） | 9 | 8 | 8 | 25 |
| 9 | 意大利（ITA） | 8 | 9 | 10 | 27 |
| 10 | 法国（FRA） | 7 | 16 | 20 | 43 |
| 11 | 荷兰（NED） | 7 | 5 | 4 | 16 |
| 12 | 乌克兰（UKR） | 7 | 4 | 11 | 22 |
| 13 | 肯尼亚（KEN） | 6 | 4 | 6 | 16 |
| 14 | 西班牙（ESP） | 5 | 11 | 3 | 19 |
| 15 | 牙买加（JAM） | 5 | 4 | 2 | 11 |
```

## 5. 性能评测

### 5.1 准确率评测

Benchamrk | MiniCPM-V 2.0 | SmolVLM | Aquila-VL-2B | Mini-Monkey-2B | InternVL2-2B | InternVL2.5-2B | Qwen2-VL-2B | **PP-DocBee**
-- | -- | -- | -- | -- | -- | -- | -- | --
Model Size | 2.43B | 2.25B | 2.18B | 2.21B | 2.21B | 2.21B | 2.21B | 2.21B
DocVQA-val | 71.9(test) | 81.6(test) | 85.0 | 87.4(test) | 86.9(test) | 88.7(test) | 89.2 | **90.1**
ChartQA-test | - | - | 76.5 | 76.5 | 76.2 | **79.2** | 73.5 | 74.6
InfoVQA-val | - | - | 58.3 | 60.1(test) | 58.9(test) | 60.9(test) | 64.1 | **65.4**
TextVQA-val | 74.1 | 72.7 | 76.4 | 76.0 | 73.4 | 74.3 | 79.7 | **81.2**
OCRBench | 605 | - | 77.2 | 79.4 | 781 | 80.4 | 79.4 | **82.8**
ChineseOCRBench | - | - |   |   | - | - | 76.1 | **80.2**
内部中文场景评估集 | - | - |   |   | 44.1 | - | 52.8 | **60.3**

注意：

1.我们在评估DocVQA和InfoVQA时默认采用了val验证集上的指标，标(test)的是竞品模型公布的测试集上的指标。

2.[ChineseOCRBench](https://huggingface.co/datasets/SWHL/ChineseOCRBench)是3410张图像和3410条问答数据，均来自ReCTS和ESTVQA数据集。

3.内部中文场景评估集包括了财报、法律法规、理工科论文、说明书、文科论文、合同、研报等场景，暂时未有计划公开。

# PP-DocBee


# PP-DocBee

## 1. 简介

PP-DocBee 是一款专注于文档理解的多模态大模型，在中文文档理解任务上具有卓越表现。该模型基于```Qwen/Qwen2-VL-2BInstruct```架构，通过近 500 万条文档理解类多模态数据和精选的纯文本数据进行微调优化。

**本仓库支持的模型权重:**

| Model              |
|--------------------|
| PaddleMIX/PPDocBee-2B-1129 |


## 2. 环境要求
- **python >= 3.10**
- **paddlepaddle-gpu 要求3.0.0b2或版本develop**
```
# develop版安装示例
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

- **paddlenlp == 3.0.0b3**
```
# 安装示例
python -m pip install paddlenlp==3.0.0b3
```

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
python paddlemix/examples/ppdocbee/ppdocbee_infer.py \
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

### 4.2 模型微调

### 4.2.1 小型示例数据集

PaddleMIX团队整理了`chartqa`数据集作为小型的示例数据集，下载链接为：

```bash
wget https://paddlenlp.bj.bcebos.com/models/community/paddlemix/benchmark/playground.tar # 1.0G
```
playground/目录下包括了图片目录`data/chartqa/`和标注目录`opensource_json/`，详见`paddlemix/examples/ppdocbee/configs/demo_chartqa_500.json`。


### 4.2.2 大型公开数据集

PP-DocBee模型的SFT训练数据集，包括了众多文档类的指令微调数据集，例如：`dvqa`、`chartqa`、`ai2d`、`docvqa`、`geoqa+`、`synthdog_en`、`LLaVA-OneVision`系列以及内部合成数据集，部分公开数据集详见`paddlemix/examples/ppdocbee/configs/ppdocbee_public_dataset.json`，内部合成数据集暂时不对外开放。

PaddleMIX团队整理后的下载链接为：
```bash
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground.tar # 50G
wget https://paddlenlp.bj.bcebos.com/datasets/paddlemix/playground/opensource_json.tar
```

注意：若先下载了示例数据集的`playground.tar`解压了，此处需删除后，再下载公开数据集的`playground.tar`并解压，opensource_json.tar需下载解压在playground/目录下，opensource_json 里是数据标注的json格式文件。

PaddleMIX团队整理后的`LLaVA-OneVision`系列数据集待开放下载链接，请关注后续更新。


### 4.3 微调命令

注意：此微调训练为语言模型微调，冻结视觉编码器而放开LLM训练，2B模型全量微调训练的显存大小约为30G。

```bash
# 2B
sh paddlemix/examples/ppdocbee/shell/ppdocbee_sft.sh

# 2B lora
sh paddlemix/examples/ppdocbee/shell/ppdocbee_lora.sh
```

注意：默认是公开数据集训练的配置，若需使用示例数据集，请在`ppdocbee_sft.sh`或`ppdocbee_lora.sh`中修改`--meta_path`为`paddlemix/examples/ppdocbee/configs/demo_chartqa_500.json`。

### 4.4 微调后使用

只需将`paddlemix/examples/ppdocbee/ppdocbee_infer.py`中的`--model_path`参数修改为微调后的模型路径即可。

```bash
python paddlemix/examples/ppdocbee/ppdocbee_infer.py \
  --model_path "your_trained_model_path" \
  --image_file "paddlemix/demo_images/medal_table.png" \
  --question "识别这份表格的内容"
```


## 5. 性能评测

### 5.1 精度评测

Benchamrk         | Params  | DocVQA-val | ChartQA-test | InfoVQA-val | TextVQA-val | OCRBench | ChineseOCRBench | **内部中文场景评估集**
----------------- | ------- | ---------- | ------------ | ----------- | ----------- | -------- | --------------- | -------------------
GPT-4V           |Closed Model| 87.2(test) | 78.1         |   75.1(test)|  78.0       | 64.5     |   -             |  -
GPT-4o           |Closed Model| 92.8(test) | 85.7         |   79.2(test)|  77.4       | 73.6     |   -             |  -
Claude 3.5 Sonnet|Closed Model| 95.2(test) | 90.8         |   74.1(test)|  74.1       | 78.8     |   -             |  -
Gemini-1.5-Pro   |Closed Model| 93.1(test) | 87.2         |   80/1(test)|  78.7       | 75.4     |   -             |  -
MiniCPM-V 2.0     | 2.43B   | 71.9(test) | -            |       -     |  74.1       | 60.5     |   -             |  -
SmolVLM           | 2.25B   | 81.6(test) | -            |       -     |  72.7       | -        |   -             |  -
Aquila-VL-2B      | 2.18B   | 85.0(test) | 76.5         | 58.3(test)  |  76.4       |  77.2    |  -              | -
Mini-Monkey-2B    | 2.21B   | 87.4(test) | 76.5         | 60.1(test)  |  76.0       |  79.4    |  -              | -
InternVL2-2B      | 2.21B   | 86.9(test) | 76.2         | 58.9(test)  |   73.4      |  78.1    | -               |    44.1
InternVL2.5-2B    | 2.21B   | 88.7(test) |  79.2        |  60.9(test) | 74.3        | 80.4     |  -              | -
DeepSeek-VL2-Tiny | *1.0B   | 88.9(test) |  **81.0**    |  66.1(test) | 80.7        | 80.9     | -               | -
Qwen2-VL-2B       | 2.21B   | 89.2       |  73.5        |  64.1       | 79.7        | 79.4     | 76.1             |  52.8
**PPDocBee-2B-1129**| 2.21B   | **90.1**   |  74.6        |  **65.4**   |   **81.2**  | **82.8** | **80.2**         | **60.3**


注意：

1.我们在评估DocVQA和InfoVQA时默认采用了val验证集上的指标，标(test)的是竞品模型公布的测试集上的指标。

2.[ChineseOCRBench](https://huggingface.co/datasets/SWHL/ChineseOCRBench)是3410张图像和3410条问答数据，均来自ReCTS和ESTVQA数据集。

3.内部中文场景评估集包括了财报、法律法规、理工科论文、说明书、文科论文、合同、研报等场景，暂时未有计划公开。


## 参考文献
```BibTeX

```

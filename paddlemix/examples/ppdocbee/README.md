# PP-DocBee

## 1. 简介

PP-DocBee 是PaddleMIX团队自研的一款专注于文档理解的多模态大模型，在中文文档理解任务上具有卓越表现。该模型通过近 500 万条文档理解类多模态数据集进行微调优化，各种数据集包括了通用VQA类、OCR类、图表类、text-rich文档类、数学和复杂推理类、合成数据类、纯文本数据等，并设置了不同训练数据配比。在学术界权威的几个英文文档理解评测榜单上，PP-DocBee基本都达到了同参数量级别模型的SOTA。在内部业务中文场景类的指标上，PP-DocBee也高于目前的热门开源和闭源模型。

**本仓库支持的模型权重:**

| Model              |
|--------------------|
| PaddleMIX/PPDocBee-2B-1129 |


## 2. 环境要求
- **python >= 3.10**
- **paddlepaddle-gpu 要求>=3.0.0b2或版本develop**
- **paddlenlp 要求>=3.0.0b2**
```
# paddlepaddle-gpu develop版安装示例
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

# paddlenlp 3.0.0b3安装示例（推荐）
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

### 5.1 英文公开评估集指标

API/Model         | DocVQA-test | ChartQA-test | InfoVQA-test | TextVQA-val | OCRBench 
----------------- | ----------- | ------------ | ------------ | ----------- | -------- 
GPT-4o API        | 92.8        | 85.7         | 79.2       | 77.4       | 73.6    
Gemini-1.5-Pro API| 93.1        | 87.2         | 80.1       | 78.7       | 75.4    
MiniCPM-V-2-2B    | 71.9        | -            |       -      | 74.1       | 60.5    
SmolVLM-Instruct-2B| 81.6       | -            |       -      | 72.7       | -        
Aquila-VL-2B      | 85.0        | 76.5         | 58.3         | 76.4       | 77.2   
Mini-Monkey-2B    | 87.4        | 76.5         | 60.1         | 76.0       | 79.4  
InternVL2-2B      | 86.9        | 76.2         | 58.9         | 73.4       | 78.1  
InternVL2.5-2B    | 88.7        | **79.2**     | 60.9        | 74.3        | 80.4     
Qwen2-VL-2B       | 90.1        | 73.5        | 65.5        | 79.7        | 79.4    
**PPDocBee-2B**   | **90.6**    | 74.6        | **66.2**    | **81.2**  | **82.8**(**83.5**)

> ⚠️注意：
> 1. OCRBench指标归一化到100分制，PPDocBee-2B的OCRBench指标中，82.8是端到端评估的分数，83.5是OCR后处理辅助评估的分数。

### 5.2 内部业务中文场景评估集指标

| API/模型 | 总分 | 印刷文字类 | 表格类 | 印章类 | 图表类 |
|---------|-----:|---------:|------:|------:|------:|
| GPT-4o API | 685 | 436 | 198 | 5 | 46 |
| GLM-4V Flash API | 547 | 339 | 169 | 5 | 34 |
| InternVL2.5-2B | 596 | 363 | 182 | 4 | **47** |
| Qwen2-VL-2B | 680 | 476 | 167 | **8** | 29 |
| **PPDocBee-2B** | **765** | **517** | **202** | 5 | 41 |

印刷文字类 (655张)、表格类 (358张)、印章类 (15张)、图表类 (176张)

> ⚠️注意：
> 1. 内部业务中文场景评测于 2024.12.09日修订，所有图像分辨率 (1680, 1204)，共1196条数据。
> 2. 内部业务中文场景评估集包括了财报、法律法规、理工科论文、说明书、文科论文、合同、研报等场景，暂时未有计划公开。

# PP-DocBee


# PP-DocBee

## 1. 简介

PP-DocBee 是一款专注于文档理解的多模态大模型，在中文文档理解任务上具有卓越表现。该模型基于 'Qwen/Qwen2-VL-2BInstruct' 架构，通过近 500 万条文档理解类多模态数据和精选的纯文本数据进行微调优化。

## 2. 环境要求
- **python >= 3.10**
- **paddlepaddle-gpu 要求版本develop**
```
# 安装示例
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

- **paddlenlp == 3.0.0b2**

> 注：(默认开启flash_attn)使用flash_attn 要求A100/A800显卡或者H20显卡。V100请用float16推理。


## 3. 在线体验

<p align="center">
<video width="80%" height="auto" controls>
  <source src="https://github.com/user-attachments/assets/8e74c364-6d65-4930-b873-6fd5df263d9a" type="video/mp4">
  您的浏览器不支持视频标签
</video>
</p>


我们提供了在线体验环境，您可以通过[AI Studio](https://aistudio.baidu.com/application/detail/60135)快速体验 PP-DocBee 的功能。

## 4. 使用指南

### 4.1 模型推理

下面展示了一个表格识别的示例：

<p align="center">
  <img src="https://github.com/user-attachments/assets/6a03a848-c396-4b2f-a7f3-47ff1441c750" width="50%" alt="示例图片"/>
</p>

```bash
python paddlemix/examples/ppdocbee/single_image_infer.py \
  --model_path "PaddleMIX/PPDocBee-2B" \
  --image_file "your_image_path" \
  --question "识别这份表格的内容"
```

输出示例：
```
名次  国家/地区  金牌  银牌  铜牌  奖牌总数
1  中国（CHN）  48  22  30  100
2  美国（USA）  36  39  37  112
3  俄罗斯（RUS）  24  13  23  60
4  英国（GBR）  19  13  19  51
5  德国（GER）  16  11  14  41
6  澳大利亚（AUS）  14  15  17  46
7  韩国（KOR）  13  11  8  32
8  日本（JPN）  9  8  8  25
9  意大利（ITA）  8  9  10  27
10  法国（FRA）  7  16  20  43
11  荷兰（NED）  7  5  4  16
12  乌克兰（UKR）  7  4  11  22
13  肯尼亚（KEN）  6  4  6  16
14  西班牙（ESP）  5  11  3  19
15  牙买加（JAM）  5  4  2  11
```

## 5. 性能评测

### 5.1 准确率评测


| Benchamrk       | MiniCPM-V 2.0 | InternVL-2B |  SmolVLM  | Qwen2-VL-2B | **PP-DocBee** |
| :-------------: | :--------: | :---------: | :---------: |:---------: | :-----------: |
| Model Size        | 2.43B     |   2.21B       | 2.25B      | 2.21B      | 2.21B          |
| DocVQA-val      | 71.9(test)| 86.9(test)    |  81.6(test)|   89.2     |   **90.1**     |
| ChartQA-test    | -         | **76.2**      |    -       |    73.5       |   74.6         |
| InfoVQA-val     | -         | 58.9(test)    |    -       |    64.1       |   **65.4**     |
| TextVQA-val     | 74.1      | 73.4          |    72.7    |    79.7       |   **81.2**     |
| OCRBench        | 605       | 781           |    -       |    794        |   **828**      |
| ChineseOCRBench | -         | -             |    -        |    76.1       |   **80.2**      |
| 内部中文评估集    | -         | -             |    -        |     52.8       |   **60.3**      |



### 5.2 速度评测



## 引用

如果您在研究中使用了 PP-DocBee，请引用以下论文：

```BibTeX
@article{Qwen2-VL,
  title={Qwen2-VL},
  author={Qwen team},
  year={2024}
}
```
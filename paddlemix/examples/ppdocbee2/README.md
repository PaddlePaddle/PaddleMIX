# PP-DocBee2

## 1. 简介

PP-DocBee2 是PaddleMIX团队自研的一款专注于文档理解的多模态大模型，在PP-DocBee的基础上，我们进一步优化了基础模型，并引入了新的数据优化方案，提高了数据质量，使用自研[数据合成策略](https://arxiv.org/abs/2503.04065)生成的少量的47万数据便使得PP-DocBee2在中文文档理解任务上表现更佳。在内部业务中文场景类的指标上，PP-DocBee2相较于PP-DocBee提升了约11.4%，同时也高于目前的同规模热门开源和闭源模型。

**本仓库支持的模型权重:**

| Model              | 模型大小 | Huggingface 仓库地址 |
|--------------------|----------|--------------------|
| PPDocBee2-3B | 3B | [PPDocBee2-3B](https://huggingface.co/PaddleMIX/PPDocBee2-3B) |

注意：使用`xxx.from_pretrained("PaddleMIX/PPDocBee2-3B")`即可自动下载该权重文件夹到缓存目录。

## 2 环境准备
1）[安装PaddlePaddle](https://github.com/PaddlePaddle/PaddleMIX?tab=readme-ov-file#3-%EF%B8%8F%E5%AE%89%E8%A3%85paddlepaddle)
- **python >= 3.10**
- **paddlepaddle-gpu 要求是>=3.0.0b2或develop版本**
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
python -m pip install -e .
python -m pip install -e ppdiffusers
python -m pip install -r requirements.txt
python -m pip install paddlenlp==3.0.0b3

# sh 脚本快速安装
sh build_env.sh
```

>
注：
* 请确保安装了以上依赖，否则无法运行。
* (默认开启flash_attn)使用flash_attn 要求A100/A800显卡或者H20显卡。V100请用float16推理。

## 3. 在线体验和部署

### 3.1 在线体验

我们提供了在线体验环境，您可以通过[AI Studio](https://aistudio.baidu.com/app/highcode/83545/app)快速体验 PP-DocBee2 的功能。

### 3.2 本地gradio部署
```bash
# 安装gradio
pip install gradio==5.6.0
# 运行gradio
python paddlemix/examples/ppdocbee2/app.py
```

## 4. 模型推理

### 4.1 单卡推理

下面展示了一个表格识别的示例：

<p align="center">
  <img src="https://github.com/user-attachments/assets/6a03a848-c396-4b2f-a7f3-47ff1441c750" width="50%" alt="示例图片"/>
</p>

```bash
python paddlemix/examples/ppdocbee2/ppdocbee2_infer.py \
  --model_path "PaddleMIX/PPDocBee2-3B" \
  --image_file "paddlemix/demo_images/medal_table.png" \
  --question "识别这份表格的内容, 以markdown格式输出"
```

输出示例：
```
| 名次 | 国家/地区 | 金牌 | 银牌 | 铜牌 | 奖牌总数 |
|---|---|---|---|---|---|
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

### 4.2 分布式推理

```bash
sh paddlemix/examples/ppdocbee2/shell/distributed_ppdocbee2_infer.sh
```
> ⚠️注意："mp_degree"需要根据显卡数量"gpus"进行调整，例如2卡推理，则设置为2。

### 4.3 高性能推理

PP-DocBee2 支持高性能推理，具体可参考 [PP-DocBee2高性能推理教程](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/deploy/ppdocbee2)

## 5. 性能评测

### 内部业务中文场景评估集指标

| API/模型 | 总分/1196 | 印刷文字类/656 | 表格类/358 | 印章类/15 | 图表类/167 |
|---------|-----:|---------:|------:|------:|------:|
| GPT-4o API | 685 | 436 | 198 | 5 | 46 |
| GLM-4V Flash API | 547 | 339 | 169 | 5 | 34 |
| InternVL2.5-2B | 596 | 363 | 182 | 4 | **47** |
| Qwen2-VL-2B | 680 | 476 | 167 | **8** | 29 |
| PPDocBee-2B | 765 | 517 | 202 | 5 | 41 |
| Qwen2.5-VL-3B | 789 | 526 | 223 | 6 | 34 |
| **PPDocBee2-3B** | **852** | **545** | **253** | 7 | **47** |


印刷文字类 (656张)、表格类 (358张)、印章类 (15张)、图表类 (167张)

> ⚠️注意：
> 1. 内部业务中文场景评测于 2024.12.09日修订，所有图像分辨率 (1680, 1204)，共1196条数据。
> 2. 内部业务中文场景评估集包括了财报、法律法规、理工科论文、说明书、文科论文、合同、研报等场景，暂时未有计划公开。

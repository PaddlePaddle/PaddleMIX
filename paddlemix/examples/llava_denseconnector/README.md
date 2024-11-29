# DenseConnector

## 1. 模型介绍

[DenseConnector](https://arxiv.org/abs/2405.13800) 旨在通过整合多层视觉特征来增强现有的多语言语言模型（MLLMs）的视觉感知能力。

**本仓库支持的模型权重:**

| Model              |
|--------------------|
| HuanjinYao/DenseConnector-v1.5-7B  |


注意：与huggingface权重同名，但权重为paddle框架的Tensor，使用`xxx.from_pretrained("HuanjinYao/DenseConnector-v1.5-7B")`即可自动下载该权重文件夹到缓存目录。


## 2 环境准备

1）[安装PaddleNLP develop分支](https://github.com/PaddlePaddle/PaddleNLP?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

版本要求：paddlenlp>=3.0.0b2

2）[安装 PaddleMIX 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/b4f97ff859e1964c839fc5fab94f7ba63b1e5959?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

注意：Python版本最好为3.10及以上版本，Python最低版本要求3.8。


## 3 快速开始
完成环境准备后，我们提供多轮对话示例：

### 多轮对话启动
```bash
python paddlemix/examples/llava_denseconnector/run_predict_denseconnector.py \
    --model-path "HuanjinYao/DenseConnector-v1.5-7B" \
    --image-file "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg" \
```
可配置参数说明：
  * `model-path`: 指定llava系列的模型名字或权重路径
  * `image-flie` :输入图片路径或url，默认None。


输入图片：<center><img src="https://github.com/LokeZhou/PaddleMIX/assets/13300429/95f73037-097e-4712-95be-17d5ca489f11" /></center>

```
USER: 描述这张照片
ASSISTANT: 这是一个照片，展示了一辆红色公交车在街道上行驶。车辆正在行驶在一个狭窄的道路上，周围有一些汽车和树木。车辆的前部有一个路灯，并且还有一个路灯在车辆的右侧。
USER: 给出公交车位置的坐标
ASSISTANT: 0.23, 0.33, 0.79, 0.78
```


### 参考文献
```BibTeX
@article{yao2024dense,
  title={Dense Connector for MLLMs},
  author={Yao, Huanjin and Wu, Wenhao and Yang, Taojiannan and Song, YuXin and Zhang, Mengxi and Feng, Haocheng and Sun, Yifan and Li, Zhiheng and Ouyang, Wanli and Wang, Jingdong},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```

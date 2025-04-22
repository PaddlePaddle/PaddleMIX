# Wan2.1视频生成

Wan2.1是阿里集团的开源视频生成模型，支持文本到视频（Text-to-Video）和图像到视频（Image-to-Video）两种模式。本仓库提供了Wan2.1的paddle实现，目前仅支持推理。


## 快速开始
### 环境准备
若曾使用PaddlePaddle主页build_paddle_env.sh脚本安装PaddlePaddle，请根据本身cuda版本手动更新版本[Installation](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)。\
\
更新diffusers:\
在ppdiffusers目录下运行以下命令:
```shell
python install -e .
```
### 目前支持模型
文生视频模型：Wan2.1-T2V：1.3B、14B模型\
图生视频模型：Wan2.1-I2V-14B：480P、720P模型
### 推理示例

#### 硬件要求
* 硬件要求：1.3B模型请保证有 30G以上显存，14B模型请保证有 80G以上显存

#### 文本到视频

```shell
cd Wan2.1
python text2video.py
```
可以通过`text2video.py`文件中的`model_id`参数选择模型，支持以下模型:\
model_id 当前支持: `Wan-AI/Wan2.1-T2V-14B-Diffusers`, `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`\
对应Wan2.1-T2V的14B版本与1.3B版本。

#### 图像到视频

```shell
python image2video.py
```
可以通过`image2video.py`文件中的`model_id`参数选择模型，支持以下模型:\
model_id 当前支持: `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`, `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`\
对应Wan2.1-I2V 14B的480P版本与720P版本。

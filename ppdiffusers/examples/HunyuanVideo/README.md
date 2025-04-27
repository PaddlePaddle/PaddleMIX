# HunyuanVideo视频生成

HunyuanVideo是由腾讯开发的13B参数量的开源视频生成模型，能够生成高质量，高动态的视频。本仓库提供了HunyuanVideo的paddle实现，目前仅支持推理。

## 快速开始
### 环境准备
若曾使用PaddlePaddle主页build_paddle_env.sh脚本安装PaddlePaddle，请根据本身cuda版本手动更新版本[Installation](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)。

更新diffusers：在ppdiffusers目录下运行以下命令:
```shell
python install -e .
```

## 推理示例

### 硬件要求
* 硬件要求：至少要求50G以上显存

### 文本到视频

运行如下脚本，生成视频
```shell
python ppdiffusers/examples/inference/text_to_video_generation-hunyuan_video.py
```
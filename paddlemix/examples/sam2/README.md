# SAM 2: Segment Anything in Images and Videos

## 1. 模型简介

[SAM2](https://ai.meta.com/sam2/) 是 Meta AI Research, FAIR 发布的图像、视频分割模型。将SAM扩展到视频，将图像视为单帧视频。模型设计是一个简单的变压器架构，具有流式存储器，用于实时视频处理。


本仓库提供该模型的Paddle实现，并提供了推理代码。

<p align="center">
  <img src="https://github.com/user-attachments/assets/62626ba4-d81f-4c09-bc79-dc8310eddd5d" align="middle" width = "600" />
</p>

## 2. 快速开始

### 获取权重

```bash
wget https://bj.bcebos.com/v1/paddlenlp/models/community/Sam/Sam2/sam2.1_hiera_large.pdparams
```

### 运行demo

```bash
python paddlemix/examples/sam2/grounded_sam2_tracking_demo.py \
       --sam2_config configs/sam2.1_hiera_l.yaml \
       --sam2_checkpoint sam2.1_hiera_large.pdparams \
       --input_path input.mp4 \
       --output_path output.mp4 \
       --prompt "input your prompt here"
```






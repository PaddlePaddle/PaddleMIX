# PaddleMIX昆仑使用说明

为了满足用户对AI芯片多样化的需求， PaddleMIX 团队基于飞桨框架在硬件兼容性和灵活性方面的优势，深度适配了昆仑P800芯片，为用户提供了国产计算芯片上的训推能力。只需安装说明安装多硬件版本的飞桨框架后，在模型配置文件中添加一个配置设备的参数，即可在相关硬件上使用PaddleMIX。当前PaddleMIX昆仑版适配涵盖了多模态理解模型Qwen2.5-VL(3b版本)。未来我们将继续在用户使用的多种算力平台上适配 PaddleMIX 更多的模型，敬请期待。

## 1. 模型列表
<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>多模态理解</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        </ul>
          <li><b>图文预训练</b></li>
        <ul>
            <li><a href="../../paddlemix/examples/qwen2_5_vl">Qwen2.5-VL</a></li>
      </ul>
      </td>
    </tr>
  </tbody>
</table>

## 2. 安装说明

### 2.1 创建标准化环境

当前 PaddleMIX 支持昆仑 P800 芯片，昆仑驱动版本为 5.0.21.14。考虑到环境差异性，我们推荐使用飞桨官方提供的标准镜像完成环境准备。

参考如下命令启动容器，CUDA_VISIBLE_DEVICES 指定可见的 XPU 卡号

```shell
docker run -it --name paddle-xpu-dev -v $(pwd):/work \
  -v /usr/local/bin/xpu-smi:/usr/local/bin/xpu-smi \
  -w=/work --shm-size=128G --network=host --privileged  \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310 /bin/bash
```

### 2.2 安装飞桨

在容器内安装飞桨

```shell
# 安装飞桨官方 xpu 版本
python -m pip install --pre paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/
```

### 2.3 安装PaddleMIX

克隆PaddleMIX仓库

```shell
git clone https://github.com/PaddlePaddle/PaddleMIX
cd PaddleMIX
```

### 2.4 安装依赖

```shell
sh build_env.sh
python -m pip install -U librosa
```

## 3. 多模态理解

多模态大模型（Multimodal LLM）是当前研究的热点，在 2024 年迎来了井喷式的发展，它将多模态输入经由特定的多模态 encoder 转化为与文本对齐的 token ，随后被输入到大语言模型中来执行多模态任务。PaddleMIX 新增了多模态大模型：Qwen2.5-VL 系列，同时支持指令微调训练和推理，模型能力覆盖了图片问答、文档图表理解、关键信息提取、场景文本理解、 OCR 识别、科学数学问答、视频理解、多图联合理解等。

Qwen2.5-VL系列模型支持昆仑 P800 芯片上训练和推理，使用昆仑 P800 芯片训练推理时请先参考本文安装说明章节中的内容安装相应版本的飞桨框架。Qwen2.5-VL模型训练推理使用方法参考如下:

### 3.1 微调训练

#### 3.1.1 数据准备

参照[文档](../../paddlemix/examples/qwen2_5_vl/README.md)进行数据准备

#### 3.1.2 环境设置

设置XPU相关环境变量

```shell
export FLAGS_use_stride_kernel=0
export FLAGS_allocator_strategy=auto_growth
```
#### 3.1.3 微调训练

执行微调训练，可以从[Qwen2.5-VL模型](../../paddlemix/examples/qwen2_5_vl/README.md)查看详细的参数说明，目前仅支持3b模型的微调训练。

```shell
# 3B (多张40G A卡 显存可运行3B模型)
sh paddlemix/examples/qwen2_5_vl/shell/baseline_3b_bs32_1e8.sh
```

### 3.2 推理

#### 3.2.1 环境设置

参考上述微调训练步骤设置环境变量

#### 3.2.2 执行推理

执行推理，可以从[Qwen2.5-VL模型](../../paddlemix/examples/qwen2_5_vl/README.md)查看详细的参数说明，目前仅支持3b模型的推理。

```shell
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/qwen2_5_vl/single_image_infer.py
```

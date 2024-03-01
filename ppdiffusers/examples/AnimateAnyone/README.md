# Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation

## 1. 模型简介

 Animate Anyone是一项角色动画技术，能将静态图像依据指定动作生成动态的角色视频。该技术利用扩散模型，以保持图像到视频转换中的时间一致性和内容细节。具体实现借鉴于[MooreThreads/Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone/tree/master)。

![](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/595032c0-6f76-49ba-834a-3e92e790ea2f)

注：上图引自 [AnimateAnyone](https://arxiv.org/pdf/2311.17117.pdf)。

## 2. 环境准备

通过 `git clone` 命令拉取 PaddleMIX 源码，并安装必要的依赖库。请确保你的 PaddlePaddle 框架版本在 2.6.0 之后，PaddlePaddle 框架安装可参考 [飞桨官网-安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

```bash
# 克隆 PaddleMIX 仓库
git clone https://github.com/PaddlePaddle/PaddleMIX

# 安装2.6.0版本的paddlepaddle-gpu，当前我们选择了cuda12.0的版本，可以查看 https://www.paddlepaddle.org.cn/ 寻找自己适合的版本
python -m pip install paddlepaddle-gpu==2.6.0.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# 进入AnimateAnyone目录
cd PaddleMIX/ppdiffusers/examples/AnimateAnyone/

# 安装新版本ppdiffusers
pip install https://paddlenlp.bj.bcebos.com/models/community/junnyu/wheels/ppdiffusers-0.24.0-py3-none-any.whl --user

# 安装其他所需的依赖, 如果提示权限不够，请在最后增加 --user 选项
pip install -r requirements.txt
```

## 3. 模型下载

运行以下自动下载脚本，下载 AnimateAnyone 相关模型权重文件，模型权重文件将存储在`./pretrained_weights`下面。

```shell
python scripts/download_weights.py
```

## 4. 模型推理

运行以下推理命令，生成指定宽高和帧数的动画，将存储在`./output`下。

```shell
python -m scripts.pose2vid --config ./configs/inference/animation.yaml -W 512 -H 784 -L 120
```

生成效果如下所示：
<video controls autoplay loop src="https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/4343b522-4449-4db2-be28-fdbbe04f90d4" muted="false"></video>

## 5. 参考资料

- [MooreThreads/Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone/tree/master)
- [AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone)

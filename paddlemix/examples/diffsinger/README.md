## 1. DiffSinger 模型介绍

该模型是 [DiffSinger（由 OpenVPI 维护的版本）](https://github.com/openvpi/DiffSinger) 的 Paddle 实现。

[DiffSinger](https://arxiv.org/abs/2105.02446) 是目前最先进的歌声合成（Singing Voice Synthesis, SVS）模型。OpenVPI 维护的版本对其进行了进一步优化，增加了更多的功能。

本仓库目前仅支持推理功能。


## 2. 环境准备

1） [安装PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

2）[安装 PaddleMix 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/b4f97ff859e1964c839fc5fab94f7ba63b1e5959?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

3）使用 pip 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```

## 4. 快速开始
完成环境准备后，运行以下脚本：

```bash
bash run_predict.sh
```

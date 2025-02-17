
# Magvit2

## 1. 模型简介
<a href="https://arxiv.org/abs/2310.05737">Magvit</a>是一个视频或图像通用的编解码 Tokenizer，可有效降低 token 数量，从而支持视频/图像生成、理解任务。
本仓库是基于 paddle 实现的 magvitv2 模型结构。

## 2. 环境准备

通过 `git clone` 命令拉取 PaddleMIX 源码，并安装必要的依赖库。
```bash
# 克隆 PaddleMIX 仓库
git clone https://github.com/PaddlePaddle/PaddleMIX

# 安装2.5.2版本的paddlepaddle-gpu，当前我们选择了cuda11.7的版本，可以查看 https://www.paddlepaddle.org.cn/ 寻找自己适合的版本
python -m pip install paddlepaddle-gpu==2.5.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# 进入stable diffusion目录
cd PaddleMIX/ppdiffusers/examples/stable_diffusion

# 安装所需的依赖, 如果提示权限不够，请在最后增加 --user 选项
pip install -r requirements.txt

# 安装magevit2 requirments.txt
cd PaddleMIX/ppdiffusers/examples/video_tokenizer/magvit2
pip install -r requirements.txt

```

## 3. 模型推理demo
```bash
cd PaddleMIX/ppdiffusers/examples/video_tokenizer/magvit2

python example.py
```

## 4. 参考资料
```bibtex
@misc{yu2023language,
    title   = {Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation},
    author  = {Lijun Yu and José Lezama and Nitesh B. Gundavarapu and Luca Versari and Kihyuk Sohn and David Minnen and Yong Cheng and Agrim Gupta and Xiuye Gu and Alexander G. Hauptmann and Boqing Gong and Ming-Hsuan Yang and Irfan Essa and David A. Ross and Lu Jiang},
    year    = {2023},
    eprint  = {2310.05737},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

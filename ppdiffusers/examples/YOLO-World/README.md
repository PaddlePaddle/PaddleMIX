# YOLO-World: Real-Time Open-Vocabulary Object Detection

## 1. 模型简介

YOLO-World 是由腾讯AI Lab、ARC Lab、腾讯PCG和华中科技大学合作提出的实时开放词汇目标检测方法
，YOLO-World在大规模视觉语言数据集（包括Objects365、GQA、Flickr30K和CC3M）上进行了预训练，这使得YOLO-World具有强大的zero-shot能力和grounding能力。

![](https://github.com/AILab-CVC/YOLO-World/blob/master/assets/yolo_arch.png)

注：上图引自 [YOLO-World](https://github.com/AILab-CVC/YOLO-World)

## 2. 环境准备

通过 `git clone` 命令拉取 PaddleMIX 源码，并安装必要的依赖库。请确保你的 PaddlePaddle 框架版本在 2.6.0 之后，PaddlePaddle 框架安装可参考 [飞桨官网-安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

```bash
# 克隆 PaddleMIX 仓库
git clone https://github.com/PaddlePaddle/PaddleMIX

# 安装2.6.0版本的paddlepaddle-gpu，当前我们选择了cuda12.0的版本，可以查看 https://www.paddlepaddle.org.cn/ 寻找自己适合的版本
python -m pip install paddlepaddle-gpu==2.6.0.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# 进入目录
cd PaddleMIX/ppdiffusers/examples/YOLO-World/

# 安装新版本ppdiffusers
pip install https://paddlenlp.bj.bcebos.com/models/community/junnyu/wheels/ppdiffusers-0.24.0-py3-none-any.whl --user

# 安装其他所需的依赖
pip install -e .
```
`YOLO-World/` 目录下的目录结构应如下：

```bash
.
├── configs
├── infer.py
├── pretrain
├── pyproject.toml
├── README.md
├── third_party
└── yolo_world
```


## 3. 下载模型
通过 [Huggingface](https://huggingface.co/wondervictor/YOLO-World/) 下载YOLO-World 的模型权重文件。可以将权重文件放在 `pretrain/` 目录下，下载完成后执行torch2paddle.py脚本将权重转化为paddle格式。

```bash
python pretrain/torch2paddle.py <path_to_weight.pth> -p <path_to_weight.pdparams>
```

## 4. 模型推理

```bash
python infer.py \
    --config <path_to_config_yaml> \
    -o weights=<path_to_pretrained_weights> \
    --image=<path_to_image_or_directory> \
    --text <text or .txt file path>\
    --topk=<number_of_top_predictions> \
    --threshold=<confidence_threshold_value> \
    --output_dir=<output_directory_path>

# 可以在cli参数中添加 --offline 选项，将会首先使用text model得到 offline text feats，之后的推理中text model将不会被加载。如：

python infer.py \
    --config <path_to_config_yaml> \
    -o weights=<path_to_pretrained_weights> \
    --image=<path_to_image_or_directory> \
    --text <text or .txt file path>\
    --topk=<number_of_top_predictions> \
    --threshold=<confidence_threshold_value> \
    --output_dir=<output_directory_path> \
    --offline
```

## 5. 参考资料

- [GitHub - AILab-CVC/YOLO-World: YOLO-WOrld](https://github.com/AILab-CVC/YOLO-World)

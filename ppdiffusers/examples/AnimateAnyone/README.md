# Animate Anyone: 角色动画视频生成模型训练与推理

## 1. 模型简介

 Animate Anyone是一项角色动画视频生成技术，能将静态图像依据指定动作生成动态的角色动画视频。该技术利用扩散模型，以保持图像到视频转换中的时间一致性和内容细节。训练由两阶段组成，对不同组网成分进行微调。具体实现借鉴于[MooreThreads/Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone/tree/master)。

![](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/595032c0-6f76-49ba-834a-3e92e790ea2f)

注：上图引自 [AnimateAnyone](https://arxiv.org/pdf/2311.17117.pdf)。

## 2. 环境准备

通过 `git clone` 命令拉取 PaddleMIX 源码，并安装ppdiffusers以及必要的依赖库。请确保你的 PaddlePaddle 框架版本在 2.6.0 之后，PaddlePaddle 框架安装可参考 [飞桨官网-安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

```bash
# 克隆 PaddleMIX 仓库
git clone https://github.com/PaddlePaddle/PaddleMIX

# 安装2.6.0版本的paddlepaddle-gpu，当前我们选择了cuda12.0的版本，可以查看 https://www.paddlepaddle.org.cn/ 寻找自己适合的版本
python -m pip install paddlepaddle-gpu==2.6.0.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# 进入ppdiffusers目录
cd PaddleMIX/ppdiffusers

# 安装ppdiffusers，若提示权限不够，请在最后增加 --user 选项
pip install -e .

# 进入AnimateAnyone目录
cd examples/AnimateAnyone/

# 安装其他所需的依赖, 若提示权限不够，请在最后增加 --user 选项
pip install -r requirements.txt
```

## 3. 模型下载

运行以下自动下载脚本，下载 AnimateAnyone 推理以及训练初始化模型权重文件，模型权重文件将存储在`./pretrained_weights`下。

```shell
python scripts/download_weights.py
```

## 4. 两阶段训练
### 4.1 数据准备
训练数据由[ubc_fashion](https://vision.cs.ubc.ca/datasets/fashion/)和bili_dance两个数据集组成，其中ubc_fashion包含598组数据，bili_dance包含2451组数据，数据获取方式如下：
```bash
# ubc_fashion数据集下载
wget https://bj.bcebos.com/paddlenlp/models/community/tsaiyue/ubcNbili_data/ubcNbili_data.tar.gz

# 文件解压
tar -xzvf ubcNbili_data.tar.gz
```
该数据集由三部分组成，分别为元数据、原始视频以及对应动作视频，其中元数据记录对应原始视频和动作视频的路径，动作视频提取方式参考自[MooreThreads/Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone/tree)，训练数据文件结构如下：
```bash
├── ubcNbili_data  # 训练数据根目录
    ├── meta_data # 元数据文件夹
        ├── ubcNbili_meta.json
    ├── video # 原始视频文件夹
        ├── 00001.mp4
        ├── 00002.mp4
        ├── ...
        ├── 03049.mp4
    ├── video_dwpose # 动作视频文件夹
        ├── 00001.mp4
        ├── 00002.mp4
        ├── ...
        ├── 03049.mp4
```
### 4.2 第一阶段训练
第一阶段由于训练参数规模较大无法在单卡 NVIDIA V100 32G GPU 或 NVIDIA A100 40G GPU 上运行，可在单机多卡下开启显存优化分组切片技术 `--sharding` 进行训练，训练命令如下：
```shell
ppdiffusers_path=PaddleMIX/ppdiffusers
export PYTHONPATH=$ppdiffusers_path:$PYTHONPATH
python -u -m paddle.distributed.launch --gpus "0,1,2,3" scripts/trainer_stage1.py \
    --do_train \
    --output_dir ./exp_output/stage1 \
    --save_strategy 'no' \
    --save_total_limit 5 \
    --save_steps 800 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.0e-5 \
    --weight_decay 1.0e-2 \
    --max_steps 30000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 1 \
    --seed 42 \
    --report_to all \
    --sharding "stage2"
```
训练脚本基于 paddlenlp.trainer 实现，支持单卡、多卡训练，可通过 `--gpus` 指定训练使用的GPU卡号，在多卡环境上支持分组切片技术以降低显存占用。训练过程中的阶段性权重以及可视化训练监控文件将存储于 `exp_output/stage1` 目录下。训练流程相关参数详见 [paddlenlp.trainer](https://github.com/PaddlePaddle/PaddleNLP/blob/a5f69e4543a5371ceb28106b7aa2ea93208620b9/paddlenlp/trainer/training_args.py)，模型与数据相关参数详见 `src/trainer/args_stage1.py`。
### 4.3 第二阶段训练
第二阶段训练支持单卡 NVIDIA V100 32G GPU 的硬件环境，训练命令如下：
```shell
ppdiffusers_path=PaddleMIX/ppdiffusers
export PYTHONPATH=$ppdiffusers_path:$PYTHONPATH
export GLOG_minloglevel=2
python -u -m paddle.distributed.launch --gpus "0" scripts/trainer_stage2.py \
    --do_train \
    --output_dir ./exp_output/stage2 \
    --save_strategy 'no' \
    --save_total_limit 5 \
    --save_steps 800 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.0e-5 \
    --weight_decay 1.0e-2 \
    --max_steps 30000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 1 \
    --seed 42 \
    --report_to all
```
该训练脚本同样基于paddlenlp.trainer实现，支持单卡、多卡训练，可通过 `--gpus` 指定训练使用的GPU卡号。训练过程中的阶段性权重以及可视化训练监控文件将存储于 `exp_output/stage2` 目录下。训练流程相关参数详见 [paddlenlp.trainer](https://github.com/PaddlePaddle/PaddleNLP/blob/a5f69e4543a5371ceb28106b7aa2ea93208620b9/paddlenlp/trainer/training_args.py)，模型与数据相关参数详见 `src/trainer/args_stage2.py`。

### 4.4 第二阶段微调前后对比
在第二阶段训练中，利用 [animatediff初始化权重](https://huggingface.co/guoyww/animatediff)对模型组网中的motion_modules进行微调，微调前后生成效果对比如下：
| Static Image | Pose Video | Before Fine-tuning | After Fine-tuning |
|--------------|------------|---------------------|-------------------|
| <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/f6c5d27b-0183-4ae5-ad6b-3e36125cb515" width="512" height="668"> | <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/abe4931d-81ca-453b-b061-510a48b62b02" width="512" height="668"> | <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/33ddb6ac-d07c-40a2-9d97-7cba9ebea88d" width="512" height="668"> | <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/8b4ba74c-5a3f-45c3-be0f-645e0ece6bcd" width="552" height="668"> |
| <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/fa0df880-d891-4a99-8272-86405f38a03f" width="512" height="668"> | <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/015640bf-9309-4a88-b1ff-7e63ab04f0b8" width="512" height="668"> | <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/86f34d9f-73a8-4a4c-9945-04d1f322d5d3" width="512" height="668"> | <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/104eb7d1-b9eb-453a-bb0b-27f2fe02f6c6" width="552" height="668"> |

## 5. 模型推理

模型可在NVIDIA V100 32G GPU下进行推理。运行以下推理命令，生成指定宽高和帧数的动画，将存储在 `./output` 下。

```shell
python -m scripts.pose2vid --config ./configs/inference/animation.yaml -W 600 -H 784 -L 120
```

生成效果如下所示：
| Static Image | Pose Video | Animation Video |
|--------------|------------|---------------------|
| <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/c55a0449-b0f2-4137-9ed0-354bd3c57936" width="512" height="668"> | <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/f856e8c4-824c-4403-8fb2-6cdf12eacea2" width="512" height="668"> | <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/23e2e55e-f505-425f-920f-cde7e04bebbe" width="552" height="668"> |
| <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/bf3ceacc-ad32-41ea-9f2c-1fb91abb2afe" width="512" height="668"> | <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/5eec36a8-7ce8-4299-b524-0c45f115bc0c" width="512" height="668"> | <img src="https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/0c7e7088-58f5-476f-8d37-bf5bb768f56c" width="552" height="668"> |

## 5. 参考资料

- [MooreThreads/Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone/tree/master)
- [AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone)
- [ubc_fashion dataset](https://vision.cs.ubc.ca/datasets/fashion/)

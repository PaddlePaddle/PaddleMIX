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

# 安装2.6.1版本的paddlepaddle-gpu，当前我们选择了cuda12.0的版本，可以查看 https://www.paddlepaddle.org.cn/ 寻找自己适合的版本
python -m pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

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
    --save_strategy 'steps' \
    --save_total_limit 2 \
    --save_steps 2000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.0e-5 \
    --weight_decay 1.0e-2 \
    --max_steps 30000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 1 \
    --seed 42 \
    --report_to all \
    --sharding "stage1" \
    --fp16 True \
    --fp16_opt_level O2
```
训练脚本基于 paddlenlp.trainer 实现，支持单卡、多卡训练，可通过 `--gpus` 指定训练使用的GPU卡号，在多卡环境上支持分组切片技术以降低显存占用。训练过程中的阶段性权重以及可视化训练监控文件将存储于 `exp_output/stage1` 目录下。训练流程相关参数详见 [paddlenlp.trainer](https://github.com/PaddlePaddle/PaddleNLP/blob/a5f69e4543a5371ceb28106b7aa2ea93208620b9/paddlenlp/trainer/training_args.py)，模型与数据相关参数详见 `src/trainer/args_stage1.py`。
### 4.3 第二阶段训练
第二阶段训练支持单卡 NVIDIA V100 32G GPU 的硬件环境，训练命令如下：
```shell
ppdiffusers_path=PaddleMIX/ppdiffusers
export PYTHONPATH=$ppdiffusers_path:$PYTHONPATH
python -u -m paddle.distributed.launch --gpus "0" scripts/trainer_stage2.py \
    --do_train \
    --output_dir ./exp_output/stage2 \
    --save_strategy 'steps' \
    --save_total_limit 2 \
    --save_steps 2000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.0e-5 \
    --weight_decay 1.0e-2 \
    --max_steps 30000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 1 \
    --seed 42 \
    --report_to all \
    --fp16 True \
    --fp16_opt_level O2 \
    --train_width 256 \
    --train_height 512
```
该训练脚本同样基于paddlenlp.trainer实现，支持单卡、多卡训练，可通过 `--gpus` 指定训练使用的GPU卡号。训练过程中的阶段性权重以及可视化训练监控文件将存储于 `exp_output/stage2` 目录下。训练流程相关参数详见 [paddlenlp.trainer](https://github.com/PaddlePaddle/PaddleNLP/blob/a5f69e4543a5371ceb28106b7aa2ea93208620b9/paddlenlp/trainer/training_args.py)，模型与数据相关参数详见 `src/trainer/args_stage2.py`。

**___Note: 可根据具体算力情况适当调整生成视频分辨率相关参数 `--train_width` 和 `--train_width`，以获得更好的训练效果。___**

### 4.4 第二阶段微调前后对比
在第二阶段训练中，利用 [animatediff初始化权重](https://huggingface.co/guoyww/animatediff)对模型组网中的motion_modules进行微调，微调前后生成效果对比如下：

| Static Image | Pose Video | Before Fine-tuning | After Fine-tuning |
|--------------|------------|--------------------|-------------------|
| ![demo](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/07a5f6cd-db53-4c69-a469-fda9edbff3f3) | ![demo](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/5442ff20-9aab-4f28-adca-711c7cd46ff9) | ![demo](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/d1f6942f-2075-4e24-b7e1-645c7a9f2c86) | ![demo](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/a2470660-3757-474b-b414-117416f1314c) |
| ![demo](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/5958967d-57ce-4501-8a15-860879e08541) | ![demo](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/6e4ca44d-5d62-49a6-ae2f-bf87e0ca29b2) | ![demo](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/b3644e24-ec5e-43e4-b44d-7d5b4e6ca2c3) | ![demo](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/dd4aa5d5-6217-49ba-984f-1ceb05ca4495) |

## 5. 模型推理

模型可在NVIDIA V100 32G GPU下进行推理。运行以下推理命令，生成指定宽高和帧数的动画，将存储在 `./output` 下。

```shell
python -m scripts.pose2vid --config ./configs/inference/animation.yaml -W 600 -H 784 -L 120
```

生成效果如下所示：

| Static Image | Pose Video | Before Fine-tuning |
|--------------|------------|--------------------|
| ![demo](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/a81e2c42-09c6-4a0b-8f0b-b7df1d77779a) | ![demo](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/973a6629-f24a-4420-b4af-7653e8ff8e92) | ![demo](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/ce2e2cd2-8ba2-46dd-bb6b-99726cd80e97) |
| ![demo](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/abb8da73-951b-41a1-b922-8095ca84b988) | ![demo](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/b1d5efa8-76e0-4d4b-a878-4c3625b65b3d) | ![demo](https://github.com/PaddlePaddle/PaddleMIX/assets/46399096/68c1a0ef-6958-4a66-92b6-6d52717354f0)|

## 5. 参考资料

- [MooreThreads/Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone/tree/master)
- [AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone)
- [ubc_fashion dataset](https://vision.cs.ubc.ca/datasets/fashion/)

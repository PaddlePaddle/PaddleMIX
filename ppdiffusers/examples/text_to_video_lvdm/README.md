## Lataent Video Diffusion Model模型训练

本教程介绍 **LVDM(Lataent Video Diffusion Model)** 的训练，这里的训练仅针对扩散模型(UNet)部分，而不涉及一阶段的模型的训练。


## 准备工作
### 安装依赖

在运行这个训练代码前，我们需要安装ppdiffusers以及相关依赖。


```bash
cd PaddleMIX/ppdiffusers
python setup.py install
pip install -r requirements.txt
```

### 数据准备
准备扩散模型训练的数据，格式需要适配`VideoFrameDataset`或`WebVidDataset`。数据集相关的配置请参考`lvdm/lvdm_args_short.py`或`lvdm/lvdm_args_text2video.py`中的`DatasetArguments`。相关数据下载链接为[Sky Timelapse](https://github.com/weixiong-ur/mdgan)、[Webvid](https://github.com/m-bain/webvid)。


### 预训练模型准备
由于一个完整的PPDiffusers Pipeline包含多个预训练模型，而我们这里仅针对扩散模型(UNet)部分进行训练，所以还需要准备好其他预训练模型参数才能够正常训练和推理，包括Txet-Encoder、VAE。此外，开发者如果不想从头开始训练而是在现有模型上微调，也可准备好UNet模型参数并基于此进行微调，提供的预训练模型权重如下：
- westfish/lvdm_short_sky
- westfish/lvdm_short_sky_no_ema
- westfish/lvdm_text2video_orig_webvid_2m

## 模型训练
模型训练时的参数配置请参考`lvdm/lvdm_args_short.py`或`lvdm/lvdm_args_text2video.py`，其包含、`ModelArguments`、`DatasetArguments`、`TrainerArguments`，分别表示预训练模型及对齐相关的参数，数据集相关的参数，Trainer相关的参数。开发者可以使用默认参数进行训练，也可以根据需要修改参数。


### 单机单卡训练
```bash
python -u train_lvdm_short.py
```
```bash
python -u train_lvdm_text2video.py
```

### 单机多卡训练
```bash
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_lvdm_short.py
```
```bash
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_lvdm_text2video.py
```

训练时可通过如下命令通过浏览器观察训练过程：
```
visualdl --logdir your_log_dir/runs --host 0.0.0.0 --port 8042
```

## 参考
https://github.com/YingqingHe/LVDM

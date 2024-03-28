# PaddleInfer ControlNet 模型高性能部署

 **目录**
   * [部署模型准备](#部署模型准备)
   * [环境依赖](#环境依赖)
   * [快速体验](#快速体验)
       * [ControlNet文图生成（ControlNet-Text-to-Image Generation）](#ControlNet文图生成)
       * [ControlNet文本引导的图像变换（ControlNet-Image-to-Image Text-Guided Generation）](#ControlNet文本引导的图像变换)
       * [ControlNet文本引导的图像编辑（ControlNet-Text-Guided Image Inpainting）](#ControlNet文本引导的图像编辑)

⚡️[PaddleInfer] 是一款全场景、易用灵活、极致高效的AI推理部署工具，为开发者提供多硬件、多推理引擎后端的部署能力。开发者只需调用一行代码即可随意切换硬件、推理引擎后端。本示例展现如何通过 PaddleInfer 将我们 PPDiffusers 训练好的 Stable Diffusion 模型进行多硬件、多推理引擎后端高性能部署。

<a name="部署模型准备"></a>

## 部署模型准备

本示例需要使用训练模型导出后的部署模型，可参考[模型导出文档](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/deploy/controlnet/export.md)导出部署模型。

<a name="环境依赖"></a>

## 环境依赖

在示例中使用了 PaddleInfer，需要执行以下命令安装依赖。

```shell
# 当前仅2.6.0+的develop版本支持PaddleInfer，请使用下面的特定版本：
python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuSome-LinuxCentos-Gcc82-Cuda117-Cudnn84-Trt84-Py310-Compile/eea10b17b24b80dcad2a6c955ad6cc1925adaa0b/paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl
```

<a name="快速体验"></a>

## 快速体验

### 静态图模型导出 (static model export)
这里以canny Controlnet模型为例，导出静态图模型。
```
export USE_PPXFORMERS=False
export FLAGS_set_to_1d=1
python export_model.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --controlnet_pretrained_model_name_or_path  lllyasviel/sd-controlnet-canny --output_path static_model/stable-diffusion-v1-5-canny
```
导出模型在static_model/stable-diffusion-v1-5-canny目录下。

### 基于静态图的推理 (Paddle后端)
```
python infer.py --model_dir static_model/stable-diffusion-v1-5-canny/ --scheduler "euler" --backend paddle --device gpu --task_name all --width 512 --height 512 --inference_steps 30 --tune False --use_fp16 True
```

### 基于静态图的推理 (TensorRT后端，建议在A100上使用)
要使用TensorRT后端，需要先进行tune，并生成TensorRT所需的模型动态shape信息。(设置--tune True)
```
python infer.py --model_dir static_model/stable-diffusion-v1-5-canny/ --scheduler "euler" --backend paddle --device gpu --task_name all --width 512 --height 512 --inference_steps 30 --tune True --use_fp16 False
```
在tune完成后，可以执行以下命令进行推理。(第一次执行时，需要等待TensorRT模型编译完成，需要较长时间，后续执行时，直接加载编译好的模型进行推理)
```
python infer.py --model_dir static_model/stable-diffusion-v1-5-canny/ --scheduler "euler" --backend paddle_tensorrt --device gpu --task_name all --width 512 --height 512 --inference_steps 50 --use_fp16 True
```
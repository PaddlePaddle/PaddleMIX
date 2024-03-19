# PaddleInfer Stable Diffusion系列模型高性能部署

 **目录**
   * [环境依赖](#环境依赖)
   * [快速体验](#快速体验)

⚡️[PaddleInfer]是一款全场景、易用灵活、极致高效的AI推理部署工具，为开发者提供多硬件、多推理引擎后端的部署能力。开发者只需调用一行代码即可随意切换硬件、推理引擎后端。本示例展现如何通过 PaddleInfer 将我们 PPDiffusers 训练好的 Stable Diffusion系列模型进行多硬件、多推理引擎后端高性能部署。

<a name="环境依赖"></a>

## 环境依赖

在示例中使用了 PaddleInfer，需要执行以下命令安装依赖。

```shell
# 当前仅2.6.0+的develop版本支持PaddleInfer，请使用下面的特定版本：
python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuSome-LinuxCentos-Gcc82-Cuda117-Cudnn84-Trt84-Py310-Compile/eea10b17b24b80dcad2a6c955ad6cc1925adaa0b/paddlepaddle_gpu-0.0.0-cp310-cp310-linux_x86_64.whl
```

<a name="快速体验"></a>

## 快速体验
当前deploy目录下提供了PaddleInfer的Stable Diffusion系列模型高性能部署示例，包含controlnet、sd15、sdxl、svd、ipadapter-sd15、ipadapter-svd等。
在每个模型的示例目录下，提供了模型的静态图导出代码，以及基于导出后的静态图进行推理的示例代码。具体可以参考每个模型目录下的sciripts文件夹内脚本文件。  
这里以sd15为例，展示如何基于导出的静态图进行推理:  
参考：ppdiffusers/deploy/sd15/scripts/tune_and_tensorrt.sh

### 静态图模型导出 (static model export)
```
export USE_PPXFORMERS=False
export FLAGS_set_to_1d=1
python export_model.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --output_path static_model/stable-diffusion-v1-5
```
导出模型在static_model/stable-diffusion-v1-5目录下。

### 基于静态图的推理 (Paddle后端)
```
python infer.py --model_dir static_model/stable-diffusion-v1-5/ --scheduler "euler" --backend paddle --device gpu --task_name all --width 512 --height 512 --inference_steps 30 --tune False --use_fp16 True
```

### 基于静态图的推理 (TensorRT后端，建议在A100上使用)
要使用TensorRT后端，需要先进行tune，并生成TensorRT所需的模型动态shape信息。(设置--tune True)
```
python infer.py --model_dir static_model/stable-diffusion-v1-5/ --scheduler "euler" --backend paddle --device gpu --task_name all --width 512 --height 512 --inference_steps 30 --tune True --use_fp16 False
```
在tune完成后，可以执行以下命令进行推理。(第一次执行时，需要等待TensorRT模型编译完成，需要较长时间，后续执行时，直接加载编译好的模型进行推理)
```
python infer.py --model_dir static_model/stable-diffusion-v1-5/ --scheduler "euler" --backend paddle_tensorrt --device gpu --task_name all --width 512 --height 512 --inference_steps 50
```
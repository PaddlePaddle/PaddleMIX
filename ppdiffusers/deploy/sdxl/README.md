# PaddleInfer Stable Diffusion XL 模型高性能部署

 **目录**
   * [环境依赖](#环境依赖)
   * [快速体验](#快速体验)
       * [文图生成（Text-to-Image Generation）](#文图生成)
       * [文本引导的图像变换（Image-to-Image Text-Guided Generation）](#文本引导的图像变换)
       * [文本引导的图像编辑（Text-Guided Image Inpainting）](#文本引导的图像编辑)

⚡️[PaddleInfer]是一款全场景、易用灵活、极致高效的AI推理部署工具，为开发者提供多硬件、多推理引擎后端的部署能力。开发者只需调用一行代码即可随意切换硬件、推理引擎后端。本示例展现如何通过 PaddleInfer 将我们 PPDiffusers 训练好的 Stable Diffusion XL模型进行多硬件、多推理引擎后端高性能部署。

<a name="环境依赖"></a>

## 环境依赖

在示例中使用了 PaddleInfer，需要执行以下命令安装依赖。

```shell
python -m pip install paddlepaddle-gpu==2.6.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

<a name="快速体验"></a>

## 快速体验
### 静态图模型导出 (static model export)
```
export USE_PPXFORMERS=False
export FLAGS_set_to_1d=1
python export_model.py --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 --output_path static_model/stable-diffusion-xl-base-1.0
```
导出模型在static_model/stable-diffusion-xl-base-1.0目录下。

### 基于静态图的推理 (Paddle后端)
```
python infer.py --model_dir static_model/stable-diffusion-xl-base-1.0 --scheduler "euler" --backend paddle --device gpu --task_name all --width 512 --height 512 --inference_steps 30 --tune False --use_fp16 True
```

### 基于静态图的推理 (TensorRT后端，建议在A100上使用)
要使用TensorRT后端，需要先进行tune，并生成TensorRT所需的模型动态shape信息。(设置--tune True)
```
python infer.py --model_dir static_model/stable-diffusion-xl-base-1.0 --scheduler "euler" --backend paddle --device gpu --task_name all --width 512 --height 512 --inference_steps 30 --tune True --use_fp16 False
```
在tune完成后，可以执行以下命令进行推理。(第一次执行时，需要等待TensorRT模型编译完成，需要较长时间，后续执行时，直接加载编译好的模型进行推理)
```
python infer.py --model_dir static_model/stable-diffusion-xl-base-1.0 --scheduler "euler" --backend paddle_tensorrt --device gpu --task_name all --width 512 --height 512 --inference_steps 50
```
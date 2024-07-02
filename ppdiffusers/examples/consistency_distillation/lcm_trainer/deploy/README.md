# Latent Consistency Models 高性能推理 （当前仅支持SD15）

 **目录**
- [LCM 模型高性能部署](#lcm-模型高性能部署)
  - [部署模型准备](#部署模型准备)
  - [环境依赖](#环境依赖)
  - [快速体验](#快速体验)
    - [文图生成（Text-to-Image Generation）](#文图生成text-to-image-generation)
    - [文本引导的图像变换（Image-to-Image Text-Guided Generation）](#文本引导的图像变换image-to-image-text-guided-generation)
    - [文本引导的图像编辑（Text-Guided Image Inpainting）](#文本引导的图像编辑text-guided-image-inpainting)
    - [参数说明](#参数说明)
  - [LCM Gradio demo 体验](#lcm-gradio-demo-体验)

⚡️[FastDeploy](https://github.com/PaddlePaddle/FastDeploy) 是一款全场景、易用灵活、极致高效的AI推理部署工具，为开发者提供多硬件、多推理引擎后端的部署能力。开发者只需调用一行代码即可随意切换硬件、推理引擎后端。本示例展现如何通过 FastDeploy 将我们 PPDiffusers 训练好的 Stable Diffusion 模型进行多硬件、多推理引擎后端高性能部署。

<a name="部署模型准备"></a>

## 部署模型准备

本示例需要使用训练模型导出后的部署模型，可参考[模型导出文档](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/examples/consistency_distillation/lcm_trainer/deploy/export.md)导出部署模型。
<a name="环境依赖"></a>

## 环境依赖

- paddlepaddle >= 2.5.2
- fastdeploy-gpu-python == 1.0.7

在示例中使用了 FastDeploy，需要执行以下命令安装依赖。

```bash
pip install fastdeploy-gpu-python==1.0.7 -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
pip install ppdiffusers==0.19.4
```

<a name="快速体验"></a>

## 快速体验

我们经过部署模型准备，即可开始测试。本目录提供了 StableDiffusion 1-5 模型支持的三种任务，分别是文图生成、文本引导的图像变换以及文本引导的图像编辑。

<a name="文图生成"></a>

### 文图生成（Text-to-Image Generation）


下面将指定模型目录，推理引擎后端，硬件以及 scheduler 类型，运行 `infer.py` 脚本，完成文图生成任务。

```sh
python infer.py --model_dir lcm-stable-diffusion-v1-5/ --backend paddle --task_name text2img --inference_steps 4
```

脚本的输入提示语句为 **"a photo of an astronaut riding a horse on mars"**， 得到的图像文件为 text2img.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

![text2img.png](https://github.com/PaddlePaddle/PaddleMIX/assets/50394665/e5207482-d5a7-4bc0-83ba-20cea97e87bc)


```sh
python infer.py --model_dir lcm-stable-diffusion-v1-5/ --backend paddle_tensorrt --use_fp16 True --device gpu --task_name text2img --inference_steps 4
```

<a name="文本引导的图像变换"></a>

### 文本引导的图像变换（Image-to-Image Text-Guided Generation）

下面将指定模型目录，推理引擎后端，硬件以及 scheduler 类型，运行 `infer.py` 脚本，完成文本引导的图像变换任务。

```sh
python infer.py --model_dir lcm-stable-diffusion-v1-5/ --backend paddle_tensorrt --use_fp16 True --device gpu --task_name img2img --inference_steps 4
```

脚本输入的提示语句为 **"A fantasy landscape, trending on artstation"**，运行得到的图像文件为 img2img.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

|       input image       |       output image       |
|:-------------------:|:-------------------:|
|![][sketch-mountains-input]|![][fantasy_landscape]|

[sketch-mountains-input]: https://user-images.githubusercontent.com/10826371/217207485-09ee54de-4ba2-4cff-9d6c-fd426d4c1831.png
[fantasy_landscape]: https://github.com/PaddlePaddle/PaddleMIX/assets/50394665/cdbc1732-e282-466d-b43c-23b01d6592a1


<a name="文本引导的图像编辑"></a>

### 文本引导的图像编辑（Text-Guided Image Inpainting）

下面将指定模型目录，推理引擎后端，硬件，运行 `infer.py` 脚本，完成文本引导的图像编辑任务。

```sh
python infer.py --model_dir lcm-stable-diffusion-v1-5/ --backend paddle_tensorrt --use_fp16 True --device gpu --task_name inpaint_legacy --inference_steps 4
```

脚本输入的提示语为 **"Face of a yellow cat, high resolution, sitting on a park bench"**，运行得到的图像文件为 inpaint_legacy.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

|       input image       |       mask image       |       output image
|:-------------------:|:-------------------:|:-------------------:|
|![][input]|![][mask]|![][output]|

[input]: https://user-images.githubusercontent.com/10826371/217423470-b2a3f8ac-618b-41ee-93e2-121bddc9fd36.png
[mask]: https://user-images.githubusercontent.com/10826371/217424068-99d0a97d-dbc3-4126-b80c-6409d2fd7ebc.png
[output]: https://github.com/PaddlePaddle/PaddleMIX/assets/50394665/9ddbc138-956f-481f-bec1-49502b5d80c8

## LCM Gradio demo 体验


我们经过部署模型准备，可以开始进行 demo 测试（这里需要你有至少15G显存的机器）
下面将演示如何指定模型目录、任务参数，运行 `gradio_demo.py` ，完成各项图像变换任务。

```sh
python gradio_demo.py --model_dir lcm-stable-diffusion-v1-5/ --backend paddle_tensorrt --use_fp16 True --device gpu
```

运行上述命令后，将启动一个 Web 服务器，在浏览器中打开 http://127.0.0.1:8654 即可体验 LCM 的 Gradio Demo。

![image](https://github.com/PaddlePaddle/PaddleMIX/assets/50394665/4f7cc0d0-e94c-4852-abc5-c72bfa53bb29)


`gradio_demo.py` 除了以上示例的命令行参数，还支持更多命令行参数的设置。展开可查看各命令行参数的说明。

| 参数 | 参数说明 |
| --- | --- |
| --model_dir | LCM 模型的目录，默认为 "lcm-stable-diffusion-v1-5/"。 |
| --backend | 推理运行时的后端，可选值为 "paddle", "paddle_tensorrt"。默认为 "paddle"。 |
| --use_fp16 | 是否使用 FP16 模式，默认为 True。 |
| --use_bf16 | 是否使用 BF16 模式，默认为 False。 |
| --device_id | 选择的 GPU ID，默认为 0。 |

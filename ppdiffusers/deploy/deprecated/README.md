# FastDeploy Stable Diffusion 模型高性能部署

 **目录**
- [FastDeploy Stable Diffusion 模型高性能部署](#fastdeploy-stable-diffusion-模型高性能部署)
  - [部署模型准备](#部署模型准备)
  - [环境依赖](#环境依赖)
  - [快速体验](#快速体验)
    - [文图生成（Text-to-Image Generation）](#文图生成text-to-image-generation)
    - [文本引导的图像变换（Image-to-Image Text-Guided Generation）](#文本引导的图像变换image-to-image-text-guided-generation)
    - [文本引导的图像编辑（Text-Guided Image Inpainting）](#文本引导的图像编辑text-guided-image-inpainting)
      - [Legacy 版本](#legacy-版本)
      - [正式版本](#正式版本)
      - [参数说明](#参数说明)
- [Stable Diffusion Gradio demo 部署](#stable-diffusion-gradio-demo-部署)
  - [模型准备](#模型准备)
  - [环境依赖](#环境依赖-1)
  - [快速体验](#快速体验-1)
    - [文生图(Text-to-Image Generation)、图生图(Image-to-Image Generation)、文本引导的图像编辑 legacy（Text-Guided Image Inpainting）](#文生图text-to-image-generation图生图image-to-image-generation文本引导的图像编辑-legacytext-guided-image-inpainting)
      - [文生图(Text-to-Image Generation)](#文生图text-to-image-generation)
      - [图生图(Image-to-Image Generation)](#图生图image-to-image-generation)
      - [文本引导的图像编辑 legacy（Text-Guided Image Inpainting）](#文本引导的图像编辑-legacytext-guided-image-inpainting)
    - [文本引导的图像编辑（Text-Guided Image Inpainting）](#文本引导的图像编辑text-guided-image-inpainting-1)
    - [条件引导的图像生成（Canny ControlNet）](#条件引导的图像生成canny-controlnet)
    - [参数说明](#参数说明-1)

⚡️[FastDeploy](https://github.com/PaddlePaddle/FastDeploy)是一款全场景、易用灵活、极致高效的AI推理部署工具，为开发者提供多硬件、多推理引擎后端的部署能力。开发者只需调用一行代码即可随意切换硬件、推理引擎后端。本示例展现如何通过 FastDeploy 将我们 PPDiffusers 训练好的 Stable Diffusion 模型进行多硬件、多推理引擎后端高性能部署。

<a name="部署模型准备"></a>

## 部署模型准备

本示例需要使用训练模型导出后的部署模型，可参考[模型导出文档](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/deploy/export.md)导出部署模型。

<a name="环境依赖"></a>

## 环境依赖

在示例中使用了 FastDeploy，需要执行以下命令安装依赖。

```shell
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

<a name="快速体验"></a>

## 快速体验

我们经过部署模型准备，可以开始进行测试。本目录提供 StableDiffusion 模型支持的三种任务，分别是文图生成、文本引导的图像变换以及文本引导的图像编辑。

<a name="文图生成"></a>

### 文图生成（Text-to-Image Generation）


下面将指定模型目录，推理引擎后端，硬件以及 scheduler 类型，运行 `infer.py` 脚本，完成文图生成任务。

```sh
python infer.py --model_dir stable-diffusion-v1-4/ --scheduler "pndm" --backend paddle --task_name text2img
```

脚本的输入提示语句为 **"a photo of an astronaut riding a horse on mars"**， 得到的图像文件为 text2img.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

![text2img.png](https://user-images.githubusercontent.com/10826371/200261112-68e53389-e0a0-42d1-8c3a-f35faa6627d7.png)

如果使用 stable-diffusion-v1-5 模型，则可执行以下命令完成推理：

```sh
python infer.py --model_dir stable-diffusion-v1-5/ --scheduler "preconfig-euler-ancestral" --backend paddle_tensorrt --use_fp16 True --device gpu --task_name text2img
```

同时，我们还提供基于两阶段 HiresFix 的文图生成示例。下面将指定模型目录，指定任务名称为 `hiresfix` 后，运行 `infer.py` 脚本，完成`两阶段hiresfix任务`，在第一阶段我们生成了 `512x512分辨率` 的图片，然后在第二阶段我们在一阶段的基础上修复生成了 `768x768分辨率` 图片。

|       without hiresfix       |       with hiresfix       |
|:-------------------:|:-------------------:|
|![][without-hiresfix]|![][with-hiresfix]|

[without-hiresfix]: https://github.com/PaddlePaddle/PaddleNLP/assets/50394665/38ab6032-b960-4b76-8d69-0e0f8b5e1f42
[with-hiresfix]: https://github.com/PaddlePaddle/PaddleNLP/assets/50394665/a472cb31-d8a2-451d-bf80-cd84c9ef0d08

在80G A100上，ppdiffusers==0.16.1、fastdeploy==1.0.7、develop paddle、cuda11.7 的环境下，我们测出了如下的速度。
- without hiresfix 的速度为：Mean latency: 1.930896 s, p50 latency: 1.932413 s, p90 latency: 1.933565 s, p95 latency: 1.933630 s.
- with hiresfix 的速度为：Mean latency: 1.442178 s, p50 latency: 1.442885 s, p90 latency: 1.446133 s, p95 latency: 1.446285 s.

```sh
python infer.py --model_dir stable-diffusion-v1-5/ --scheduler "euler-ancestral" --backend paddle_tensorrt --use_fp16 True --device gpu --task_name hiresfix
```

<a name="文本引导的图像变换"></a>

### 文本引导的图像变换（Image-to-Image Text-Guided Generation）

下面将指定模型目录，推理引擎后端，硬件以及 scheduler 类型，运行 `infer.py` 脚本，完成文本引导的图像变换任务。

```sh
python infer.py --model_dir stable-diffusion-v1-4/ --scheduler "pndm" --backend paddle_tensorrt --use_fp16 True --device gpu --task_name img2img
```

脚本输入的提示语句为 **"A fantasy landscape, trending on artstation"**，运行得到的图像文件为 img2img.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

|       input image       |       output image       |
|:-------------------:|:-------------------:|
|![][sketch-mountains-input]|![][fantasy_landscape]|

[sketch-mountains-input]: https://user-images.githubusercontent.com/10826371/217207485-09ee54de-4ba2-4cff-9d6c-fd426d4c1831.png
[fantasy_landscape]: https://user-images.githubusercontent.com/10826371/217200795-811a8c73-9fb3-4445-b363-b445c7ee52cd.png



如果使用 stable-diffusion-v1-5 模型，则可执行以下命令完成推理：

```sh
python infer.py --model_dir stable-diffusion-v1-5/ --scheduler "euler-ancestral" --backend paddle_tensorrt --use_fp16 True --device gpu --task_name img2img
```


同时，我们还提供基于 CycleDiffusion 的文本引导的图像变换示例。下面将指定模型目录，运行 `infer.py` 脚本，完成文本引导的图像变换任务。

```sh
python infer.py --model_dir stable-diffusion-v1-4/ --backend paddle_tensorrt --use_fp16 True --device gpu --task_name cycle_diffusion
```

脚本输入的源提示语句为 **"An astronaut riding a horse"**，目标提示语句为 **"An astronaut riding an elephant"**，运行得到的图像文件为 cycle_diffusion.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

|       input image       |       output image       |
|:-------------------:|:-------------------:|
|![][horse]|![][elephant]|

[horse]: https://raw.githubusercontent.com/ChenWu98/cycle-diffusion/main/data/dalle2/An%20astronaut%20riding%20a%20horse.png
[elephant]: https://user-images.githubusercontent.com/10826371/223315865-4490b586-1de7-4616-a245-9c008c3ffb6b.png

<a name="文本引导的图像编辑"></a>

### 文本引导的图像编辑（Text-Guided Image Inpainting）

注意！当前有两种版本的图像编辑代码，一个是 Legacy 版本，一个是正式版本，下面将分别介绍两种版本的使用示例。

#### Legacy 版本

下面将指定模型目录，推理引擎后端，硬件以及 scheduler 类型，运行 `infer.py` 脚本，完成文本引导的图像编辑任务。

```sh
python infer.py --model_dir stable-diffusion-v1-4/ --scheduler euler-ancestral --backend paddle_tensorrt --use_fp16 True --device gpu --task_name inpaint_legacy
```

脚本输入的提示语为 **"Face of a yellow cat, high resolution, sitting on a park bench"**，运行得到的图像文件为 inpaint_legacy.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

|       input image       |       mask image       |       output image
|:-------------------:|:-------------------:|:-------------------:|
|![][input]|![][mask]|![][output]|

[input]: https://user-images.githubusercontent.com/10826371/217423470-b2a3f8ac-618b-41ee-93e2-121bddc9fd36.png
[mask]: https://user-images.githubusercontent.com/10826371/217424068-99d0a97d-dbc3-4126-b80c-6409d2fd7ebc.png
[output]: https://user-images.githubusercontent.com/10826371/217455594-187aa99c-b321-4535-aca0-9159ad658a97.png

如果使用 stable-diffusion-v1-5 模型，则可执行以下命令完成推理：

```sh
python infer.py --model_dir stable-diffusion-v1-5/ --scheduler euler-ancestral --backend paddle_tensorrt --use_fp16 True --device gpu --task_name inpaint_legacy
```

#### 正式版本

下面将指定模型目录，推理引擎后端，硬件以及 scheduler 类型，运行 `infer.py` 脚本，完成文本引导的图像编辑任务。

```sh
python infer.py --model_dir stable-diffusion-v1-5-inpainting/ --scheduler euler-ancestral --backend paddle_tensorrt --use_fp16 True --device gpu --task_name inpaint
```

脚本输入的提示语为 **"Face of a yellow cat, high resolution, sitting on a park bench"**，运行得到的图像文件为 inpaint.png。生成的图片示例如下（每次生成的图片都不相同，示例仅作参考）：

|       input image       |       mask image       |       output image
|:-------------------:|:-------------------:|:-------------------:|
|![][input_2]|![][mask_2]|![][output_2]|

[input_2]: https://user-images.githubusercontent.com/10826371/217423470-b2a3f8ac-618b-41ee-93e2-121bddc9fd36.png
[mask_2]: https://user-images.githubusercontent.com/10826371/217424068-99d0a97d-dbc3-4126-b80c-6409d2fd7ebc.png
[output_2]: https://user-images.githubusercontent.com/10826371/217454490-7d6c6a89-fde6-4393-af8e-05e84961b354.png

#### 参数说明

`infer.py` 除了以上示例的命令行参数，还支持更多命令行参数的设置。展开可查看各命令行参数的说明。


| 参数 |参数说明 |
|----------|--------------|
| --model_dir | 导出后模型的目录。默认为 `runwayml/stable-diffusion-v1-5@fastdeploy` |
| --backend | 推理引擎后端。默认为 `paddle_tensorrt`，可选列表：`['onnx_runtime', 'paddle', 'paddlelite', 'paddle_tensorrt', 'tensorrt']`。 |
| --device | 运行设备。默认为 `gpu`，可选列表：`['cpu', 'gpu', 'huawei_ascend_npu', 'kunlunxin_xpu']`。 |
| --device_id | `gpu` 设备的 id。若 `device_id` 为`-1`，视为使用 `cpu` 推理。 |
| --inference_steps | `UNet` 模型运行的次数，默认为 `50`。 |
| --benchmark_steps | `Benchmark` 运行的次数，默认为 `1`。 |
| --use_fp16 | 是否使用 `fp16` 精度。默认为 `False`。使用 `paddle_tensorrt` 后端及 `kunlunxin_xpu` 设备时可以设为 `True` 开启。 |
| --task_name | 任务类型，默认为`text2img`，可选列表：`['text2img', 'img2img', 'inpaint', 'inpaint_legacy', 'cycle_diffusion', 'hiresfix', 'all']`。 注意，当`task_name`为`inpaint`时候，我们需要配合`runwayml/stable-diffusion-inpainting@fastdeploy`权重才能正常使用。|
| --scheduler | 采样器类型。默认为 `'preconfig-euler-ancestral'`。可选列表：`['pndm', 'lms', 'euler', 'euler-ancestral', 'preconfig-euler-ancestral', 'dpm-multi', 'dpm-single', 'unipc-multi', 'ddim', 'ddpm', 'deis-multi', 'heun', 'kdpm2-ancestral', 'kdpm2']`。|
| --infer_op | 推理所采用的op，可选列表 `['zero_copy_infer', 'raw', 'all']`，`zero_copy_infer`推理速度更快，默认值为`zero_copy_infer`。 |
| --parse_prompt_type | 处理prompt文本所使用的方法，可选列表 `['raw', 'lpw']`，`lpw`可强调句子中的单词，并且支持更长的文本输入，默认值为`lpw`。 |
| --width | 生成图片的宽度，取值范围 512~768。默认值为 512。|
| --height | 生成图片的高度，取值范围 512~768。默认值为 512。|
| --hr_resize_width | hiresfix 所要生成的宽度，取值范围 512~768。默认值为 768。|
| --hr_resize_height | hiresfix 所要生成的高度，取值范围 512~768。默认值为 768。|
| --is_sd2_0 | 是否为sd2.0的模型？默认为 False 。|

# Stable Diffusion Gradio demo 部署

## 模型准备

本示例需要使用训练模型导出后的部署模型，可参考[模型导出文档](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/deploy/export.md)导出部署模型，或者直接下载已导出的部署模型：

```sh
wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/fd_trt_model/runwayml_stable-diffusion-v1-5_fd.tar.gz && tar -zxvf runwayml_stable-diffusion-v1-5_fd.tar.gz

wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/fd_trt_model/stable-diffusion-v1-5-inpainting.tar.gz && tar -zxvf stable-diffusion-v1-5-inpainting.tar.gz

wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/fd_trt_model/control_sd15_canny.tar.gz && tar -zxvf control_sd15_canny.tar.gz
```

其中`runwayml_stable-diffusion-v1-5_fd`模型用于文生图、图生图与文本引导的图像编辑(legacy version)任务，`stable-diffusion-v1-5-inpainting` 模型用于 inpaint 任务，`control_sd15_canny` 模型用于 canny controlnet 任务。

## 环境依赖

在示例中使用了 FastDeploy 以及 ppdiffusers，需要执行以下命令安装推荐的 Paddle 环境依赖。

```sh
python -m pip install paddlepaddle-gpu==2.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html

pip install https://paddlenlp.bj.bcebos.com/models/community/junnyu/fd_trt_model/ppdiffusers-0.19.4-py3-none-any.whl
```

## 快速体验

我们经过部署模型准备，可以开始进行 demo 测试（这里需要你有至少15G显存的机器）。本 demo 提供 StableDiffusion 模型支持的三种任务，分别是文本引导与图像引导的图片生成、文字引导的图像编辑（正式版与 legacy 版），条件引导的图像生成（Canny ControlNet）。

下面将演示如何指定模型目录、任务参数，运行 `gradio_demo.py` ，完成各项图像变换任务。

我们可以通过后缀参数指定任务和运行模型地址，也可以环境变量设定，例子如下：

```sh
# 通过环境变量指定模型任务
export model_dir='./runwayml_stable-diffusion-v1-5_fd'
export task_name='text2img_img2img_inpaint_legacy'
python gradio_demo.py
```

```sh
# 通过后缀参数指定模型与任务
python gradio_demo.py --model_dir='./runwayml_stable-diffusion-v1-5_fd' --task_name='text2img_img2img_inpaint_legacy'
```

请注意，没有指定 `--backend paddle_tensorrt` 的时候默认运行的是 paddle 后端，如果你想使用 tensorrt 加速请显式指定对应参数。

### 文生图(Text-to-Image Generation)、图生图(Image-to-Image Generation)、文本引导的图像编辑 legacy（Text-Guided Image Inpainting）

准备好模型后，通过下列命令我们可以启动基本任务：

```sh
python gradio_demo.py --model_dir='./runwayml_stable-diffusion-v1-5_fd' --task_name='text2img_img2img_inpaint_legacy'
```

#### 文生图(Text-to-Image Generation)

将 `Tab` 切换到 `text2img`
输入正向提示语句为 `a photo of an astronaut riding a horse on mars`，`seed` 为 2345, 生成的图片示例如下:

![image](https://github.com/PaddlePaddle/PaddleMIX/assets/96160062/81663d76-3693-4c07-991f-25bcc260441c)


#### 图生图(Image-to-Image Generation)

将 `Tab` 切换到 `img2img`
基于输入图像，输入正向提示语句为`dog`，`seed` 为 123456, 生成的图片示例如下:

|       input image       |       output image      |
|:-------------------:|:-------------------:|
|![][cat]|![][dog1]|

[cat]: https://github.com/PaddlePaddle/PaddleMIX/assets/96160062/ac7de478-b987-4f99-9192-6e89d5cdcd55

[dog1]: https://github.com/PaddlePaddle/PaddleMIX/assets/96160062/3e4c6882-9738-4b50-9c3f-dc9655b74d04



#### 文本引导的图像编辑 legacy（Text-Guided Image Inpainting）

将 `Tab` 切换到 `inpaint_legacy`
基于输入图像，输入正向提示语句为`dog`，生成的图片示例如下:

|       input image       |       output image      |
|:-------------------:|:-------------------:|
|![][cat]|![][dog2]|

[cat]: https://github.com/PaddlePaddle/PaddleMIX/assets/96160062/ac7de478-b987-4f99-9192-6e89d5cdcd55

[dog2]: https://github.com/PaddlePaddle/PaddleMIX/assets/96160062/3dab5db8-c05d-4f7c-9a0b-46fac7bd5f67

### 文本引导的图像编辑（Text-Guided Image Inpainting）


准备好模型后，通过下列命令我们可以启动文本引导的图像编辑任务：

```sh
python gradio_demo.py --model_dir='./stable-diffusion-v1-5-inpainting' --task_name='inpaint'
```

基于输入图像，输入正向提示语句为`dog`，生成的图片示例如下:


|       input image       |       output image      |
|:-------------------:|:-------------------:|
|![][cat]|![][dog3]|

[cat]: https://github.com/PaddlePaddle/PaddleMIX/assets/96160062/ac7de478-b987-4f99-9192-6e89d5cdcd55

[dog3]: https://github.com/PaddlePaddle/PaddleMIX/assets/96160062/81459565-ff80-4910-a3ad-6a7278b50526


### 条件引导的图像生成（Canny ControlNet）


准备好模型后，通过下列命令我们可以启动条件引导的图像生成任务：

```sh
python gradio_demo.py --model_dir='./control_sd15_canny' --task_name='controlnet_canny'
```

基于输入图像，输入正向提示语句为`dog`，传入图片后会自动提取 mask，设此时的 `seed` 为 2345, `conditioning_scale` 为0.8,生成的图片示例如下:


|       input image       |       output image      |
|:-------------------:|:-------------------:|
|![][cat]|![][dog4]|

[cat]: https://github.com/PaddlePaddle/PaddleMIX/assets/96160062/ac7de478-b987-4f99-9192-6e89d5cdcd55

[dog4]: https://github.com/PaddlePaddle/PaddleMIX/assets/96160062/da81b589-7e61-4615-b592-8961e335d001


### 参数说明

`gradio_demo.py` 除了以上示例的命令行参数，还支持更多命令行参数的设置。展开可查看各命令行参数的说明。

| 参数 | 参数说明 |
| --- | --- |
| --model_dir | Diffusion 模型的目录，默认为 "stable-diffusion-v1-5"。 |
| --task_name | 任务名称，可选值为 "text2img_img2img_inpaint_legacy", "inpaint", "controlnet_canny"。默认为 "text2img_img2img_inpaint_legacy"。 |
| --backend | 推理运行时的后端，可选值为 "paddle", "paddle_tensorrt"。默认为 "paddle"。 |
| --use_fp16 | 是否使用 FP16 模式，默认为 True。 |
| --use_bf16 | 是否使用 BF16 模式，默认为 False。 |
| --device_id | 选择的 GPU ID，默认为 0。 |
| --parse_prompt_type | 解析提示类型，可选值为 "raw", "lpw"。默认为 "lpw"。 |
| --is_sd2_0 | 是否为 sd2_0 模型，默认为 False。 |

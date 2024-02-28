# **PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding**

## 1. 模型简介

[PhotoMaker](https://huggingface.co/papers/2312.04461) 是腾讯ARC实验室、南开大学和东京大学推出的一款创新的AI人物生成和照片风格化的开源模型，它具备快速生成人物照片和艺术画像的功能。该模型的核心技术是堆叠ID嵌入（Stacked ID Embedding），这一技术能够将多张照片的信息融合到一个统一的数据结构中。通过保留单一ID的特征并整合不同ID的特征，为用户提供更为丰富和多样的创作选择。PhotoMaker可以定制多种风格的人物照片、改变人物的年龄和性别，以及整合不同人物的特征来创造全新的人物形象。

![](https://camo.githubusercontent.com/c004ae7f537e0fc3a13da99577b79a4f3e354412d1af5c07ee54d51961f9e572/68747470733a2f2f63646e2d75706c6f6164732e68756767696e67666163652e636f2f70726f64756374696f6e2f75706c6f6164732f3632383561393133336162363634323137393135383934342f4259425a4e79666d4e346a424b427878743475787a2e6a706567)

注：该图引自 [TencentARC/PhotoMaker](https://github.com/TencentARC/PhotoMaker)

## 2. 环境准备

PhotoMaker 要求以下环境和依赖项：

- Python >= 3.8

- CUDA >= 11.2

- PaddlePaddle-gpu >= 2.6.0

- PaddleNLP >= 2.7.2

- PPDiffusers >= 0.24.0

- Gradio >= 4.0.0 (运行可视化界面需要安装)

通过 `git clone` 命令拉取 PaddleMIX 源码，并安装必要的依赖库。请确保你的 PaddlePaddle 框架版本在 2.6.0 之后，PaddlePaddle 框架安装可参考 [飞桨官网-安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

```bash
# 克隆 PaddleMIX 仓库
git clone https://github.com/PaddlePaddle/PaddleMIX

# 安装PaddleNLP和PPDiffusers
pip install https://paddlenlp.bj.bcebos.com/models/community/junnyu/wheels/ppdiffusers-0.24.0-py3-none-any.whl
pip install paddlenlp==2.7.2

# 如果运行可视化界面，则需要安装gradio
pip install gradio==4.0.0

# 进入项目主目录
cd /PaddleMIX/ppdiffusers/examples/PhotoMaker/
```

## 3. 硬件要求

本项目需要在GPU硬件平台上运行，需要显存 ≥12GB（推荐16GB及以上），如果本地机器不符合要求，建议前往 [AI Studio](https://aistudio.baidu.com/index) 运行。

## 4. 模型推理

- 加载模型

```python
import os
os.environ["USE_PEFT_BACKEND"] = "True"
# ignore warning
os.environ["GLOG_minloglevel"] = "2"

import paddle
from ppdiffusers.utils import load_image
from ppdiffusers import EulerDiscreteScheduler
from photomaker import PhotoMakerStableDiffusionXLPipeline

base_model_path = "SG161222/RealVisXL_V3.0"
photomaker_path = "TencentARC/PhotoMaker"
photomaker_ckpt = "photomaker-v1.bin"

### Load base model
pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    base_model_path,  # can change to any base model based on SDXL
    paddle_dtype=paddle.float16,
    use_safetensors=True,
    variant="fp16",
    low_cpu_mem_usage=True
)

### Load PhotoMaker checkpoint
pipe.load_photomaker_adapter(
    photomaker_path,
    weight_name=photomaker_ckpt,
    from_hf_hub=True,
    from_diffusers=True,
    trigger_word="img"  # define the trigger word
)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

pipe.fuse_lora()
```

- 输入特征ID图片

```python
### define the input ID images
input_folder_name = './examples/newton_man'
image_basename_list = os.listdir(input_folder_name)
image_path_list = sorted([os.path.join(input_folder_name, basename) for basename in image_basename_list])

input_id_images = []
for image_path in image_path_list:
    input_id_images.append(load_image(image_path))
```

![](https://ai-studio-static-online.cdn.bcebos.com/2e47931848b94121bf5edb02794a3b9a82c75a29c7444e329a7b455763eb3a31)

- 生成新图像

```python
# Note that the trigger word `img` must follow the class word for personalization
prompt = "a photo of a man,golden dragon,new year,red style,festive background img, pixar-style, studio anime, Disney, high-quality"
negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth, grayscale"
generator = paddle.Generator().manual_seed(5959596333)
gen_images = pipe(
    prompt=prompt,
    input_id_images=input_id_images,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    num_inference_steps=50,
    start_merge_step=10,
    generator=generator,
).images[0]
gen_images.save('out_photomaker.png')
```

![](https://ai-studio-static-online.cdn.bcebos.com/7530d54f2b654963b7526ff448f182935e1c527ffb724502bce9f467f88d3b9c)

## 参考资料

- [GitHub - TencentARC/PhotoMaker: PhotoMaker](https://github.com/TencentARC/PhotoMaker)

# InstantID: Zero-shot Identity-Preserving Generation in Seconds

## 1. 模型简介

InstantID 是由InstantX团队、小红书和北京大学推出的一种SOTA的tuning-free方法，只需单个图像即可实现 ID 保留生成，并支持各种下游任务。

![](https://github.com/InstantID/InstantID/raw/main/assets/applications.png)

注：上图引自 [InstantID](https://instantid.github.io/)

## 2. 环境准备

通过 `git clone` 命令拉取 PaddleMIX 源码，并安装必要的依赖库。请确保你的 PaddlePaddle 框架版本在 2.6.0 之后，PaddlePaddle 框架安装可参考 [飞桨官网-安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

```bash
# 克隆 PaddleMIX 仓库
git clone https://github.com/PaddlePaddle/PaddleMIX

# 安装2.6.0版本的paddlepaddle-gpu，当前我们选择了cuda12.0的版本，可以查看 https://www.paddlepaddle.org.cn/ 寻找自己适合的版本
python -m pip install paddlepaddle-gpu==2.6.0.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# 进入consistency_distillation目录
cd PaddleMIX/ppdiffusers/examples/InstantID/

# 安装新版本ppdiffusers
pip install https://paddlenlp.bj.bcebos.com/models/community/junnyu/wheels/ppdiffusers-0.24.0-py3-none-any.whl --user

# 安装其他所需的依赖, 如果提示权限不够，请在最后增加 --user 选项
pip install -r requirements.txt
```

## 3. 下载模型

通过 [Huggingface](https://huggingface.co/InstantX/InstantID) 下载 InstantID 的模型权重文件，你可以通过Python执行以下代码：

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json", local_dir="./checkpoints")
hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/diffusion_pytorch_model.safetensors", local_dir="./checkpoints")
hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="./checkpoints")
hf_hub_download(repo_id="latent-consistency/lcm-lora-sdxl", filename="pytorch_lora_weights.safetensors", local_dir="./checkpoints")
```

此外，本项目面部特征编码器使用了 [insightface](https://github.com/deepinsight/insightface/) ，权重模型需要前往 [antelopev2.zip](https://drive.google.com/file/u/0/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view?usp=sharing&pli=1) 下载并放到 `models/antelopev2` 目录下。当所有的模型权重下载完成后，`InstantID/` 目录下的目录结构应如下所示：

```bash
  .
  ├── models
  ├── examples
  ├── gradio_demo
  ├── checkpoints
  ├── predict.py
  ├── pipeline_stable_diffusion_xl_instantid.py
  ├── resampler.py
  ├── requirements.txt
  └── README.md
```

## 4. 模型推理

### 基础推理

```python
import paddle
import cv2
import os
os.environ["USE_PEFT_BACKEND"] = "True"
import numpy as np
from PIL import Image
from ppdiffusers import ControlNetModel, AutoencoderKL
from ppdiffusers.utils import load_image
from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
face_adapter = f'./InstantID/checkpoints/ip-adapter.bin'
controlnet_path = f'./InstantID/checkpoints/ControlNetModel'
controlnet = ControlNetModel.from_pretrained(controlnet_path,
                                             paddle_dtype=paddle.float16,
                                             use_safetensors=True,
                                             from_hf_hub=True,
                                             from_diffusers=True)

base_model_path = "wangqixun/YamerMIX_v8"

vae = AutoencoderKL.from_pretrained(base_model_path, from_diffusers=True, from_hf_hub=True, subfolder="vae")
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(base_model_path,
                                                          controlnet=controlnet,
                                                          paddle_dtype=paddle.float16,
                                                          from_diffusers=True,
                                                          from_hf_hub=True,
                                                          low_cpu_mem_usage=True)
pipe.vae = vae
pipe.load_ip_adapter_instantid(face_adapter,
                               weight_name=os.path.basename("face_adapter"),
                               from_diffusers=True)
```

然后，输入人脸图像和风格的 Prompts

```python
# load an image
face_image = load_image('./examples/yann-lecun_resize.jpg')

# prepare face emb
face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * x['bbox'][3] - x['bbox'][1])[-1]
face_emb = face_info['embedding']
face_kps = draw_kps(face_image, face_info['kps'])

# prompt
prompt = (
    "watercolor painting, Red festive, Family reunion to celebrate, Plane design, Chinese Dragon in background,"
    "New Year'sDay, Festival celebration, Chinese cultural theme style, soft tones, warm palettes, vibrantillustrations,"
    "Color mural, Minimalism, beautiful, painterly, detailed, textural, artistic"
)
n_prompt = (
    "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, anime, photorealistic, 35mm film, deformed,"
    "glitch, low contrast, noisy"
)

# generate image
generator = paddle.Generator().manual_seed(42)
image = pipe(prompt=prompt,
             negative_prompt=n_prompt,
             image_embeds=face_emb,
             image=face_kps,
             controlnet_conditioning_scale=0.8,
             ip_adapter_scale=0.8,
             num_inference_steps=30,
             generator=generator,
             guidance_scale=5).images[0]

image.save('result.jpg')
```

图像生成效果如下所示： ![](https://ai-studio-static-online.cdn.bcebos.com/34a10e8fc74c4255a6808443d1051b1caea13a4b4e10470f811ab451a4e1fa41)

### 使用 LCM-LoRA 加速

InstantID 兼容 [LCM-LoRA](https://github.com/luosiallen/latent-consistency-model) 方法，只需下载对应的模型到 `checkpoints` 目录下。

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="latent-consistency/lcm-lora-sdxl", filename="pytorch_lora_weights.safetensors", local_dir="./checkpoints")
```

使用 LCM-LoRA 加速时， `num_inference_steps` 参数可以使用比较小的值（如 10） ，以及 `guidance_scale` 建议设置范围是 [0, 1]。

```python
import os
os.environ["USE_PEFT_BACKEND"] = "True"
from ppdiffusers import LCMScheduler

lora_state_dict = './checkpoints/pytorch_lora_weights.safetensors'

pipe.scheduler=LCMScheduler.from_pretrained(base_model_path,
                                            subfolder="scheduler",
                                            from_hf_hub=True,
                                            from_diffusers=True)
pipe.load_lora_weights(lora_state_dict, from_diffusers=True)
pipe.fuse_lora()

num_inference_steps = 10
guidance_scale = 0
```

## 5. 参考资料

[InstantID/InstantID: InstantID : Zero-shot Identity-Preserving Generation in Seconds 🔥](https://github.com/InstantID/InstantID/)

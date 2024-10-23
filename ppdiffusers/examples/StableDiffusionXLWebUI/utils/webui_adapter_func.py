# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 标准库
# import json
import math
import os

import numpy as np

# 第三方库
import paddle
from PIL import Image

from ppdiffusers import (  # UNet2DConditionModel,
    AutoencoderKL,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from ppdiffusers.transformers import CLIPVisionModelWithProjection
from ppdiffusers.utils import load_image, smart_load

# 自定义函数
from .baidufanyi import multi_tasks_translate
from .check_image import custom_save_image
from .ip_adapter import IPAdapterPlusXL, StableDiffusionXLCustomPipeline
from .ip_adapter.utils import swich_state

# import sys


HOME_DIR = os.path.expanduser("~")
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
STATIC_DIR = os.path.join(PROJECT_DIR, "assets")


class LoadTypesModel:
    def __init__(
        self,
        pipe="SG161222/RealVisXL_V3.0",
        unet_dir="SG161222/RealVisXL_V3.0/unet",
        vae_dir="madebyollin/sdxl-vae-fp16-fix",
        outputs="out_puts",
    ):
        # 初始状态
        self.pipe = pipe
        self.type_to_img = ""
        self.scheduler_type = ""
        self.adapter = ""
        self.unet_dir = unet_dir
        self.vae_dir = vae_dir
        self.enable_xformers = ""

        self.image_encoder_path = [
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        ][-2]

        ip_ckpt = [
            os.path.join(HOME_DIR, "data", "data264879", "ip-adapter-plus-face_sdxl_vit-h.bin"),
            os.path.join(PROJECT_DIR, "ip-adapter", "ip-adapter-plus-face_sdxl_vit-h.safetensors"),
        ]
        self.ip_ckpt = ip_ckpt[0] if os.path.exists(ip_ckpt[0]) else ip_ckpt[-1]

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.image_encoder_path,
            paddle_dtype=paddle.float16,
        )

        # load unet
        # self.unet = UNet2DConditionModel.from_pretrained(
        #     unet_dir,
        #     paddle_dtype=paddle.float16,
        #     variant="fp16",
        #     low_cpu_mem_usage=True,
        #     # use_safetensors=True,
        # )

        # load vae
        self.vae = AutoencoderKL.from_pretrained(vae_dir, paddle_dtype=paddle.float16)

        paddle.device.cuda.empty_cache()

        # load base model
        # 根据权重格式选择加载方式
        is_fp16_model = True
        try:
            for _, _, fns in os.walk(pipe):
                is_fp16_model = False
                # 查找文件名中是否包含.fp16.
                for fn in fns:
                    if ".fp16." in fn:
                        is_fp16_model = True
                        print("Varient fp16")
                        break
                break
                print("Varient NOT fp16")
        except Exception as e:
            print(e)
            is_fp16_model = True

        if is_fp16_model:
            self.model = StableDiffusionXLPipeline.from_pretrained(
                pipe,
                paddle_dtype=paddle.float16,
                vae=self.vae,
                use_safetensors=True,
                variant="fp16",
                safety_checker=None,
                low_cpu_mem_usage=True,
                # unet=self.unet,
            )
        else:
            self.model = StableDiffusionXLPipeline.from_pretrained(
                pipe,
                paddle_dtype=paddle.float16,
                vae=self.vae,
                safety_checker=None,
                low_cpu_mem_usage=True,
                # unet=self.unet,
            )

        # get face ip-adapter weight
        if ".safetensors" in self.ip_ckpt:
            f = smart_load(self.ip_ckpt)
            self.state_dict = {"image_proj": {}, "ip_adapter": {}}
            for key in list(f.keys()):
                if key.startswith("image_proj."):
                    self.state_dict["image_proj"][key.replace("image_proj.", "")] = f.pop(key).cast(paddle.float32)
                if key.startswith("ip_adapter."):
                    self.state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.pop(key).cast(paddle.float16)
            self.state_dict["image_proj"] = swich_state(self.state_dict["image_proj"], dtype="float32")
            self.state_dict["ip_adapter"] = swich_state(self.state_dict["ip_adapter"], dtype="float16")
        if ".bin" in self.ip_ckpt:
            self.state_dict = smart_load(self.ip_ckpt)
            self.state_dict["image_proj"] = swich_state(self.state_dict["image_proj"], dtype="float32")
            self.state_dict["ip_adapter"] = swich_state(self.state_dict["ip_adapter"], dtype="float16")

        self.outputs = os.path.join(HOME_DIR, outputs)
        # 创建图片路径
        os.makedirs(self.outputs, exist_ok=True)

    # 图片生成
    def mix2img(
        self,
        pipe,
        prompt,
        negative_prompt,
        guidance_scale=5,
        height=1024,
        width=1024,
        num_inference_steps=50,
        num_images_per_prompt=1,
        scheduler_type="ddpm",
        seed=-1,
        type_to_img="text",
        image=None,
        enable_xformers="ON",
        mask_image=None,
        strength=0.999999,
        adapter="ON",
        face_image=None,  # crop and resize image
        control_guidance_start=0.1,  # 控制姿势，值越小越接近原图0~1
        control_guidance_end=1.0,
        scale=0.75,
        max_size=1800,
        **kwargs
    ):
        # 图片配置参数
        metadata = {
            "pipe": pipe,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "scheduler_type": scheduler_type,
            "type_to_img": type_to_img,
            "adapter": adapter,
            # "width": None or int(width),
            # "height": None or int(height),
            # "strength": strength,
            # "seed": int(seeds),
            # "control_guidance_start": control_guidance_start,
            # "control_guidance_end": control_guidance_end,
            # "scale": scale,
        }

        if type_to_img != "text":
            metadata["strength"] = strength

        if adapter == "ON":
            metadata["adapter_params"] = {}
            metadata["adapter_params"]["scale"] = scale
            metadata["adapter_params"]["control_guidance_start"] = control_guidance_start
            metadata["adapter_params"]["control_guidance_end"] = control_guidance_end

        # 只要任意一个变量变化都会触发重新加载
        if (
            self.pipe != pipe
            or self.type_to_img != type_to_img
            or self.scheduler_type != scheduler_type
            or self.adapter != adapter
            or self.enable_xformers != enable_xformers
        ):

            # # 切换模型时释放内存
            # paddle.device.cuda.empty_cache()
            # 加载model
            model = self.load_types_model(pipe, type_to_img, enable_xformers, scheduler_type, adapter=adapter)
            self.pipelines = model
            if adapter == "ON":
                self.pipelines = IPAdapterPlusXL(
                    model,
                    ip_ckpt=self.ip_ckpt,
                    image_encoder_path=self.image_encoder,
                    device=paddle.get_device(),
                    num_tokens=16,
                ).generate
                assert face_image is not None, "Face image must be required."
                face_image = load_image(face_image).resize((224, 224))

        # 保存状态值
        self.pipe = pipe
        self.type_to_img = type_to_img
        self.scheduler_type = scheduler_type
        self.adapter = adapter
        self.enable_xformers = enable_xformers

        if (image is not None) and ((width is None or height is None) or min(height, width) < 128):
            (width, height) = load_image(image).size
        if (image is None) and width is None or height is None or min(height, width) < 128:
            width, height = 1024, 1024
        # 目标尺寸调整,设置尺寸上限（最大边）max_size=1900，防止崩溃
        width, height = self.re_size(width, height, max_size=max_size)
        metadata["height"] = int(height)
        metadata["width"] = int(width)

        # 读取图片
        if image:
            image = load_image(image).resize((width, height))
        # 读取mask图片
        if mask_image:
            mask_image = load_image(mask_image).resize((width, height))

        # 设置随机种子，我们可以复现下面的结果！
        seeds = np.random.randint(0, np.iinfo(np.int32).max) if (int(seed) == -1) else int(seed)
        generator = paddle.Generator().manual_seed(seeds)
        metadata["seed"] = int(seeds)  # gr.JSON()无法解析numpy.int64, 需做转换

        # 中文prompt翻译成英文
        prompt, negative_prompt = tuple(multi_tasks_translate(prompt, negative_prompt))

        imgs = self.pipelines(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            strength=strength,
            image=image,
            mask_image=mask_image,
            pil_image=face_image,  # crop and resize image
            control_guidance_start=control_guidance_start,  # 控制姿势，值越小越接近原图0~1
            control_guidance_end=control_guidance_end,
            scale=scale,  # 值越小与原图差异越大0~1
        ).images
        paddle.device.cuda.empty_cache()
        for image in imgs:
            custom_save_image(image, metadata, self.outputs, pipe)
        return imgs, metadata

    def resize_img(self, img_path, up_scale, max_limit_size=2048):
        # 读取图片
        img = load_image(img_path)

        # 以最大尺寸边归一化
        img = np.array(img.size)
        max_w_h = img.max()
        img_normal = img / max_w_h

        # 如果超分后尺寸大于2048*2048, 需要将图片调整到2048px
        max_size = max_limit_size if ((max_w_h * up_scale) >= max_limit_size) else max_w_h * up_scale
        # 重新调整尺寸
        img_new = np.floor(img_normal * max_size / 4).astype("int")
        # 重新调整尺寸
        return load_image(img_path).resize((img_new[0], img_new[1]), Image.ANTIALIAS)

    def re_size(self, w, h, max_size=2048):
        img = np.array([w, h])
        max_w_h = img.max()
        img_normal = img / max_w_h

        # 如果超分后尺寸大于2048*2048, 需要将图片调整到2048px
        max_w_h = max_size if max_w_h > max_size else max_w_h
        return np.floor(img_normal * max_w_h // 8 * 8).astype("int")

    def num_differ_imgs(self, height, width, num_images_per_prompt, max_limit=2048):
        # 单次生成图数目，
        # 比如num_images_per_prompt=20，1024就会分成5*4；2048就会分成20*1
        num_images_per_iter_per_prompt = math.floor(max_limit / height) * math.floor(max_limit / width)
        num_images_per_iter_per_prompt = (
            1
            if num_images_per_iter_per_prompt < 1
            else num_images_per_iter_per_prompt
            if num_images_per_iter_per_prompt < 4
            else 4
        )
        # 在最大尺寸限制下生成指定数目图片需要的批次
        num_iters = int(math.ceil(num_images_per_prompt / num_images_per_iter_per_prompt))
        return num_iters, num_images_per_iter_per_prompt

    # 加载模型
    def load_types_model(self, pipe, type_to_img="text", enable_xformers="ON", scheduler_type="ddpm", adapter="ON"):
        # 获取pipeline
        if type_to_img == "text":
            pipe_line = StableDiffusionXLCustomPipeline if adapter == "ON" else StableDiffusionXLPipeline
        if type_to_img == "img":
            pipe_line = StableDiffusionXLImg2ImgPipeline
        if type_to_img == "inp":
            pipe_line = StableDiffusionXLInpaintPipeline

        # 判断条件,加载pipeline
        return self.conditional_model(pipe, pipe_line, enable_xformers=enable_xformers, scheduler_type=scheduler_type)

    # 根据条件加载模型
    def conditional_model(self, pipe, pipe_line, enable_xformers="ON", scheduler_type="ddpm"):
        if scheduler_type != "default":
            model = pipe_line(**self.model.components)
            self.switch_scheduler(model, scheduler_type)
        else:
            model = pipe_line(**self.model.components)

        if enable_xformers == "ON":
            model.enable_xformers_memory_efficient_attention()
        else:
            model.enable_xformers_memory_efficient_attention(False)

        return model

    # 切换采样器
    def switch_scheduler(self, model, scheduler_type="ddim"):
        scheduler_type = scheduler_type.lower()
        from ppdiffusers import (
            DDIMScheduler,
            DDPMScheduler,
            DEISMultistepScheduler,
            DPMSolverMultistepScheduler,
            DPMSolverSinglestepScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            HeunDiscreteScheduler,
            KDPM2AncestralDiscreteScheduler,
            KDPM2DiscreteScheduler,
            LMSDiscreteScheduler,
            PNDMScheduler,
            UniPCMultistepScheduler,
        )

        if scheduler_type == "pndm":
            scheduler = PNDMScheduler.from_config(model.scheduler.config, skip_prk_steps=True)
        elif scheduler_type == "lms":
            scheduler = LMSDiscreteScheduler.from_config(model.scheduler.config)
        elif scheduler_type == "heun":
            scheduler = HeunDiscreteScheduler.from_config(model.scheduler.config)
        elif scheduler_type == "euler":
            scheduler = EulerDiscreteScheduler.from_config(model.scheduler.config)
        elif scheduler_type == "euler-ancestral":
            scheduler = EulerAncestralDiscreteScheduler.from_config(model.scheduler.config)
        elif scheduler_type == "dpm-multi":
            scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)
        elif scheduler_type == "dpm-single":
            scheduler = DPMSolverSinglestepScheduler.from_config(model.scheduler.config)
        elif scheduler_type == "kdpm2-ancestral":
            scheduler = KDPM2AncestralDiscreteScheduler.from_config(model.scheduler.config)
        elif scheduler_type == "kdpm2":
            scheduler = KDPM2DiscreteScheduler.from_config(model.scheduler.config)
        elif scheduler_type == "unipc-multi":
            scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)
        elif scheduler_type == "ddim":
            scheduler = DDIMScheduler.from_config(
                model.scheduler.config,
                steps_offset=1,
                clip_sample=False,
                set_alpha_to_one=False,
            )
        elif scheduler_type == "ddpm":
            scheduler = DDPMScheduler.from_config(
                model.scheduler.config,
            )
        elif scheduler_type == "deis-multi":
            scheduler = DEISMultistepScheduler.from_config(
                model.scheduler.config,
            )
        else:
            raise ValueError(
                f"Scheduler of type {scheduler_type} doesn't exist! Please choose in {self.supported_scheduler}!"
            )
        model.scheduler = scheduler


if __name__ == "__main__":
    LoadTypesModel()

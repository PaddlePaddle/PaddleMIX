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

import math
import os
import re

import cv2
import numpy as np
import paddle
from PIL import Image

from ppdiffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from ppdiffusers.utils import load_image

from .baidufanyi import multi_tasks_translate
from .check_image import custom_save_image

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
STATIC_DIR = os.path.join(PROJECT_DIR, "assets")
HOME_DIR = os.path.expanduser("~")


class LoadTypesModel:
    def __init__(
        self,
        pipe="SG161222/RealVisXL_V3.0",
        controlnet_model_name="diffusers/controlnet-canny-sdxl-1.0",
        controlnet="ON",
        unet_dir="SG161222/RealVisXL_V3.0/unet",
        vae_dir="madebyollin/sdxl-vae-fp16-fix",
        outputs="out_puts",
    ):
        self.pipe = ""
        self.controlnet_model_name = ""
        self.controlnet = ""
        self.type_to_img = ""
        self.scheduler_type = ""
        self.unet_dir = unet_dir
        self.vae_dir = vae_dir

        # load vae
        self.vae = AutoencoderKL.from_pretrained(vae_dir, paddle_dtype=paddle.float16)

        # load controlnet
        self.control_model = ControlNetModel.from_pretrained(controlnet_model_name, paddle_dtype=paddle.float16)

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

        if controlnet == "ON":
            if is_fp16_model:
                self.model = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                    pipe,
                    paddle_dtype=paddle.float16,
                    vae=self.vae,
                    # unet=self.unet,
                    use_safetensors=True,
                    variant="fp16",
                    safety_checker=None,
                    low_cpu_mem_usage=True,
                    controlnet=self.control_model,
                )
            else:
                self.model = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                    pipe,
                    paddle_dtype=paddle.float16,
                    vae=self.vae,
                    # unet=self.unet,
                    safety_checker=None,
                    low_cpu_mem_usage=True,
                    controlnet=self.control_model,
                )
        else:
            if is_fp16_model:
                self.model = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    pipe,
                    paddle_dtype=paddle.float16,
                    vae=self.vae,
                    # unet=self.unet,
                    use_safetensors=True,
                    variant="fp16",
                    safety_checker=None,
                    low_cpu_mem_usage=True,
                )
            else:
                self.model = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    pipe,
                    paddle_dtype=paddle.float16,
                    vae=self.vae,
                    # unet=self.unet,
                    safety_checker=None,
                    low_cpu_mem_usage=True,
                )

        self.outputs = os.path.join(HOME_DIR, outputs)
        # 创建图片路径
        os.makedirs(self.outputs, exist_ok=True)

    # 图片生成
    def mix2img(
        self,
        pipe,
        lora,
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
        ration=0.5,
        image=None,
        enable_xformers="ON",
        control_image=None,
        mask_image=None,
        strength=0.999999,
        controlnet="OFF",
        model_name_input=None,
        controlnet_model_name="diffusers/controlnet-canny-sdxl-1.0",
        max_size=1900,
        control_guidance_start=0.1,
        control_guidance_end=0.99,
        **kwargs
    ):

        metadata = {
            "pipe": pipe,
            "lora": lora,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "scheduler_type": scheduler_type,
            "num_inference_steps": num_inference_steps,
            "type_to_img": type_to_img,
            "strength": strength,
            "controlnet": controlnet,
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
        }

        # 只要任意一个变量变化都会触发重新加载
        if (
            self.controlnet != controlnet
            or self.type_to_img != type_to_img
            or self.scheduler_type != scheduler_type
            or self.enable_xformers != enable_xformers
        ):

            # 加载model
            self.pipeline = self.load_types_model(controlnet, type_to_img, enable_xformers, scheduler_type)

        # 保存当前状态
        self.pipe = pipe
        self.controlnet_model_name = controlnet_model_name
        self.controlnet = controlnet
        self.type_to_img = type_to_img
        self.scheduler_type = scheduler_type
        self.enable_xformers = enable_xformers

        # 根据模型类型改变配置
        if type_to_img == "text":
            image = control_image
            metadata.pop("strength")

        if not lora:
            self.pipeline.unfuse_lora()
        try:
            # 提取 lora 缩放比例
            pattern_scale = re.compile(r"<lora:(-?\d*\.?\d*)>")
            lora_scales = re.findall(pattern_scale, prompt)
            if lora_scales:
                lora_scale = float(lora_scales[0])
            else:
                lora_scale = 0

            # 移除所有 <*> 格式的内容
            prompt = re.sub(r"<.*?>", "", prompt)

            # 加载和融合 lora 权重
            self.pipeline.load_lora_weights(lora, from_diffusers=True, from_hf_hub=True)
            self.pipeline.fuse_lora(lora_scale)
        except Exception as e:
            print("处理lora时发生错误：", e)

        # 中文prompt翻译成英文
        prompt, negative_prompt = tuple(multi_tasks_translate(prompt, negative_prompt))

        if int(height) < 128 or int(width) < 128:
            width = None
            height = None
        if (image is not None) and (width is None or height is None):
            (width, height) = load_image(image).size
        if image is None and (width is None or height is None):
            width, height = 1024, 1024

        if controlnet == "ON":
            # controlnet图片处理
            if control_image is not None:
                control_image = load_image(control_image)
                # 如果其中一个为零则使用controlnet图尺寸
                if min(height, width) < 128 or width is None or height is None:
                    width, height = control_image.size
                # controlnet图生成线稿
                control_image = np.array(control_image)
                control_image = cv2.Canny(control_image, 100, 200)
                control_image = control_image[:, :, None]
                control_image = np.concatenate([control_image, control_image, control_image], axis=2)
                control_image = Image.fromarray(control_image)

                control_image = control_image.resize(self.re_size(control_image.size[0], control_image.size[1]))
            metadata["controlnet_params"] = {}
            metadata["controlnet_params"]["controlnet_model_name"] = controlnet_model_name
            metadata["controlnet_params"]["ratoin"] = ration
        else:
            control_image = None

        # 目标尺寸调整,设置尺寸上限（最大边）max_size=1980，防止崩溃
        height, width = self.re_size(height, width, max_size=max_size)
        metadata["height"] = int(height)
        metadata["width"] = int(width)

        # 读取图片
        if image:
            image = load_image(image).resize((width, height))
        # 读取mask图片
        if mask_image:
            mask_image = load_image(mask_image).resize((width, height))

        seeds = np.random.randint(1, np.iinfo(np.int32).max) if (int(seed) == -1) else int(seed)
        # 设置随机种子，我们可以复现下面的结果！
        generator = paddle.Generator().manual_seed(seeds)
        metadata["seed"] = int(seeds)

        # 生成多张图片，最大为4
        imgs = self.pipeline(
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
            control_image=control_image,
            mask_image=mask_image,
            controlnet_conditioning_scale=ration,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
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
        return np.floor(img_normal * max_w_h).astype("int")

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
    def load_types_model(self, controlnet="OFF", type_to_img="text", enable_xformers="ON", scheduler_type="ddpm"):

        # 获取pipeline
        if type_to_img == "text":
            pipe_line = StableDiffusionXLPipeline if controlnet == "OFF" else StableDiffusionXLControlNetPipeline
        if type_to_img == "img":
            pipe_line = (
                StableDiffusionXLImg2ImgPipeline if controlnet == "OFF" else StableDiffusionXLControlNetImg2ImgPipeline
            )
        if type_to_img == "inp":
            pipe_line = (
                StableDiffusionXLInpaintPipeline if controlnet == "OFF" else StableDiffusionXLControlNetInpaintPipeline
            )

        return self.conditional_model(pipe_line, enable_xformers=enable_xformers, scheduler_type=scheduler_type)

    # 加载模型
    def conditional_model(self, pipe_line, enable_xformers="ON", scheduler_type="ddpm"):
        model = pipe_line(**self.model.components)

        if scheduler_type != "default":
            self.switch_scheduler(model, scheduler_type)

        if enable_xformers == "ON":
            model.enable_xformers_memory_efficient_attention()
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

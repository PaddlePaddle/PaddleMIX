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

import os

import cv2
import numpy as np
import paddle
from PIL import Image

from ppdiffusers import StableDiffusionUpscalePipeline
from ppdiffusers.utils import load_image

from .gan import df2k, drn, esrgan, lesr


def srgan(image_path, method="df2k", output="out_puts"):
    try:
        if method == "df2k":
            pipe = df2k(output=output)
        if method == "drn":
            pipe = drn(output=output)
        if method == "esrgan":
            pipe = esrgan(output=output)
        if method == "lesr":
            pipe = lesr(output=output)
        img = pipe.run(image_path)[0]
        paddle.device.cuda.empty_cache()
    except Exception as e:
        print(e)
        paddle.device.cuda.empty_cache()
    return img


def upscale_x4(
    image_path,
    method="df2k",
    size=(1024, 1024),
    num_inference_steps=20,
    kernel_size=(5, 5),
    pix=5,
    num_cv2=2,
    output_dir="out_puts",
    suffix="jpg",
):
    """
    Args：
        image_path：
            图片路径；
        num_inference_steps：
            推理步数；
        kernel_size：
            通过高斯核模糊处理拼接痕迹，核必须是奇数，值越大越模糊；
        pix：
            修复宽度；
        size：
            切块的大小；
        scale：
            图片放大的倍数；
        num_cv2：
            num_cv2降噪步数
        output_dir：
            图片存储目录。
    """

    if method == "sd":
        size = (256, 256)
        # load model and scheduler
        model_id = "stabilityai/stable-diffusion-x4-upscaler"
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, paddle_dtype=paddle.float16)
        pipeline.enable_xformers_memory_efficient_attention()
        prompt = "ultra-high resolution, detailed and crisp,"
        negative_prompt = "blurry, ugly, distortions, low-quality"
        generator = paddle.Generator().manual_seed(56465165)

    def pipe_line(image, method="df2k"):
        if method == "sd":
            with paddle.amp.auto_cast(False):
                with paddle.no_grad():
                    return pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        guidance_scale=0,
                        num_inference_steps=num_inference_steps,
                        image=image,
                        generator=generator,
                    ).images[0]
        if method == "df2k":
            return srgan(image_path=image, method="df2k", output=None)
        if method == "drn":
            return srgan(image_path=image, method="drn", output=None)
        if method == "esrgan":
            return srgan(image_path=image, method="esrgan", output=None)
        if method == "lesr":
            return srgan(image_path=image, method="lesr", output=None)

    # output_dir="out_puts" # 文件保存目录
    os.makedirs(output_dir, exist_ok=True)

    scale = 4  # 放大的倍数，由模型决定

    # 获取图片的宽度和高度
    img = load_image(image_path)  # 默认RGB格式
    width, height = img.size
    if width < size[0] and height < size[1]:
        # 保存平滑后的图像
        filename_without_extension = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{filename_without_extension}_upscaled-x4.{suffix}"
        if len(output_dir.strip()) > 0:
            output_path = f"{output_dir}/{filename_without_extension}_upscaled-x4.{suffix}"
        img = pipe_line(img, method).save(output_path)
        print(f"Saving {output_path} have done!")
        return output_path, img
    # 计算在宽度和高度上分别需要分割成多少块
    tiles_x = (width + size[0] - 1) // size[0]
    tiles_y = (height + size[1] - 1) // size[1]
    # 创建放大后图片粘贴空间
    target_size = scale * width, scale * height
    output_img = Image.new("RGB", target_size)

    # 遍历所有块放大4倍
    # 重新优化了切块方法，避免无法整除而产生黑色区域
    borders = {"x": [], "y": []}  # 用于拼接图片
    for i in range(tiles_x):
        for j in range(tiles_y):
            # 计算当前块的左上角坐标
            box = (i * size[0], j * size[1], (i + 1) * size[0], (j + 1) * size[1])

            # 如果tile小于块尺寸，剩余量超分会明显产生差异
            width_diff = width - (i + 1) * size[0]
            height_diff = height - (j + 1) * size[1]
            if width_diff < 0:
                box = (i * size[0] + width_diff, j * size[1], width, (j + 1) * size[1])
            if height_diff < 0:
                box = (i * size[0], j * size[1] + height_diff, (i + 1) * size[0], height)
            if width_diff < 0 and height_diff < 0:
                box = (i * size[0] + width_diff, j * size[1] + height_diff, width, height)

            x = box[0] * scale  # 图片拼接X坐标
            y = box[1] * scale  # 图片拼接Y坐标
            borders["x"].append(x)
            borders["y"].append(y)

            # 裁剪图片块
            tile = img.crop(box)

            # 超分x4
            tile = pipe_line(image=tile, method=method)

            # 将放大后的图片块粘贴到输出图片中
            output_img.paste(tile, (x, y))

    # 将pil转成numpy数组，以便cv2的作接缝处理
    output_img = np.array(output_img)  # hwc

    # 获取接缝区域的掩模版，接缝处理区域宽度x+1-x+1=2pixs
    mask = np.zeros(output_img.shape[:2], dtype=np.uint8)
    for x in borders["x"]:
        mask[:, x - pix : x + pix] = 255
    for y in borders["y"]:
        mask[y - pix : y + pix, :] = 255
    # cv2.imwrite('mask.jpg', mask) # 保存查看mask是否正确
    mask = mask / 255.0  # 归一化mask到0-1范围

    # 扩展mask以匹配图像的维度
    mask = np.expand_dims(mask, axis=2)  # 形状变为 (6240, 2880, 1)
    mask = np.repeat(mask, 3, axis=2)  # 复制mask以匹配颜色通道，形状变为 (6240, 2880, 3)

    # 应用高斯平滑的区域
    blurred_region = output_img
    for i in range(num_cv2):
        blurred_region = cv2.GaussianBlur(blurred_region, kernel_size, 0)  # 通过高斯核模糊处理拼接痕迹

    # 使用mask将原始图像和模糊后的图像混合
    # 使用NumPy的where函数根据mask的值选择像素
    output_img_np = np.where(mask == 1, blurred_region, output_img)

    # 如果需要，保存平滑后的图像
    filename_without_extension = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"{filename_without_extension}_upscaled-x4.{suffix}"
    if len(output_dir.strip()) > 0:
        output_path = f"{output_dir}/{filename_without_extension}_upscaled-x4.{suffix}"

    # output_img_np = cv2.GaussianBlur(output_img_np, (3, 3), 0) # 全局降噪
    # Pillow方式保存图片
    output_img = Image.fromarray(output_img_np)
    output_img.save(output_path)
    print(f"Saving {output_path} have done!")
    paddle.device.cuda.empty_cache()
    return output_path, output_img

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


import gradio as gr
import imageio
from PIL import Image

from paddlemix.appflow import Appflow


# upscaling
def ups_fun(low_res_img, prompt):
    low_res_img = Image.fromarray(low_res_img.astype("uint8")).convert("RGB")
    app = Appflow(app="image2image_text_guided_upscaling", models=["stabilityai/stable-diffusion-x4-upscaler"])
    image = app(prompt=prompt, image=low_res_img)["result"]
    return image


# text_guided_generation
def tge_fun(image, prompt_pos, prompt_neg):
    image = Image.fromarray(image.astype("uint8")).convert("RGB")
    app = Appflow(app="image2image_text_guided_generation", models=["Linaqruf/anything-v3.0"])
    image = app(prompt=prompt_pos, negative_prompt=prompt_neg, image=image)["result"]
    return image


# video_generation
def vge_fun(prompt):
    app = Appflow(app="text_to_video_generation", models=["damo-vilab/text-to-video-ms-1.7b"])
    video_frames = app(prompt=prompt, num_inference_steps=25)["result"]
    imageio.mimsave("gen_video.gif", video_frames, duration=8)
    return "gen_video.gif"


with gr.Blocks() as demo:
    gr.Markdown("# Appflow应用：text2image")
    with gr.Tab("文本引导的图像放大"):
        with gr.Row():
            ups_image_in = gr.Image(label="输入图片")
            ups_image_out = gr.Image(label="输出图片")
        ups_text_in = gr.Text(label="Prompt")
        ups_button = gr.Button()
        ups_button.click(fn=ups_fun, inputs=[ups_image_in, ups_text_in], outputs=[ups_image_out])
    with gr.Tab("文本引导的图像变换"):
        with gr.Row():
            tge_image_in = gr.Image(label="输入图片")
            tge_image_out = gr.Image(label="输出图片")
        tge_text_pos_in = gr.Text(label="Positive Prompt")
        tge_text_neg_in = gr.Text(label="Negative Prompt")
        tge_button = gr.Button()
        tge_button.click(fn=tge_fun, inputs=[tge_image_in, tge_text_pos_in, tge_text_neg_in], outputs=[tge_image_out])
    with gr.Tab("文本条件的视频生成"):
        vge_text_in = gr.Text(label="Prompt")
        vge_video_out = gr.Video(label="输出视频")
        vge_button = gr.Button()
        vge_button.click(fn=vge_fun, inputs=[vge_text_in], outputs=[vge_video_out])

demo.launch()

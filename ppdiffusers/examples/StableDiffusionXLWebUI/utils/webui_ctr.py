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

import gc
import json
import os
from glob import glob

import gradio as gr
import paddle

from ppdiffusers.utils import load_image

from .baidufanyi import multi_tasks_translate
from .check_image import check_image_infos
from .upscale import upscale_x4
from .webui_ctr_func import LoadTypesModel

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
STATIC_DIR = os.path.join(PROJECT_DIR, "static")
HOME_DIR = os.path.expanduser("~")


def get_dirs_depth(path, depth):
    current_dirs = [root for root, _, _ in os.walk(path) if str(root).count("/") == depth]
    return current_dirs


# lora模型所在目录
LORA_DIR = os.path.join(STATIC_DIR, "lora")
# 获取LORA模型路径列表
lora_name_list = glob(LORA_DIR + "/*")
lora_list = [name for name in lora_name_list]
lora_list.extend([""])

# model模型所在目录
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
# 获取model模型路径列表
model_name_list = get_dirs_depth(MODEL_DIR, 2) or [""]
model_list = [name for name in model_name_list]
model_list.extend(
    [
        "SG161222/RealVisXL_V3.0",
        "stabilityai/stable-diffusion-xl-base-1.0",
        os.path.join(PROJECT_DIR, "Pony_Pencil-Xl-V1.0.2"),
    ]
)

# 获取ctn model模型路径列表
CTN_MODEL_DIR = os.path.join(PROJECT_DIR, "controlnet")
control_model_name_list = get_dirs_depth(CTN_MODEL_DIR, 1) or [""]
control_model_list = [name for name in control_model_name_list]
control_model_list.extend(["diffusers/controlnet-canny-sdxl-1.0"])

# unet
unet_model = [
    "",
    "SG161222/RealVisXL_V3.0/unet",
]

# vae
vae_model = [
    "",
    "madebyollin/sdxl-vae-fp16-fix",
]

# Scheduler列表
supported_scheduler = [
    "default",
    "pndm",
    "lms",
    "euler",
    "euler-ancestral",
    "dpm-multi",
    "dpm-single",
    "unipc-multi",
    "ddim",
    "ddpm",
    "deis-multi",
    "heun",
    "kdpm2-ancestral",
    "kdpm2",
]

# 超分算法
method_list = ["df2k", "drn", "esrgan", "lesr"]

negative_prompt = "nsfw, lowres, (bad), multy girls, text, extra fingers, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"

# Grdio WebUI
with gr.Blocks(css=os.path.join(STATIC_DIR, "style.css")) as demo:
    # 顶部文字
    gr.Markdown(
        """
        # AI绘画
        ### 基于PPdiffusers的文生图webui。如果宽或高小于128则使用输入图片尺寸
        """
    )
    with gr.Row():
        with gr.Column():
            # 换模型
            with gr.Row():
                model_name = gr.Dropdown(
                    model_list,
                    label="BaseModel",
                    value="SG161222/RealVisXL_V3.0",
                    multiselect=False,
                    interactive=True,
                    scale=1,
                )
                enable_xformers = gr.Radio(["ON", "OFF"], value="ON", label="EnableXformers", scale=0)
            model_name_input = gr.Textbox(
                label="优先选用输入模型路径。",
                lines=1,
                placeholder="例如：/home/aistudio/PPdiffusersWebUI/Animagine-Xl-3.1",
                interactive=True,
                value=None,
                scale=1,
            )

            with gr.Row():
                with gr.Accordion(label="History Model", open=False):
                    history_models = gr.JSON(label="History Model", scale=1)
                max_size_limit = gr.Slider(
                    minimum=1024, maximum=2048, value=1800, step=8, label="Max Target Size", interactive=True
                )

            with gr.Accordion(label="More", open=False):
                with gr.Row():
                    lora_name = gr.Dropdown(
                        lora_list,
                        label="Lora",
                        placeholder="请将Lora放在static/lora目录",
                        multiselect=False,
                        interactive=True,
                        scale=1,
                    )
                    lora_tag_button = gr.Button('Get Lora Tag"<lora:0.5>"', scale=0, elem_id="btn1")
                    lora_tag = gr.Textbox(label="Lora Tag", scale=1, placeholder="<lora:0.5>")

                with gr.Row():
                    vae_dir = gr.Dropdown(
                        vae_model,
                        label="Vae Model",
                        value="",
                        multiselect=False,
                        interactive=True,
                    )
                    vae_model_name_input = gr.Textbox(
                        label="优先选用输入Vae Model",
                        lines=1,
                        placeholder="例如：SG161222/RealVisXL_V3.0/vae，默认为空",
                        interactive=True,
                        value="",
                    )

            with gr.Row():
                with gr.Column(scale=5):
                    controlnet_model_name = gr.Dropdown(
                        control_model_list,
                        label="Control Model",
                        value="diffusers/controlnet-canny-sdxl-1.0",
                        multiselect=False,
                        interactive=True,
                        scale=1,
                    )
                    ration = gr.Slider(minimum=0, maximum=0.95, value=0.5, step=0.01, label="Ration", interactive=True)
                    ctn = gr.Radio(["ON", "OFF"], value="ON", label="ControlNet", scale=0)
                controlnet_image = gr.Image(label="controlnet参考图", scale=5, type="pil")

            with gr.Row():
                scheduler_type = gr.Radio(supported_scheduler, value="ddpm", label="Scheduler")

        with gr.Tabs():
            with gr.TabItem("文生图"):
                with gr.Row():
                    with gr.Column():
                        text2img_prompt = gr.Textbox(
                            label="prompt", lines=2, placeholder="请输入正面描述", interactive=True, value=None
                        )
                        text2img_negative_prompt = gr.Textbox(
                            label="negative_prompt", lines=2, placeholder="请输入负面描述", interactive=True, value=None
                        )
                    text2img_prompt_zh2en_button = gr.Button("汉译英", min_width=50, elem_id="btn2", scale=0)
                    with gr.Column():
                        text2img_prompt_zh2en = gr.Textbox(label="翻译结果", lines=2, interactive=True, value=None)
                        text2img_negative_prompt_zh2en = gr.Textbox(lines=2, interactive=True, value=None)

                with gr.Row():
                    text2img_steps = gr.Slider(
                        minimum=1, maximum=400, value=30, step=1, label="Sampling steps", interactive=True
                    )
                    text2img_cfg_scale = gr.Slider(
                        minimum=1, maximum=20, value=4, step=0.1, label="CFG Scale", interactive=True
                    )
                    text2img_width = gr.Slider(
                        minimum=8, maximum=2048, value=760, step=8, label="Width", interactive=True
                    )
                    text2img_height = gr.Slider(
                        minimum=8, maximum=2048, value=1440, step=8, label="Height", interactive=True
                    )
                    text2img_num_images = gr.Slider(
                        minimum=1, maximum=400, value=1, step=1, label="Num Images", interactive=True
                    )
                    text2img_seed = gr.Textbox(label="seed", value="-1", lines=1, placeholder="请输入种子，默认-1")
                    with gr.Row(elem_id="btn"):
                        text2img_button = gr.Button(
                            "RUN",
                        )
                with gr.Accordion(label="Generation Parameters", open=False):
                    text2img_metadata = gr.JSON()

            with gr.TabItem("图生图"):
                with gr.Row():
                    img2img_prompt = gr.Textbox(
                        label="prompt", lines=3, placeholder="请输入正面描述", interactive=True, value=None
                    )
                    img2img_negative_prompt = gr.Textbox(
                        label="negative_prompt", lines=2, placeholder="请输入负面描述", interactive=True, value=None
                    )
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            img2img_steps = gr.Slider(
                                minimum=1, maximum=401, value=30, step=1, label="Sampling steps", interactive=True
                            )
                            img2img_width = gr.Slider(
                                minimum=8, maximum=2048, value=1024, step=8, label="Width", interactive=True
                            )
                            img2img_height = gr.Slider(
                                minimum=8, maximum=2048, value=1024, step=8, label="Height", interactive=True
                            )
                        with gr.Row():
                            img2img_num_images = gr.Slider(
                                minimum=1, maximum=400, value=1, step=1, label="Num Images", interactive=True
                            )
                            img2img_cfg_scale = gr.Slider(
                                minimum=1, maximum=20, value=4, step=0.1, label="CFG Scale", interactive=True
                            )
                        with gr.Row():
                            img2img_strength = gr.Slider(
                                minimum=0, maximum=1, value=0.5, step=0.01, label="Strength", interactive=True
                            )
                            img2img_seed = gr.Textbox(label="seed", value="-1", lines=1, placeholder="请输入种子，默认-1")
                    with gr.Column():
                        img2img_img = gr.Image(type="pil")
                        img2img_button = gr.Button("RUN", elem_id="btn", scale=0, min_width=100)
                    with gr.Accordion(label="Generation Parameters", open=False):
                        img2img_metadata = gr.JSON()

            with gr.TabItem("局绘图"):
                with gr.Row():
                    inp2img_prompt = gr.Textbox(
                        label="prompt", lines=3, placeholder="请输入正面描述", interactive=True, value=None
                    )
                    inp2img_negative_prompt = gr.Textbox(
                        label="negative_prompt", lines=2, placeholder="请输入负面描述", interactive=True, value=None
                    )
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            inp2img_steps = gr.Slider(
                                minimum=1, maximum=400, value=30, step=1, label="Steps", interactive=True
                            )
                            inp2img_width = gr.Slider(
                                minimum=8, maximum=2048, value=8, step=8, label="Width", interactive=True
                            )
                            inp2img_height = gr.Slider(
                                minimum=8, maximum=2048, value=8, step=8, label="Height", interactive=True
                            )
                            inp2img_num_images = gr.Slider(
                                minimum=1, maximum=100, value=1, step=1, label="Num Images", interactive=True
                            )
                            inp2img_strength = gr.Slider(
                                minimum=0, maximum=1.0, value=0.8, step=0.01, label="Strength", interactive=True
                            )
                            inp2img_cfg_scale = gr.Slider(
                                minimum=1, maximum=20, value=4, step=0.1, label="CFG Scale", interactive=True
                            )
                            inp2img_seed = gr.Textbox(label="seed", value="-1", lines=1)
                    with gr.Column():
                        inp2img_img = gr.Image(tool="sketch", type="pil")
                        inp2img_button = gr.Button("RUN", elem_id="btn", scale=0, min_width=100)
                    inp2img_img_inputs = gr.Gallery()
                with gr.Accordion(label="Generation Parameters", open=False):
                    inp2img_metadata = gr.JSON()

            with gr.TabItem("放大四倍"):
                svr_img_path = gr.Image(type="filepath")
                with gr.Row():
                    svr_img_method = gr.Dropdown(method_list, value="df2k")
                    svr_img_patch_size = gr.Slider(
                        minimum=256, maximum=1536, value=512, step=256, label="Patch Size", interactive=True
                    )
                    svr2img_button = gr.Button("RUN", min_width=200, scale=0, elem_id="btn")
                sr_outputs = gr.Image()

            with gr.TabItem("查看图片信息"):
                check_img_input = gr.Image(type="pil")
                check_img_button = gr.Button("获取参数")
                with gr.Accordion(label="Generation Parameters", open=False):
                    check_img_outputs = gr.JSON()

    with gr.Column():
        with gr.Row():
            all_outputs_button = gr.Button("查看所有结果")
            remove_images = gr.Button("删除所有结果")
            all_outputs_text = gr.Text(lines=1)
        all_outputs = gr.Gallery()

    # 获取图片路径
    def get_imgs_path():
        try:
            if os.path.exists(os.path.join(HOME_DIR, "out_puts")):
                imgs_path = os.path.join(HOME_DIR, "out_puts")
                files_list = []
                for r, d, f in os.walk(imgs_path):
                    files_list.extend(
                        [
                            os.path.join(r, fn)
                            for fn in f
                            if (fn.lower().endswith((".jpg", ".png")) and "-checkpoint." not in fn.lower())
                        ]
                    )
                return files_list
        except:
            return None

    # 画廊展示
    all_outputs_button.click(fn=get_imgs_path, inputs=None, outputs=all_outputs)

    # 删除图片
    def remove_files():
        imgs_path = os.path.join(HOME_DIR, "out_puts")
        os.system(f"rm -rf {imgs_path}")
        return "清理完成"

    remove_images.click(fn=remove_files, inputs=None, outputs=all_outputs_text)

    # 获取lora tag
    def get_lora_tag(lora):
        lora = "<lora:1.5>"
        return lora

    lora_tag_button.click(get_lora_tag, lora_name, lora_tag)

    def prompt_zh2en(prompt="", negative_prompt=""):
        return tuple(multi_tasks_translate(prompt, negative_prompt))

    # 文生图翻译
    text2img_prompt_zh2en_button.click(
        fn=prompt_zh2en,
        inputs=[text2img_prompt, text2img_negative_prompt],
        outputs=[text2img_prompt_zh2en, text2img_negative_prompt_zh2en],
    )

    # 初始化模型
    load_model = None

    # 文生图函数
    def text2img(
        pipe,
        lora,
        prompt,
        negative_prompt,
        guidance_scale=5,
        height=1024,
        width=1024,
        num_inference_steps=30,
        seed=-1,
        num_images_per_prompt=1,
        controlnet="OFF",
        ration=0.1,
        scheduler_type="ddim",
        model_name_input=None,
        control_image=None,
        enable_xformers="ON",
        controlnet_model_name="diffusers/controlnet-canny-sdxl-1.0",
        vae_dir="",
        vae_dir_input="",
        unet_dir="",
        unet_dir_input="",
        max_size=1900,
        type_to_img="text",
        **kwargs
    ):
        pipe = model_name_input or pipe
        unet_dir = unet_dir_input or unet_dir
        unet_dir = unet_dir if unet_dir else (pipe + "/unet")
        vae_dir = vae_dir_input or vae_dir
        vae_dir = vae_dir if vae_dir else (pipe + "/vae")
        # 如果模型改变则重新加载模型
        global load_model
        if not load_model:
            load_model = LoadTypesModel(pipe, controlnet_model_name, controlnet, unet_dir=unet_dir, vae_dir=vae_dir)
        elif (
            load_model.pipe != pipe
            or load_model.controlnet_model_name != controlnet_model_name
            or load_model.controlnet != controlnet
            or load_model.vae_dir != vae_dir
        ):
            # 切换模型时释放内存
            del load_model
            gc.collect()
            load_model = LoadTypesModel(pipe, controlnet_model_name, controlnet, unet_dir=unet_dir, vae_dir=vae_dir)
            if pipe not in model_list:
                model_list.append(pipe)

        return load_model.mix2img(
            pipe,
            lora,
            prompt,
            negative_prompt,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            scheduler_type=scheduler_type,
            seed=seed,
            ration=ration,
            control_image=control_image,
            enable_xformers=enable_xformers,
            type_to_img=type_to_img,
            controlnet=controlnet,
            model_name_input=model_name_input,
            max_size=max_size,
        ) + (json.dumps(model_list),)

    # 文生图按钮
    text2img_button.click(
        fn=prompt_zh2en,
        inputs=[text2img_prompt, text2img_negative_prompt],
        outputs=[text2img_prompt_zh2en, text2img_negative_prompt_zh2en],
    ).then(
        text2img,
        inputs=[
            model_name,
            lora_name,
            text2img_prompt,
            text2img_negative_prompt,
            text2img_cfg_scale,
            text2img_height,
            text2img_width,
            text2img_steps,
            text2img_seed,
            text2img_num_images,
            ctn,
            ration,
            scheduler_type,
            model_name_input,
            controlnet_image,
            enable_xformers,
            controlnet_model_name,
            vae_dir,
            vae_model_name_input,
            max_size_limit,
        ],
        outputs=[all_outputs, text2img_metadata, history_models],
    )

    # 图生图
    def img2img(
        pipe,
        lora,
        prompt,
        img_image,
        negative_prompt,
        guidance_scale=5,
        height=None,
        width=None,
        num_inference_steps=50,
        seed=-1,
        num_images_per_prompt=1,
        strength=0.5,
        controlnet="OFF",
        ration=0.7,
        scheduler_type="ddpm",
        model_name_input=None,
        control_image=None,
        enable_xformers="ON",
        controlnet_model_name="diffusers/controlnet-canny-sdxl-1.0",
        vae_dir="",
        vae_dir_input="",
        unet_dir="",
        unet_dir_input="",
        max_size=1900,
        type_to_img="img",
        **kwargs
    ):
        pipe = model_name_input or pipe
        unet_dir = unet_dir_input or unet_dir
        unet_dir = unet_dir if unet_dir else (pipe + "/unet")
        vae_dir = vae_dir_input or vae_dir
        vae_dir = vae_dir if vae_dir else (pipe + "/vae")
        # 如果模型改变则重新加载模型
        global load_model
        if not load_model:
            load_model = LoadTypesModel(pipe, controlnet_model_name, controlnet, unet_dir=unet_dir, vae_dir=vae_dir)
        elif (
            load_model.pipe != pipe
            or load_model.controlnet_model_name != controlnet_model_name
            or load_model.controlnet != controlnet
            or load_model.vae_dir != vae_dir
        ):
            # 切换模型时释放内存
            del load_model
            gc.collect()
            load_model = LoadTypesModel(pipe, controlnet_model_name, controlnet, unet_dir=unet_dir, vae_dir=vae_dir)
            if pipe not in model_list:
                model_list.append(pipe)

        return load_model.mix2img(
            pipe,
            lora,
            prompt,
            negative_prompt,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            scheduler_type=scheduler_type,
            seed=seed,
            ration=ration,
            image=img_image,
            control_image=control_image,
            strength=strength,
            enable_xformers=enable_xformers,
            type_to_img=type_to_img,
            controlnet=controlnet,
            model_name_input=model_name_input,
            controlnet_model_name=controlnet_model_name,
            max_size=max_size,
        ) + (json.dumps(model_list),)

    img2img_button.click(
        img2img,
        inputs=[
            model_name,
            lora_name,
            img2img_prompt,
            img2img_img,
            img2img_negative_prompt,
            img2img_cfg_scale,
            img2img_height,
            img2img_width,
            img2img_steps,
            img2img_seed,
            img2img_num_images,
            img2img_strength,
            ctn,
            ration,
            scheduler_type,
            model_name_input,
            controlnet_image,
            enable_xformers,
            controlnet_model_name,
            vae_dir,
            vae_model_name_input,
            max_size_limit,
        ],
        outputs=[all_outputs, img2img_metadata, history_models],
    )

    # 局部绘图
    def inp2img(
        pipe,
        lora,
        prompt,
        inp_image,
        negative_prompt,
        guidance_scale=5,
        height=None,
        width=None,
        num_inference_steps=30,
        seed=-1,
        num_images_per_prompt=1,
        strength=0.75,
        controlnet="OFF",
        ration=0.7,
        scheduler_type="ddpm",
        model_name_input=None,
        control_image=None,
        enable_xformers="ON",
        controlnet_model_name="diffusers/controlnet-canny-sdxl-1.0",
        vae_dir="",
        vae_dir_input="",
        unet_dir="",
        unet_dir_input="",
        max_size=1900,
        type_to_img="inp",
        **kwargs
    ):
        pipe = model_name_input or pipe
        unet_dir = unet_dir_input or unet_dir
        unet_dir = unet_dir if unet_dir else (pipe + "/unet")
        vae_dir = vae_dir_input or vae_dir
        vae_dir = vae_dir if vae_dir else (pipe + "/vae")
        # 如果模型改变则重新加载模型
        global load_model
        if not load_model:
            load_model = LoadTypesModel(pipe, controlnet_model_name, controlnet, unet_dir=unet_dir, vae_dir=vae_dir)
        elif (
            load_model.pipe != pipe
            or load_model.controlnet_model_name != controlnet_model_name
            or load_model.controlnet != controlnet
            or load_model.vae_dir != vae_dir
        ):
            # 切换模型时释放内存
            del load_model
            gc.collect()
            load_model = LoadTypesModel(pipe, controlnet_model_name, controlnet, unet_dir=unet_dir, vae_dir=vae_dir)
            if pipe not in model_list:
                model_list.append(pipe)
        # 读取图片
        image_image = load_image(inp_image.get("image"))
        mask_image = load_image(inp_image.get("mask"))

        return load_model.mix2img(
            pipe,
            lora,
            prompt,
            negative_prompt,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            scheduler_type=scheduler_type,
            seed=seed,
            ration=ration,
            image=image_image,
            mask_image=mask_image,
            control_image=control_image,
            strength=strength,
            enable_xformers=enable_xformers,
            type_to_img=type_to_img,
            controlnet=controlnet,
            model_name_input=model_name_input,
            controlnet_model_name=controlnet_model_name,
            max_size=max_size,
        ) + (json.dumps(model_list),)

    inp2img_button.click(fn=lambda x: [x["image"], x["mask"]], inputs=inp2img_img, outputs=inp2img_img_inputs).then(
        inp2img,
        inputs=[
            model_name,
            lora_name,
            inp2img_prompt,
            inp2img_img,
            inp2img_negative_prompt,
            inp2img_cfg_scale,
            inp2img_height,
            inp2img_width,
            inp2img_steps,
            inp2img_seed,
            inp2img_num_images,
            inp2img_strength,
            ctn,
            ration,
            scheduler_type,
            model_name_input,
            controlnet_image,
            enable_xformers,
            controlnet_model_name,
            vae_dir,
            vae_model_name_input,
            max_size_limit,
        ],
        outputs=[all_outputs, inp2img_metadata, history_models],
    )

    # 放大4倍
    def sr_x4(image_or_path, method="df2k", size=512):
        paddle.device.cuda.empty_cache()
        return load_image(
            upscale_x4(
                image_or_path,
                method=method,
                size=(size, size),
                suffix="jpg",
                output_dir=os.path.join(HOME_DIR, "out_puts"),
            )[0]
        )

    svr2img_button.click(fn=sr_x4, inputs=[svr_img_path, svr_img_method, svr_img_patch_size], outputs=sr_outputs)

    # 获取图片信息
    check_img_button.click(
        fn=check_image_infos, inputs=check_img_input, outputs=check_img_outputs, api_name="Get image info"
    )

    from .prompts import examples

    exec(examples, globals(), locals())
    # exec(examples)


def main():
    demo.queue().launch(share=True)


if __name__ == "__main__":
    main()

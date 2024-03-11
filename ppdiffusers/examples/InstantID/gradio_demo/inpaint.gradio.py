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

os.environ["USE_PEFT_BACKEND"] = "True"
import sys

sys.path.append("..")

import argparse
import math
import random
from typing import Tuple

import cv2
import gradio as gr
import numpy as np
import paddle
import PIL
from insightface.app import FaceAnalysis
from PIL import Image
from pipeline_stable_diffusion_xl_instantid_inpaint import (
    StableDiffusionXLInstantIDInpaintPipeline,
)
from style_template import styles

from ppdiffusers import AutoencoderKL, DDIMScheduler, LCMScheduler
from ppdiffusers.models import ControlNetModel
from ppdiffusers.utils import load_image

# global variable
MAX_SEED = np.iinfo(np.int32).max
dtype = paddle.float16
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Watercolor"


def get_path(abpath: str) -> str:
    build_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(build_dir, abpath)


def init_env():
    # 检查当前文件夹下是否存在ckpts文件夹，如果不存在则执行init_ckpts.sh脚本
    if not os.path.exists(get_path("./ckpts")):
        os.system("bash init_ckpts.sh")
        print("ckpts文件夹不存在，已执行init_ckpts.sh脚本拉取预训练模型")


init_env()

# Load face encoder
app = FaceAnalysis(
    name="antelopev2", root=get_path("./ckpts"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

# Path to InstantID models
# face_adapter = "../checkpoints/ip-adapter.bin"
# controlnet_path = "../checkpoints/ControlNetModel"
# lora_state_dict = "../checkpoints/pytorch_lora_weights.safetensors"

face_adapter = get_path("./ckpts/InstantID/ip-adapter.bin")
controlnet_path = get_path("./ckpts/InstantID/ControlNetModel")
lora_state_dict = get_path("./ckpts/hf-latent-consistency/lcm-lora-sdxl")

# Load pipeline
controlnet = ControlNetModel.from_pretrained(
    controlnet_path, paddle_dtype=paddle.float16, use_safetensors=True, from_hf_hub=True, from_diffusers=True
)


def main(pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0", enable_lcm_arg=False):
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    pipe = StableDiffusionXLInstantIDInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path,
        controlnet=controlnet,
        vae=vae,
        paddle_dtype=paddle.float16,
        # from_diffusers=True,
        # from_hf_hub=True,
        low_cpu_mem_usage=False,
    )
    pipe.vae = vae.to(dtype=paddle.float32)

    pipe.load_ip_adapter_instantid(face_adapter, weight_name=os.path.basename("face_adapter"), from_diffusers=True)
    # pipe.load_lora_weights(lora_state_dict, from_diffusers=True, adapter_name="lcm")
    # pipe.set_adapters("lcm")

    def toggle_lcm_ui(value):
        if value:
            return (
                gr.update(minimum=0, maximum=100, step=1, value=5),
                gr.update(minimum=0.1, maximum=20.0, step=0.1, value=1.5),
            )
        else:
            return (
                gr.update(minimum=5, maximum=100, step=1, value=30),
                gr.update(minimum=0.1, maximum=20.0, step=0.1, value=5),
            )

    def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        return seed

    def remove_tips():
        return gr.update(visible=False)

    def get_example():
        case = [
            [
                "./examples/yann-lecun_resize.jpg",
                "a man",
                "Snow",
                "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
            ],
        ]
        return case

    def run_for_examples(face_file, prompt, style, negative_prompt):
        return generate_image(face_file, None, None, prompt, negative_prompt, style, 30, 0.8, 0.8, 5, 42, False, True)

    def convert_from_cv2_to_image(img: np.ndarray) -> Image:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def convert_from_image_to_cv2(img: Image) -> np.ndarray:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
        stickwidth = 4
        limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
        kps = np.array(kps)

        w, h = image_pil.size
        out_img = np.zeros([h, w, 3])

        for i in range(len(limbSeq)):
            index = limbSeq[i]
            color = color_list[index[0]]

            x = kps[index][:, 0]
            y = kps[index][:, 1]
            length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
            polygon = cv2.ellipse2Poly(
                (int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
        out_img = (out_img * 0.6).astype(np.uint8)

        for idx_kp, kp in enumerate(kps):
            color = color_list[idx_kp]
            x, y = kp
            out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

        out_img_pil = Image.fromarray(out_img.astype(np.uint8))
        return out_img_pil

    def resize_img(
        input_image,
        max_side=1280,
        min_side=1024,
        size=None,
        pad_to_max_side=False,
        mode=PIL.Image.BILINEAR,
        base_pixel_number=64,
    ):

        w, h = input_image.size
        if size is not None:
            w_resize_new, h_resize_new = size
        else:
            ratio = min_side / min(h, w)
            w, h = round(ratio * w), round(ratio * h)
            ratio = max_side / max(h, w)
            input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
            w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
            h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
        input_image = input_image.resize([w_resize_new, h_resize_new], mode)

        if pad_to_max_side:
            res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
            offset_x = (max_side - w_resize_new) // 2
            offset_y = (max_side - h_resize_new) // 2
            res[offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new] = np.array(input_image)
            input_image = Image.fromarray(res)
        return input_image

    def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
        p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
        return p.replace("{prompt}", positive), n + " " + negative

    def generate_image(
        face_image_path,
        pose_image_path,
        template_image_dict,
        prompt,
        negative_prompt,
        style_name,
        num_steps,
        identitynet_strength_ratio,
        adapter_strength_ratio,
        guidance_scale,
        seed,
        enable_LCM,
        enhance_face_region,
        # progress=gr.Progress(track_tqdm=True),
    ):
        if enable_LCM:
            pipe.enable_lora()
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        else:
            pipe.disable_lora()
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        if face_image_path is None:
            raise gr.Error("Cannot find any input face image! Please upload the face image")

        template_image = template_image_dict["image"].convert("RGB")
        mask_image = template_image_dict["mask"].convert("RGB")
        template_image.save("template_image.png")
        mask_image.save("template_image_mask.png")

        if prompt is None:
            prompt = "a person"

        # apply the style template
        prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

        face_image = load_image(face_image_path)
        face_image = resize_img(face_image)
        face_image_cv2 = convert_from_image_to_cv2(face_image)
        height, width, _ = face_image_cv2.shape

        # Extract face features
        face_info = app.get(face_image_cv2)

        if len(face_info) == 0:
            raise gr.Error("Cannot find any face in the image! Please upload another person image")

        face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * x["bbox"][3] - x["bbox"][1])[
            -1
        ]  # only use the maximum face
        face_emb = face_info["embedding"]
        face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info["kps"])

        if pose_image_path is not None:
            pose_image = load_image(pose_image_path)
            pose_image = resize_img(pose_image)
            pose_image_cv2 = convert_from_image_to_cv2(pose_image)

            face_info = app.get(pose_image_cv2)

            if len(face_info) == 0:
                raise gr.Error("Cannot find any face in the reference image! Please upload another person image")

            face_info = face_info[-1]
            face_kps = draw_kps(pose_image, face_info["kps"])

            width, height = face_kps.size

        # if enhance_face_region:
        #     control_mask = np.zeros([height, width, 3])
        #     x1, y1, x2, y2 = face_info["bbox"]
        #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #     control_mask[y1:y2, x1:x2] = 255
        #     control_mask = Image.fromarray(control_mask.astype(np.uint8))
        # else:
        #     control_mask = None

        generator = paddle.Generator().manual_seed(seed)

        print("Start inference...")
        print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")

        pipe.set_ip_adapter_scale(adapter_strength_ratio)
        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            control_image=face_kps,  # controlnet输入
            image=template_image,  # 模版
            mask_image=mask_image,  # 模版mask
            width=template_image.size[0],
            height=template_image.size[1],
            controlnet_conditioning_scale=float(identitynet_strength_ratio),
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images

        return images[0], gr.update(visible=True)

    # Description
    title = r"""
    <h1 align="center">InstantID: Zero-shot Identity-Preserving Generation in Seconds</h1>
    """

    description = r"""
    <b>Official 🤗 Gradio demo</b> for <a href='https://github.com/InstantID/InstantID' target='_blank'><b>InstantID: Zero-shot Identity-Preserving Generation in Seconds</b></a>.<br>

    How to use:<br>
    1. Upload an image with a face. For images with multiple faces, we will only detect the largest face. Ensure the face is not too small and is clearly visible without significant obstructions or blurring.
    2. (Optional) You can upload another image as a reference for the face pose. If you don't, we will use the first detected face image to extract facial landmarks. If you use a cropped face at step 1, it is recommended to upload it to define a new face pose.
    3. Enter a text prompt, as done in normal text-to-image models.
    4. Click the <b>Submit</b> button to begin customization.
    5. Share your customized photo with your friends and enjoy! 😊
    """

    article = r"""
    ---
    📝 **Citation**
    <br>
    If our work is helpful for your research or applications, please cite us via:
    ```bibtex
    @article{wang2024instantid,
    title={InstantID: Zero-shot Identity-Preserving Generation in Seconds},
    author={Wang, Qixun and Bai, Xu and Wang, Haofan and Qin, Zekui and Chen, Anthony},
    journal={arXiv preprint arXiv:2401.07519},
    year={2024}
    }
    ```
    📧 **Contact**
    <br>
    If you have any questions, please feel free to open an issue or directly reach us out at <b>haofanwang.ai@gmail.com</b>.
    """

    tips = r"""
    ### Usage tips of InstantID
    1. If you're not satisfied with the similarity, try increasing the weight of "IdentityNet Strength" and "Adapter Strength."
    2. If you feel that the saturation is too high, first decrease the Adapter strength. If it remains too high, then decrease the IdentityNet strength.
    3. If you find that text control is not as expected, decrease Adapter strength.
    4. If you find that realistic style is not good enough, go for our Github repo and use a more realistic base model.
    """

    css = """
    .gradio-container {width: 85% !important}
    """
    with gr.Blocks(css=css) as demo:

        # description
        gr.Markdown(title)
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                # upload face image
                face_file = gr.Image(label="Upload a photo of your face", type="filepath")

                # optional: upload a reference pose image
                pose_file = gr.Image(label="Upload a reference pose image (optional)", type="filepath")

                template_image_dict = gr.Image(
                    tool="sketch", elem_id="image_upload", type="pil", label="Upload", height=400
                )

                # prompt
                prompt = gr.Textbox(
                    label="Prompt",
                    info="Give simple prompt is enough to achieve good face fidelity",
                    placeholder="A photo of a person",
                    value="",
                )

                submit = gr.Button("Submit", variant="primary")

                enable_LCM = gr.Checkbox(
                    label="Enable Fast Inference with LCM",
                    value=enable_lcm_arg,
                    info="LCM speeds up the inference step, the trade-off is the quality of the generated image. It performs better with portrait face images rather than distant faces",
                )
                style = gr.Dropdown(label="Style template", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME)

                # strength
                identitynet_strength_ratio = gr.Slider(
                    label="IdentityNet strength (for fidelity)",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.80,
                )
                adapter_strength_ratio = gr.Slider(
                    label="Image adapter strength (for detail)",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.80,
                )

                with gr.Accordion(open=False, label="Advanced Options"):
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="low quality",
                        value="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
                    )
                    num_steps = gr.Slider(
                        label="Number of sample steps",
                        minimum=20,
                        maximum=100,
                        step=1,
                        value=5 if enable_lcm_arg else 30,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.1,
                        maximum=10.0,
                        step=0.1,
                        value=0 if enable_lcm_arg else 5,
                    )
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=42,
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    enhance_face_region = gr.Checkbox(label="Enhance non-face region", value=True)

            with gr.Column():
                gallery = gr.Image(label="Generated Images")
                usage_tips = gr.Markdown(label="Usage tips of InstantID", value=tips, visible=False)

            submit.click(fn=remove_tips, outputs=usage_tips,).then(
                fn=randomize_seed_fn,
                inputs=[seed, randomize_seed],
                outputs=seed,
                queue=False,
                api_name=False,
            ).then(
                fn=generate_image,
                inputs=[
                    face_file,
                    pose_file,
                    template_image_dict,
                    prompt,
                    negative_prompt,
                    style,
                    num_steps,
                    identitynet_strength_ratio,
                    adapter_strength_ratio,
                    guidance_scale,
                    seed,
                    enable_LCM,
                    enhance_face_region,
                ],
                outputs=[gallery, usage_tips],
            )

            enable_LCM.input(fn=toggle_lcm_ui, inputs=[enable_LCM], outputs=[num_steps, guidance_scale], queue=False)

        # gr.Examples(
        #     examples=get_example(),
        #     inputs=[face_file, prompt, style, negative_prompt],
        #     run_on_click=True,
        #     fn=run_for_examples,
        #     outputs=[gallery, usage_tips],
        #     cache_examples=True,
        # )

        gr.Markdown(article)

    demo.launch(server_name="0.0.0.0", server_port=8081)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    )
    parser.add_argument("--enable_LCM", type=bool, default=os.environ.get("ENABLE_LCM", False))

    args = parser.parse_args()

    main(args.pretrained_model_name_or_path, args.enable_LCM)

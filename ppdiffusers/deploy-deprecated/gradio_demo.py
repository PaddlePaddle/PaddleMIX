# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
import os

import cv2
import fastdeploy as fd
import gradio as gr
import numpy as np
import paddle
from paddlenlp.trainer.argparser import strtobool
from PIL import Image

from ppdiffusers import FastDeployStableDiffusionMegaPipeline


def create_paddle_inference_runtime(
    use_trt=False,
    dynamic_shape=None,
    use_fp16=False,
    use_bf16=False,
    device_id=0,
    disable_paddle_trt_ops=[],
    disable_paddle_pass=[],
    paddle_stream=None,
    workspace=None,
):
    assert not use_fp16 or not use_bf16, "use_fp16 and use_bf16 are mutually exclusive"
    option = fd.RuntimeOption()
    option.use_paddle_backend()
    if device_id == -1:
        option.use_cpu()
    else:
        option.use_gpu(device_id)
    if paddle_stream is not None and use_trt:
        option.set_external_raw_stream(paddle_stream)
    for pass_name in disable_paddle_pass:
        option.paddle_infer_option.delete_pass(pass_name)
    if use_bf16:
        option.paddle_infer_option.inference_precision = "bfloat16"
    if use_trt:
        option.paddle_infer_option.disable_trt_ops(disable_paddle_trt_ops)
        option.paddle_infer_option.enable_trt = True
        if workspace is not None:
            option.set_trt_max_workspace_size(workspace)
        if use_fp16:
            option.trt_option.enable_fp16 = True
        else:
            # Note(zhoushunjie): These four passes don't support fp32 now.
            # Remove this line of code in future.
            only_fp16_passes = [
                "trt_cross_multihead_matmul_fuse_pass",
                "trt_flash_multihead_matmul_fuse_pass",
                "preln_elementwise_groupnorm_act_pass",
                "elementwise_groupnorm_act_pass",
            ]
            for curr_pass in only_fp16_passes:
                option.paddle_infer_option.delete_pass(curr_pass)

        # Need to enable collect shape
        if dynamic_shape is not None:
            option.paddle_infer_option.collect_trt_shape = True
            for key, shape_dict in dynamic_shape.items():
                option.trt_option.set_shape(
                    key,
                    shape_dict["min_shape"],
                    shape_dict.get("opt_shape", None),
                    shape_dict.get("max_shape", None),
                )
    return option


def create_trt_runtime(workspace=(1 << 31), dynamic_shape=None, use_fp16=False, device_id=0):
    option = fd.RuntimeOption()
    option.use_trt_backend()
    option.use_gpu(device_id)
    if use_fp16:
        option.enable_trt_fp16()
    if workspace is not None:
        option.set_trt_max_workspace_size(workspace)
    if dynamic_shape is not None:
        for key, shape_dict in dynamic_shape.items():
            option.set_trt_input_shape(
                key,
                min_shape=shape_dict["min_shape"],
                opt_shape=shape_dict.get("opt_shape", None),
                max_shape=shape_dict.get("max_shape", None),
            )
    return option


def pipe_init(args):
    paddle.set_device(f"gpu:{args.device_id}")
    paddle_stream = paddle.device.cuda.current_stream(args.device_id).cuda_stream
    vae_in_channels = 4
    text_encoder_max_length = 77
    unet_max_length = text_encoder_max_length * 3  # lpw support max_length is 77x3
    min_image_size = 384
    max_image_size = 768
    hidden_states = 1024 if args.is_sd2_0 else 768
    unet_in_channels = 9 if args.task_name == "inpaint" else 4
    bs = 2

    text_encoder_dynamic_shape = {
        "input_ids": {
            "min_shape": [1, text_encoder_max_length],
            "max_shape": [1, text_encoder_max_length],
            "opt_shape": [1, text_encoder_max_length],
        }
    }

    vae_encoder_dynamic_shape = {
        "sample": {
            "min_shape": [1, 3, min_image_size, min_image_size],
            "max_shape": [1, 3, max_image_size, max_image_size],
            "opt_shape": [1, 3, min_image_size, min_image_size],
        }
    }

    vae_decoder_dynamic_shape = {
        "latent_sample": {
            "min_shape": [1, vae_in_channels, min_image_size // 8, min_image_size // 8],
            "max_shape": [1, vae_in_channels, max_image_size // 8, max_image_size // 8],
            "opt_shape": [1, vae_in_channels, min_image_size // 8, min_image_size // 8],
        }
    }

    unet_dynamic_shape = {
        "sample": {
            "min_shape": [
                1,
                unet_in_channels,
                min_image_size // 8,
                min_image_size // 8,
            ],
            "max_shape": [
                bs,
                unet_in_channels,
                max_image_size // 8,
                max_image_size // 8,
            ],
            "opt_shape": [
                2,
                unet_in_channels,
                min_image_size // 8,
                min_image_size // 8,
            ],
        },
        "timestep": {
            "min_shape": [1],
            "max_shape": [1],
            "opt_shape": [1],
        },
        "encoder_hidden_states": {
            "min_shape": [1, text_encoder_max_length, hidden_states],
            "max_shape": [bs, unet_max_length, hidden_states],
            "opt_shape": [2, text_encoder_max_length, hidden_states],
        },
    }
    # 4. Init runtime
    if args.backend == "tensorrt":
        runtime_options = dict(
            text_encoder=create_trt_runtime(
                dynamic_shape=text_encoder_dynamic_shape,
                use_fp16=args.use_fp16,
                device_id=args.device_id,
            ),
            vae_encoder=create_trt_runtime(
                dynamic_shape=vae_encoder_dynamic_shape,
                use_fp16=args.use_fp16,
                device_id=args.device_id,
            ),
            vae_decoder=create_trt_runtime(
                dynamic_shape=vae_decoder_dynamic_shape,
                use_fp16=args.use_fp16,
                device_id=args.device_id,
            ),
            unet=create_trt_runtime(
                dynamic_shape=unet_dynamic_shape,
                use_fp16=args.use_fp16,
                device_id=args.device_id,
            ),
        )
    elif args.backend == "paddle" or args.backend == "paddle_tensorrt":
        args.use_trt = args.backend == "paddle_tensorrt"
        runtime_options = dict(
            text_encoder=create_paddle_inference_runtime(
                use_trt=args.use_trt,
                dynamic_shape=text_encoder_dynamic_shape,
                use_fp16=args.use_fp16,
                use_bf16=args.use_bf16,
                device_id=args.device_id,
                disable_paddle_trt_ops=["arg_max", "range", "lookup_table_v2"],
                paddle_stream=paddle_stream,
            ),
            vae_encoder=create_paddle_inference_runtime(
                use_trt=args.use_trt,
                dynamic_shape=vae_encoder_dynamic_shape,
                use_fp16=args.use_fp16,
                use_bf16=args.use_bf16,
                device_id=args.device_id,
                paddle_stream=paddle_stream,
            ),
            vae_decoder=create_paddle_inference_runtime(
                use_trt=args.use_trt,
                dynamic_shape=vae_decoder_dynamic_shape,
                use_fp16=args.use_fp16,
                use_bf16=args.use_bf16,
                device_id=args.device_id,
                paddle_stream=paddle_stream,
            ),
            unet=create_paddle_inference_runtime(
                use_trt=args.use_trt,
                dynamic_shape=unet_dynamic_shape,
                use_fp16=args.use_fp16,
                use_bf16=args.use_bf16,
                device_id=args.device_id,
                paddle_stream=paddle_stream,
            ),
        )
    pipe = FastDeployStableDiffusionMegaPipeline.from_pretrained(
        args.model_dir,
        runtime_options=runtime_options,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="stable-diffusion-v1-5",
        help="The model directory of diffusion_model.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="text2img_img2img_inpaint_legacy",
        choices=[
            "text2img_img2img_inpaint_legacy",
            "inpaint",
            "controlnet_canny",
        ],
        help="The task can be one of [text2img_img2img_inpaint_legacy, inpaint, controlnet_canny]. ",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="paddle",
        # Note(zhoushunjie): Will support 'tensorrt' soon.
        choices=["paddle", "paddle_tensorrt"],
        help="The inference runtime backend of unet model and text encoder model.",
    )
    parser.add_argument("--use_fp16", type=strtobool, default=True, help="Whether to use FP16 mode")
    parser.add_argument("--use_bf16", type=strtobool, default=False, help="Whether to use BF16 mode")
    parser.add_argument("--device_id", type=int, default=0, help="The selected gpu id.")
    parser.add_argument(
        "--parse_prompt_type",
        type=str,
        default="lpw",
        choices=[
            "raw",
            "lpw",
        ],
        help="The parse_prompt_type can be one of [raw, lpw]. ",
    )
    parser.add_argument("--is_sd2_0", type=strtobool, default=False, help="Is sd2_0 model?")
    return parser.parse_args()


def get_canny_image(image):
    if image is not None:
        low_threshold = 100
        high_threshold = 200
        image = cv2.Canny(np.array(image), low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
    return image


def infer(
    taskname,
    image,
    mask,
    prompt,
    negative_prompt,
    steps,
    height,
    width,
    seed,
    strength,
    guidance_scale,
    scheduler,
    conditioning_scale,
):
    task_name = taskname
    fd_pipe.change_scheduler(scheduler)

    if int(seed) != -1:
        generator = paddle.Generator("cuda").manual_seed(seed)
    else:
        generator = None

    if image is not None:
        if isinstance(image, dict):
            image["image"] = cv2.resize(image["image"], (width, height))
            image["mask"] = cv2.resize(image["mask"], (width, height))
        else:
            image = cv2.resize(image, (width, height))
    if mask is not None:
        mask = cv2.resize(mask, (width, height))

    if task_name == "text2img":
        images = fd_pipe.text2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            parse_prompt_type=parse_prompt_type,
            infer_op_dict=infer_op_dict,
            generator=generator,
        )
    elif task_name == "img2img":
        images = fd_pipe.img2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=Image.fromarray(np.array(image)).convert("RGB"),
            num_inference_steps=steps,
            height=height,
            width=width,
            strength=strength,
            guidance_scale=guidance_scale,
            parse_prompt_type=parse_prompt_type,
            infer_op_dict=infer_op_dict,
            generator=generator,
        )
    elif task_name == "inpaint_legacy":
        if mask is not None:
            mask_image = mask
        else:
            mask_image = image["mask"]
        image = image["image"]
        images = fd_pipe.inpaint_legacy(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=Image.fromarray(np.array(image)).convert("RGB"),
            mask_image=Image.fromarray(mask_image).convert("RGB"),
            num_inference_steps=steps,
            height=height,
            width=width,
            strength=strength,
            guidance_scale=guidance_scale,
            parse_prompt_type=parse_prompt_type,
            infer_op_dict=infer_op_dict,
            generator=generator,
        )
    elif task_name == "inpaint":
        if mask is not None:
            mask_image = mask
        else:
            mask_image = image["mask"]
        image = image["image"]
        images = fd_pipe.inpaint(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=Image.fromarray(np.array(image)).convert("RGB"),
            mask_image=Image.fromarray(mask_image).convert("RGB"),
            num_inference_steps=steps,
            height=height,
            width=width,
            strength=strength,
            guidance_scale=guidance_scale,
            parse_prompt_type=parse_prompt_type,
            infer_op_dict=infer_op_dict,
            generator=generator,
        )

    elif task_name == "controlnet_canny":
        canny_image = Image.fromarray(mask)

        images = fd_pipe.text2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            parse_prompt_type=parse_prompt_type,
            controlnet_cond=canny_image,
            controlnet_conditioning_scale=conditioning_scale,
            infer_op_dict=infer_op_dict,
            generator=generator,
        )
    else:
        return gr.Error(f"task error! {task_name} not found ")

    return images[0][0]


scheduler_choices = [
    "pndm",
    "lms",
    "euler",
    "euler-ancestral",
    "preconfig-euler-ancestral",
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

# some param init
args = parse_arguments()
if "model_dir" and "task_name" in os.environ:
    args.model_dir = os.environ["model_dir"]
    args.task_name = os.environ["task_name"]

fd_pipe = pipe_init(args)
parse_prompt_type = args.parse_prompt_type
if args.backend == "paddle":
    print("When device is kunlunxin_xpu or backend is paddle, we will use `raw` infer op.")
    infer_op_mode = "raw"
else:
    infer_op_mode = "zero_copy_infer"
infer_op_dict = {
    "vae_encoder": infer_op_mode,
    "vae_decoder": infer_op_mode,
    "text_encoder": infer_op_mode,
    "unet": infer_op_mode,
}

with gr.Blocks() as demo:
    gr.Markdown("# FastDeploy Stablediffusion")
    if args.task_name == "text2img_img2img_inpaint_legacy":
        with gr.Tab("text2img"):
            with gr.Row():
                with gr.Column():
                    text2img_taskname = gr.State(value="text2img")
                    text2img_img = gr.State(value=None)
                    text2img_mask = gr.State(value=None)
                    text2img_prompt = gr.Textbox(label="正向描述词", lines=2)
                    text2img_negative_prompt = gr.Textbox(label="负向描述词", lines=2)
                    text2img_steps = gr.Slider(label="steps", minimum=1, maximum=60, step=1, value=20)
                    with gr.Row():
                        text2img_height = gr.Slider(label="height", minimum=384, maximum=768, step=8, value=512)
                        text2img_width = gr.Slider(label="width", minimum=384, maximum=768, step=8, value=512)
                    text2img_seed = gr.Textbox(label="seed", value="-1")
                    text2img_strength = gr.State(value=None)
                    text2img_guidance_scale = gr.Slider(
                        label="guidance_scale", minimum=1, maximum=30, step=0.5, value=7.5
                    )
                    text2img_scheduler = gr.Radio(label="采样方法", choices=scheduler_choices, value="ddim")
                    text2img_conditioning_scale = gr.State(value=None)
                with gr.Column():
                    text2img_output = gr.Image(type="numpy", label="result")
                    text2img_button = gr.Button("生成")
            text2img_button.click(
                fn=infer,
                inputs=[
                    text2img_taskname,
                    text2img_img,
                    text2img_mask,
                    text2img_prompt,
                    text2img_negative_prompt,
                    text2img_steps,
                    text2img_height,
                    text2img_width,
                    text2img_seed,
                    text2img_strength,
                    text2img_guidance_scale,
                    text2img_scheduler,
                    text2img_conditioning_scale,
                ],
                outputs=[text2img_output],
            )

        with gr.Tab("img2img"):
            with gr.Row():
                with gr.Column():
                    img2img_taskname = gr.State(value="img2img")
                    img2img_img = gr.Image(label="原图")
                    img2img_mask = gr.State(value=None)
                    img2img_prompt = gr.Textbox(label="请输入描述词", lines=2)
                    img2img_negative_prompt = gr.Textbox(label="负向描述词", lines=2)
                    img2img_steps = gr.Slider(label="steps", minimum=1, maximum=60, step=1, value=20)
                    with gr.Row():
                        img2img_height = gr.Slider(label="height", minimum=384, maximum=768, step=8, value=512)
                        img2img_width = gr.Slider(label="width", minimum=384, maximum=768, step=8, value=512)
                    img2img_seed = gr.Textbox(label="seed", value="-1")
                    img2img_strength = gr.Slider(
                        label="Denoising strength", minimum=0, maximum=1, step=0.01, value=0.75
                    )
                    img2img_guidance_scale = gr.Slider(
                        label="guidance_scale", minimum=1, maximum=30, step=0.5, value=7.5
                    )
                    img2img_scheduler = gr.Radio(label="采样方法", choices=scheduler_choices, value="ddim")
                    img2img_conditioning_scale = gr.State(value=None)
                with gr.Column():
                    img2img_output = gr.Image(type="numpy", label="result")
                    img2img_button = gr.Button("生成")
            img2img_button.click(
                fn=infer,
                inputs=[
                    img2img_taskname,
                    img2img_img,
                    img2img_mask,
                    img2img_prompt,
                    img2img_negative_prompt,
                    img2img_steps,
                    img2img_height,
                    img2img_width,
                    img2img_seed,
                    img2img_strength,
                    img2img_guidance_scale,
                    img2img_scheduler,
                    img2img_conditioning_scale,
                ],
                outputs=[img2img_output],
            )

        with gr.Tab("inpaint_legacy"):
            with gr.Row():
                with gr.Column():
                    inpaint_legacy_taskname = gr.State(value="inpaint_legacy")
                    inpaint_legacy_img = gr.ImageMask(label="传入原图并涂鸦mask")
                    inpaint_legacy_mask = gr.Image(label="重绘mask（可选，若不涂鸦则需要传入）", image_mode="L")
                    inpaint_legacy_prompt = gr.Textbox(label="请输入正向描述词", lines=2)
                    inpaint_legacy_negative_prompt = gr.Textbox(label="负向描述词", lines=2)
                    inpaint_legacy_steps = gr.Slider(label="steps", minimum=1, maximum=60, step=1, value=20)
                    with gr.Row():
                        inpaint_legacy_height = gr.Slider(label="height", minimum=384, maximum=768, step=8, value=512)
                        inpaint_legacy_width = gr.Slider(label="width", minimum=384, maximum=768, step=8, value=512)
                    inpaint_legacy_seed = gr.Textbox(label="seed", value="-1")
                    inpaint_legacy_strength = gr.Slider(
                        label="Denoising strength", minimum=0, maximum=1, step=0.01, value=0.75
                    )
                    inpaint_legacy_guidance_scale = gr.Slider(
                        label="guidance_scale", minimum=1, maximum=30, step=0.5, value=7.5
                    )
                    inpaint_legacy_scheduler = gr.Radio(label="采样方法", choices=scheduler_choices, value="ddim")
                    inpaint_legacy_conditioning_scale = gr.State(value=None)
                with gr.Column():
                    inpaint_legacy_output = gr.Image(type="numpy", label="result")
                    inpaint_legacy_button = gr.Button("生成")
            inpaint_legacy_button.click(
                fn=infer,
                inputs=[
                    inpaint_legacy_taskname,
                    inpaint_legacy_img,
                    inpaint_legacy_mask,
                    inpaint_legacy_prompt,
                    inpaint_legacy_negative_prompt,
                    inpaint_legacy_steps,
                    inpaint_legacy_height,
                    inpaint_legacy_width,
                    inpaint_legacy_seed,
                    inpaint_legacy_strength,
                    inpaint_legacy_guidance_scale,
                    inpaint_legacy_scheduler,
                    inpaint_legacy_conditioning_scale,
                ],
                outputs=[inpaint_legacy_output],
            )

    elif args.task_name == "inpaint":
        with gr.Tab("inpaint"):
            with gr.Row():
                with gr.Column():
                    inpaint_taskname = gr.State(value="inpaint")
                    inpaint_img = gr.ImageMask(label="传入原图并涂鸦mask")
                    inpaint_mask = gr.Image(label="重绘mask（可选，若不涂鸦则需要传入）", image_mode="L")
                    inpaint_prompt = gr.Textbox(label="请输入正向描述词", lines=2)
                    inpaint_negative_prompt = gr.Textbox(label="负向描述词", lines=2)
                    inpaint_steps = gr.Slider(label="steps", minimum=1, maximum=60, step=1, value=20)
                    with gr.Row():
                        inpaint_height = gr.Slider(label="height", minimum=384, maximum=768, step=8, value=512)
                        inpaint_width = gr.Slider(label="width", minimum=384, maximum=768, step=8, value=512)
                    inpaint_seed = gr.Textbox(label="seed", value="-1")
                    inpaint_strength = gr.Slider(
                        label="Denoising strength", minimum=0, maximum=1, step=0.01, value=0.75
                    )
                    inpaint_guidance_scale = gr.Slider(
                        label="guidance_scale", minimum=1, maximum=30, step=0.5, value=7.5
                    )
                    inpaint_scheduler = gr.Radio(label="采样方法", choices=scheduler_choices, value="ddim")
                    inpaint_conditioning_scale = gr.State(value=None)
                with gr.Column():
                    inpaint_output = gr.Image(type="numpy", label="result")
                    inpaint_button = gr.Button("生成")

            inpaint_button.click(
                fn=infer,
                inputs=[
                    inpaint_taskname,
                    inpaint_img,
                    inpaint_mask,
                    inpaint_prompt,
                    inpaint_negative_prompt,
                    inpaint_steps,
                    inpaint_height,
                    inpaint_width,
                    inpaint_seed,
                    inpaint_strength,
                    inpaint_guidance_scale,
                    inpaint_scheduler,
                    inpaint_conditioning_scale,
                ],
                outputs=[inpaint_output],
            )

    elif args.task_name == "controlnet_canny":
        with gr.Tab("controlnet_canny"):
            with gr.Row():
                with gr.Column():
                    controlnet_canny_taskname = gr.State(value="controlnet_canny")
                    controlnet_canny_img = gr.Image(label="canny参考图")
                    controlnet_canny_mask = gr.Image(label="canny图（可选传入）")
                    controlnet_canny_prompt = gr.Textbox(label="请输入正向描述词", lines=2)
                    controlnet_canny_negative_prompt = gr.Textbox(label="负向描述词", lines=2)
                    controlnet_canny_steps = gr.Slider(label="steps", minimum=1, maximum=60, step=1, value=20)
                    with gr.Row():
                        controlnet_canny_height = gr.Slider(
                            label="height", minimum=384, maximum=768, step=8, value=512
                        )
                        controlnet_canny_width = gr.Slider(label="width", minimum=384, maximum=768, step=8, value=512)
                    controlnet_canny_seed = gr.Textbox(label="seed", value="-1")
                    controlnet_canny_strength = gr.Slider(
                        label="Denoising strength", minimum=0, maximum=1, step=0.01, value=0.75
                    )
                    controlnet_canny_guidance_scale = gr.Slider(
                        label="guidance_scale", minimum=1, maximum=30, step=0.5, value=7.5
                    )
                    controlnet_canny_scheduler = gr.Radio(label="采样方法", choices=scheduler_choices, value="ddim")
                    controlnet_canny_conditioning_scale = gr.Slider(
                        label="conditioning_scale", minimum=0, maximum=2, step=0.05, value=1
                    )
                with gr.Column():
                    controlnet_canny_output = gr.Image(type="numpy", label="result")
                    controlnet_canny_button = gr.Button("生成")
            controlnet_canny_img.change(
                fn=get_canny_image, inputs=[controlnet_canny_img], outputs=[controlnet_canny_mask]
            )
            controlnet_canny_button.click(
                fn=infer,
                inputs=[
                    controlnet_canny_taskname,
                    controlnet_canny_img,
                    controlnet_canny_mask,
                    controlnet_canny_prompt,
                    controlnet_canny_negative_prompt,
                    controlnet_canny_steps,
                    controlnet_canny_height,
                    controlnet_canny_width,
                    controlnet_canny_seed,
                    controlnet_canny_strength,
                    controlnet_canny_guidance_scale,
                    controlnet_canny_scheduler,
                    controlnet_canny_conditioning_scale,
                ],
                outputs=[controlnet_canny_output],
            )

if __name__ == "__main__":
    demo.launch(show_error=True)

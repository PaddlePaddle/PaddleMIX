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

import os
import subprocess

import gradio as gr
from utils import (
    FeedbackManager,
    add_watermark,
    extract_first_frame,
    process_input_video,
)

env = os.environ
env["CUDA_VISIBLE_DEVICES"] = "1"


def generate_masked_video(input_video, subject_prompt):

    masked_video_path = input_video.replace(".mp4", "_masked.mp4")
    mask_video_path = input_video.replace(".mp4", "_mask.mp4")
    sam_command = [
        "python",
        "ppdiffusers/examples/ppvctrl/anchor/extract_mask.py",
        "--sam2_config",
        "configs/sam2.1_hiera_l.yaml",
        "--sam2_checkpoint",
        "ppdiffusers/examples/ppvctrl/weights/sam2/sam2.1_hiera_large.pdparams",
        "--input_path",
        input_video,
        "--control_video_path",
        masked_video_path,
        "--mask_video_path",
        mask_video_path,
        "--prompt",
        subject_prompt,
    ]
    subprocess.run(sam_command, check=True, cwd="/".join(os.getcwd().split("/")[:-3]), env=env)
    return masked_video_path, mask_video_path


def cogvideox_5b_i2v_vctrl_process(
    masked_video,
    mask_video,
    first_mask_image_path,
    first_rgb_image_path,
    prompt,
    controlnet_seed,
    controlnet_num_inference_steps,
    controlnet_guidance_scale,
    controlnet_conditioning_scale,
    num_inference_steps,
    conditioning_scale,
    guidance_scale,
    low_threshold,
    high_threshold,
    max_frame,
    use_controlnet,
):
    use_controlnet = True if use_controlnet == "yes" else False
    controlnet_command = [
        "python",
        "tools/controlnet_gradio.py",
        "--image_path",
        first_rgb_image_path,
        "--mask_path",
        first_mask_image_path,
        "--prompt",
        prompt,
        "--task",
        "mask",
        "--reverse_mask",
        "True",
        "--controlnet_seed",
        str(controlnet_seed),
        "--controlnet_num_inference_steps",
        str(controlnet_num_inference_steps),
        "--controlnet_guidance_scale",
        str(controlnet_guidance_scale),
        "--controlnet_conditioning_scale",
        str(controlnet_conditioning_scale),
    ]

    output_dir = "infer_outputs/mask2video/i2v"
    vctrl_command = [
        "python",
        "infer_cogvideox_i2v_vctrl_cli.py",
        "--pretrained_model_name_or_path",
        "paddlemix/cogvideox-5b-i2v-vctrl",
        "--vctrl_path",
        "weights/mask/vctrl_5b_i2v_mask.pdparams",
        "--vctrl_config",
        "vctrl_configs/cogvideox_5b_i2v_vctrl_config.json",
        "--control_video_path",
        masked_video,
        "--control_mask_video_path",
        mask_video,
        "--output_dir",
        output_dir,
        "--task",
        "mask",
        "--ref_image_path",
        first_rgb_image_path.replace(".jpg", "_controlnet.jpg") if use_controlnet else first_rgb_image_path,
        "--prompt_path",
        prompt,
        "--width",
        "720",
        "--height",
        "480",
        "--max_frame",
        str(max_frame),
        "--guidance_scale",
        str(guidance_scale),
        "--num_inference_steps",
        str(num_inference_steps),
        "--conditioning_scale",
        str(conditioning_scale),
    ]
    if use_controlnet:
        subprocess.run(controlnet_command, check=True, env=env)
    subprocess.run(vctrl_command, check=True, env=env)
    return (
        add_watermark(os.path.join(output_dir, "output.mp4")),
        os.path.join(output_dir, "origin_predict.mp4"),
        os.path.join(output_dir, "test_1.mp4"),
        first_rgb_image_path.replace(".jpg", "_controlnet.jpg") if use_controlnet else first_rgb_image_path,
    )


def cogvideox_5b_vctrl_process(
    masked_video,
    mask_video,
    first_rgb_image_path,
    prompt,
    num_inference_steps,
    conditioning_scale,
    guidance_scale,
    low_threshold,
    high_threshold,
    max_frame,
    use_controlnet,
):
    output_dir = "infer_outputs/mask2video/t2v"
    vctrl_command = [
        "python",
        "infer_cogvideox_t2v_vctrl_cli.py",
        "--pretrained_model_name_or_path",
        "paddlemix/cogvideox-5b-vctrl",
        "--vctrl_path",
        "weights/mask/vctrl_5b_t2v_mask.pdparams",
        "--vctrl_config",
        "vctrl_configs/cogvideox_5b_vctrl_config.json",
        "--control_video_path",
        masked_video,
        "--control_mask_video_path",
        mask_video,
        "--output_dir",
        output_dir,
        "--task",
        "mask",
        "--prompt_path",
        prompt,
        "--width",
        "720",
        "--height",
        "480",
        "--max_frame",
        str(max_frame),
        "--guidance_scale",
        str(guidance_scale),
        "--num_inference_steps",
        str(num_inference_steps),
        "--conditioning_scale",
        str(conditioning_scale),
    ]
    subprocess.run(vctrl_command, check=True, env=env)
    return (
        add_watermark(os.path.join(output_dir, "output.mp4")),
        os.path.join(output_dir, "origin_predict.mp4"),
        os.path.join(output_dir, "test_1.mp4"),
        first_rgb_image_path,
    )


def process(
    model,
    input_video,
    subject_prompt,
    prompt,
    controlnet_seed,
    controlnet_num_inference_steps,
    controlnet_guidance_scale,
    controlnet_conditioning_scale,
    num_inference_steps,
    conditioning_scale,
    guidance_scale,
    low_threshold,
    high_threshold,
    max_frame,
    use_controlnet,
):
    input_video = process_input_video(input_video, task="mask", height=480, width=720)
    masked_video, mask_video = generate_masked_video(input_video, subject_prompt)
    first_mask_image_path = os.path.join(os.path.dirname(mask_video), "first_mask_image.jpg")
    extract_first_frame(mask_video, first_mask_image_path)
    first_rgb_image_path = os.path.join(os.path.dirname(input_video), "first_rgb_image.jpg")
    extract_first_frame(input_video, first_rgb_image_path, convert_rgb=True)

    if model == "cogvideox_5b_i2v_vctrl":
        output = cogvideox_5b_i2v_vctrl_process(
            masked_video,
            mask_video,
            first_mask_image_path,
            first_rgb_image_path,
            prompt,
            controlnet_seed,
            controlnet_num_inference_steps,
            controlnet_guidance_scale,
            controlnet_conditioning_scale,
            num_inference_steps,
            conditioning_scale,
            guidance_scale,
            low_threshold,
            high_threshold,
            max_frame,
            use_controlnet,
        )
    elif model == "cogvideox_5b_vctrl":
        output = cogvideox_5b_vctrl_process(
            masked_video,
            mask_video,
            first_rgb_image_path,
            prompt,
            num_inference_steps,
            conditioning_scale,
            guidance_scale,
            low_threshold,
            high_threshold,
            max_frame,
            use_controlnet,
        )
    else:
        raise ValueError(f"Invalid model name: {model}")

    current_output_dir = os.path.dirname(output[0])
    feedback_mgr.store_conversation(prompt, input_video, current_output_dir)
    feedback_mgr.mark_chat_complete()

    return output


def get_feedback_stats():
    """从 FeedbackManager 获取当前反馈数据"""
    return f"{feedback_mgr.feedback_data['likes']} 👍 | {feedback_mgr.feedback_data['dislikes']} 👎"


feedback_mgr = FeedbackManager(os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback"))

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## 🤖 PP-VCtrl: Multimodal Video Editing (Mask) Demo")
    with gr.Row():
        gr.Markdown(
            "📚 原始模型来自 [PaddleMIX](https://github.com/PaddlePaddle/PaddleMIX) （🌟 一个基于飞桨PaddlePaddle框架构建的多模态大模型套件）"
        )
    with gr.Row():
        gr.Markdown("PP-VCtrl是一个统一的视频生成控制模型：")
    with gr.Row():
        gr.Markdown("- 它通过引入辅助条件编码器，实现了对各类控制信号的灵活接入和精确控制，同时保持了高效的计算性能")
    with gr.Row():
        gr.Markdown("- 它可以高效地应用在各类视频生成场景，尤其是人物动画、场景转换、视频编辑等需要精确控制的任务。")
    with gr.Row():
        gr.Markdown("**（基于Mask）视频编辑的使用方法：**")
    with gr.Row():
        gr.Markdown("- 上传视频")
    with gr.Row():
        gr.Markdown("- 输入subject prompt，表示想要替换的视频中的目标，例如person，sky")
    with gr.Row():
        gr.Markdown("- 输入prompt描述新生成的视频，参考样例中prompt")
    with gr.Row():
        gr.Markdown("- 点云Run进行生成")
    with gr.Row():
        gr.Markdown('- 点击"👍 Like"或"👎 Dislike"对模型回答进行反馈')
    with gr.Row():
        gr.Markdown("**注意事项：**")
    with gr.Row():
        gr.Markdown("- 视频要求：为获得最好的生成效果，建议提供720（宽）*480（高）视频，视频长度在2s-5s；视频名称不能存在中文，空格和特殊符号")
    with gr.Row():
        gr.Markdown("- subject prompt要求：subject目标应在视频中出现，并且最好为主要目标")
    with gr.Row():
        gr.Markdown("- prompt要求：和原始视频具有一定的相关性；描述尽可能详细，单词数量应<80个单词")
    with gr.Row():
        gr.Markdown("- 运行时长：大约在20min，请耐心等待")
    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Upload Video")
            subject_prompt = gr.Textbox(label="Subject Prompt")
            prompt = gr.Textbox(label="Prompt")
            model = gr.Dropdown(
                label="Select Model",
                choices=["cogvideox_5b_vctrl", "cogvideox_5b_i2v_vctrl"],
                value="cogvideox_5b_i2v_vctrl",
                interactive=True,
            )
            use_controlnet = gr.Dropdown(
                label="Use ControlNet with ‘cogvideox_5b_i2v_vctrl’",
                choices=["yes", "no"],
                value="yes",
                interactive=True,
            )
            run_button = gr.Button(value="Run")
            with gr.Accordion("Advanced options", open=False):
                controlnet_seed = gr.Slider(
                    label="controlnet Seed",
                    minimum=0,
                    maximum=100,
                    value=42,
                    step=1,
                )
                controlnet_num_inference_steps = gr.Slider(
                    label="controlnet Steps",
                    minimum=1,
                    maximum=100,
                    value=28,
                    step=1,
                )
                controlnet_guidance_scale = gr.Slider(
                    label="controlnet Guidance Scale",
                    minimum=0.1,
                    maximum=30.0,
                    value=7.0,
                    step=0.1,
                )
                controlnet_conditioning_scale = gr.Slider(
                    label="controlnet Conditioning Scale",
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.1,
                )
                conditioning_scale = gr.Slider(
                    label="Control Strength",
                    minimum=0.0,
                    maximum=5.0,
                    value=1.0,
                    step=0.05,
                )
                low_threshold = gr.Slider(
                    label="Canny low threshold",
                    minimum=1,
                    maximum=255,
                    value=100,
                    step=1,
                )
                high_threshold = gr.Slider(
                    label="Canny high threshold",
                    minimum=1,
                    maximum=255,
                    value=200,
                    step=1,
                )
                num_inference_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=0.1,
                    maximum=30.0,
                    value=9.0,
                    step=0.1,
                )
                max_frame = gr.Slider(
                    label="Max frames",
                    minimum=1,
                    maximum=100,
                    value=49,
                    step=1,
                )
        with gr.Column():
            display_video = gr.Video(label="Presentation video")
            generated_video = gr.Video(label="Generated Video")
            compared_video = gr.Video(label="Compared Video")
            ref_image = gr.Image(label="Reference Image")

    with gr.Row():
        feedback_display = gr.Textbox(value=get_feedback_stats(), label="📊 Feedback Stats", interactive=False)

    with gr.Row():
        like_button = gr.Button("👍 Like")
        dislike_button = gr.Button("👎 Dislike")

    like_button.click(fn=lambda: feedback_mgr.update_feedback("like"), outputs=feedback_display)
    dislike_button.click(fn=lambda: feedback_mgr.update_feedback("dislike"), outputs=feedback_display)

    ips = [
        model,
        input_video,
        subject_prompt,
        prompt,
        controlnet_seed,
        controlnet_num_inference_steps,
        controlnet_guidance_scale,
        controlnet_conditioning_scale,
        num_inference_steps,
        conditioning_scale,
        guidance_scale,
        low_threshold,
        high_threshold,
        max_frame,
        use_controlnet,
    ]

    run_button.click(fn=process, inputs=ips, outputs=[display_video, generated_video, compared_video, ref_image])

block.launch(server_name="0.0.0.0", server_port=8233, share=True)

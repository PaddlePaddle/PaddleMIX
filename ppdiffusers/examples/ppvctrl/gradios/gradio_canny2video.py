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
import subprocess

import cv2
import gradio as gr
import moviepy.editor as mpy
import numpy as np
from decord import VideoReader, cpu
from utils import FeedbackManager, add_watermark, process_input_video

env = os.environ
env["CUDA_VISIBLE_DEVICES"] = "3"


def extract_canny(video_path, low_threshold=100, high_threshold=255):
    output_video_path = video_path.replace(".mp4", "_canny.mp4")
    first_rgb_image_path = video_path.replace(".mp4", "_first_frame_rgb.jpg")
    first_canny_image_path = video_path.replace(".mp4", "_first_frame_canny.jpg")
    vr = VideoReader(video_path, ctx=cpu(0))
    target_fps = vr.get_avg_fps()
    first_frame = vr[0].asnumpy()
    first_frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(first_rgb_image_path, first_frame_bgr)
    print(f"第一帧rgb已保存到 {first_rgb_image_path}")
    frames = []
    for i in range(len(vr)):
        frame = vr[i].asnumpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.Canny(frame, low_threshold, high_threshold)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        frames.append(frame)
    cv2.imwrite(first_canny_image_path, frames[0])
    print(f"第一帧canny已保存到 {first_canny_image_path}")

    def downsample_fps(frames, target_fps=8):
        length = len(frames)
        avg_fps = float(vr.get_avg_fps())
        interval_frame = max(1, int(round(avg_fps / target_fps)))
        bg_frame_id = 0
        frame_ids = list(range(bg_frame_id, length, interval_frame))
        len_frame_ids = len(frame_ids)
        len_frame_ids = len_frame_ids - (len_frame_ids - 1) % 8
        frame_ids = frame_ids[:len_frame_ids]
        frames = [frames[frame_id] for frame_id in frame_ids]
        return frames

    frames = downsample_fps(frames, target_fps)

    if len(frames) < 49:
        height, width, channels = frames[0].shape if frames else (480, 640, 3)
        blank_frame = np.zeros((height, width, channels), dtype=np.uint8)
        frames_to_add = 49 - len(frames)
        for _ in range(frames_to_add):
            frames.append(blank_frame.copy())

    def write_video(frames, output_video_path):
        output_fps = target_fps

        def make_frame(t):
            return frames[int(t * output_fps)]

        clip = mpy.VideoClip(make_frame, duration=len(frames) / output_fps)

        clip.write_videofile(output_video_path, fps=output_fps)

    write_video(frames, output_video_path)

    print(f"视频处理完成，已保存到 {output_video_path}")

    return output_video_path, first_rgb_image_path, first_canny_image_path


def cogvideox_5b_i2v_vctrl_process(
    input_video,
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
    input_video = process_input_video(input_video, task="canny", height=480, width=720)

    canny_video_path, first_rgb_image_path, first_canny_image_path = extract_canny(
        input_video, low_threshold, high_threshold
    )
    controlnet_command = [
        "python",
        "tools/controlnet_gradio.py",
        "--image_path",
        first_canny_image_path,
        "--prompt",
        prompt,
        "--task",
        "canny",
        "--controlnet_seed",
        str(controlnet_seed),
        "--controlnet_num_inference_steps",
        str(controlnet_num_inference_steps),
        "--controlnet_guidance_scale",
        str(controlnet_guidance_scale),
        "--controlnet_conditioning_scale",
        str(controlnet_conditioning_scale),
    ]
    vctrl_command = [
        "python",
        "infer_cogvideox_i2v_vctrl_cli.py",
        "--pretrained_model_name_or_path",
        "paddlemix/cogvideox-5b-i2v-vctrl",
        "--vctrl_path",
        "weights/canny/vctrl_canny_5b_i2v_vctrl-tiny.pdparams",
        "--vctrl_config",
        "vctrl_configs/cogvideox_5b_i2v_vctrl_tiny_config.json",
        "--control_video_path",
        canny_video_path,
        "--output_dir",
        "infer_outputs/canny2video/i2v",
        "--task",
        "canny",
        "--ref_image_path",
        first_canny_image_path.replace(".jpg", "_controlnet.jpg") if use_controlnet else first_rgb_image_path,
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
        add_watermark("infer_outputs/canny2video/i2v/output.mp4"),
        "infer_outputs/canny2video/i2v/origin_predict.mp4",
        "infer_outputs/canny2video/i2v/test_1.mp4",
        first_canny_image_path.replace(".jpg", "_controlnet.jpg") if use_controlnet else first_rgb_image_path,
    )


def cogvideox_5b_vctrl_process(
    input_video,
    prompt,
    num_inference_steps,
    conditioning_scale,
    guidance_scale,
    low_threshold,
    high_threshold,
    max_frame,
):
    input_video = process_input_video(input_video, task="canny", height=480, width=720)
    canny_video_path, _, _ = extract_canny(input_video)
    vctrl_command = [
        "python",
        "infer_cogvideox_t2v_vctrl_cli.py",
        "--pretrained_model_name_or_path",
        "paddlemix/cogvideox-5b-vctrl",
        "--vctrl_path",
        "weights/canny/vctrl_canny_5b_t2v.pdparams",
        "--vctrl_config",
        "vctrl_configs/cogvideox_5b_vctrl_config.json",
        "--control_video_path",
        canny_video_path,
        "--output_dir",
        "infer_outputs/canny2video/t2v",
        "--task",
        "canny",
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
        add_watermark("infer_outputs/canny2video/t2v/output.mp4"),
        "infer_outputs/canny2video/t2v/origin_predict.mp4",
        "infer_outputs/canny2video/t2v/test_1.mp4",
        None,
    )


def process(
    model,
    input_video,
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
    if model == "cogvideox_5b_i2v_vctrl":
        output = cogvideox_5b_i2v_vctrl_process(
            input_video,
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
            input_video,
            prompt,
            num_inference_steps,
            conditioning_scale,
            guidance_scale,
            low_threshold,
            high_threshold,
            max_frame,
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
        gr.Markdown("## 🤖 PP-VCtrl: Multimodal Scene Transitions (Canny) Demo")
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
        gr.Markdown("**（基于Canny）场景转换的使用方法：**")
    with gr.Row():
        gr.Markdown("- 上传视频（用于提取边缘）")
    with gr.Row():
        gr.Markdown("- 输入prompt描述新生成的视频，例如样例中将场景转换为snow mountain")
    with gr.Row():
        gr.Markdown("- 点云Run进行生成")
    with gr.Row():
        gr.Markdown('- 点击"👍 Like"或"👎 Dislike"对模型回答进行反馈')
    with gr.Row():
        gr.Markdown("**注意事项：**")
    with gr.Row():
        gr.Markdown("- 视频要求：为获得最好的生成效果，建议提供720（宽）*480（高）视频，视频长度在2s-5s；视频名称不能存在中文，空格和特殊符号")
    with gr.Row():
        gr.Markdown("- prompt要求：和原始视频具有一定的相关性；描述尽可能详细，单词数量应<80个单词")
    with gr.Row():
        gr.Markdown("- 运行时长：大约在10min，请耐心等待")

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Upload Video")
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
                    label="Controlnet Seed",
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                )
                controlnet_num_inference_steps = gr.Slider(
                    label="Controlnet Steps",
                    minimum=1,
                    maximum=100,
                    value=28,
                    step=1,
                )
                controlnet_guidance_scale = gr.Slider(
                    label="Controlnet Guidance Scale",
                    minimum=0.1,
                    maximum=30.0,
                    value=7.0,
                    step=0.1,
                )
                controlnet_conditioning_scale = gr.Slider(
                    label="Controlnet Conditioning Scale",
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

block.launch(server_name="0.0.0.0", server_port=8513, share=True)

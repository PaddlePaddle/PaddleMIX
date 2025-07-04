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

import json
import os
import random
import time
from datetime import datetime

import cv2
import moviepy.editor as mpy
from decord import VideoReader
from moviepy.editor import CompositeVideoClip, ImageClip, VideoFileClip
from PIL import Image, ImageDraw, ImageFont


def add_watermark(video_path):
    """
    给上传的视频添加水印并返回处理后的视频路径。

    参数:
    video_path (str): 上传的视频文件路径。

    返回:
    str: 处理后的视频文件路径。
    """
    clip = VideoFileClip(video_path)

    txt = "PaddleMIX"
    font = ImageFont.load_default()
    img_width, img_height = 200, 50
    img = Image.new("RGBA", (img_width, img_height), (255, 255, 255, 0))
    d = ImageDraw.Draw(img)

    text_bbox = d.textbbox((0, 0), txt, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (img_width - text_width) / 2
    text_y = (img_height - text_height) / 2
    d.text((text_x, text_y), txt, font=font, fill=(255, 255, 255, 128))

    img_path = "watermark.png"
    img.save(img_path)

    position = (clip.w - img_width - 10, clip.h - img_height - 10)
    watermark = ImageClip(img_path).set_position(position).set_duration(clip.duration)
    video = CompositeVideoClip([clip, watermark])
    output_path = os.path.join(os.path.dirname(video_path), "watermarked_" + os.path.basename(video_path))
    video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    os.remove(img_path)

    return output_path


class FeedbackManager:
    def __init__(self, save_path, save_interval=30):
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.save_json_path = os.path.join(self.save_path, "feedback.json")
        self.save_interval = save_interval
        self.feedback_data = self.load_feedback()
        self.last_save_time = time.time()

        # Track current conversation state
        self.current_chat_completed = False
        self.feedback_given = False

        print(f"Feedback will be saved to: {self.save_path}")

    def load_feedback(self):
        if os.path.exists(self.save_json_path):
            try:
                with open(self.save_json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading feedback file: {e}")
                return {"likes": 0, "dislikes": 0, "dislike_details": []}
        return {"likes": 0, "dislikes": 0, "dislike_details": []}

    def save_prompt_txt(self, prompt, save_path):
        with open(os.path.join(save_path, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)

    def save_feedback(self, current_timestamp):
        try:
            with open(self.save_json_path, "w", encoding="utf-8") as f:
                json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
            self.last_save_time = time.time()
            print(f"Feedback saved successfully at {current_timestamp}")

            current_save_path = os.path.join(self.save_path, current_timestamp)
            if not os.path.exists(current_save_path):
                os.mkdir(current_save_path)

            os.system("cp {} {}".format(self.current_input_video, current_save_path))
            os.system("cp {} {}".format(os.path.join(self.current_output_dir, "test_1.mp4"), current_save_path))
            self.save_prompt_txt(self.current_prompt, current_save_path)

        except Exception as e:
            print(f"Error saving feedback: {e}")

    def store_conversation(self, prompt, input_video, output_dir):
        """Store current conversation details"""
        self.current_prompt = prompt
        self.current_input_video = input_video
        self.current_output_dir = output_dir

    def mark_chat_complete(self):
        """Mark current conversation as completed"""
        self.current_chat_completed = True
        self.feedback_given = False

    def can_give_feedback(self):
        """Check if feedback can be given"""
        if not self.current_chat_completed:
            return False, "请先完成一次对话再进行反馈"
        if self.feedback_given:
            self.current_chat_completed = False
            return False, "该对话已经收到过反馈"
        return True, ""

    def update_feedback(self, feedback_type):
        can_feedback, message = self.can_give_feedback()
        if not can_feedback:
            return f"{self.feedback_data['likes']} 👍 | {self.feedback_data['dislikes']} 👎\n{message}"

        if feedback_type in ["like", "dislike"]:
            key = "likes" if feedback_type == "like" else "dislikes"
            self.feedback_data[key] += 1
            current_timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            # Store detailed information for dislikes
            if feedback_type == "dislike" and all(
                [self.current_prompt, self.current_input_video, self.current_output_dir]
            ):
                dislike_detail = {
                    "timestamp": current_timestamp,
                }
                # for key, value in meta_info.items():
                #     dislike_detail[key] = value
                self.feedback_data["dislike_details"].append(dislike_detail)

            self.feedback_given = True

            # if time.time() - self.last_save_time >= self.save_interval:
            #     self.save_feedback(current_timestamp)
            self.save_feedback(current_timestamp)
            return f"{self.feedback_data['likes']} 👍 | {self.feedback_data['dislikes']} 👎"
        return ""


def extract_first_frame(video_path, first_frame_path, convert_rgb=False):
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0))
    first_frame = vr[0].asnumpy()
    if convert_rgb:
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(first_frame_path, first_frame)
    print(f"第一帧已保存到 {first_frame_path}")


def compute_crop_params(vid_width, vid_height, height, width):
    def rand_trunc_norm(center, min_val, max_val, std_dev):
        """
        从截断的正态分布中生成随机整数
        """
        while True:
            value = int(random.gauss(center, std_dev))
            if min_val <= value <= max_val:
                return value

    if vid_height / vid_width > height / width:
        crop_width = vid_width
        crop_height = int(vid_width * height / width)
        h_max = vid_height - crop_height

        center_h = h_max // 2

        # h0 = rand_trunc_norm(center_h, min_h, max_h, std_dev)
        h0 = center_h
        w0 = 0
    else:
        crop_width = int(vid_height * width / height)
        crop_height = vid_height
        w_max = vid_width - crop_width

        center_w = w_max // 2

        # w0 = rand_trunc_norm(center_w, min_w, max_w, std_dev)
        w0 = center_w
        h0 = 0

    return w0, h0, crop_width, crop_height


def process_input_video(video_path, task, height, width):
    dynamic_fps = True
    target_fps = 30
    sample_n_frames = 49

    vr = VideoReader(video_path)  # [b, h, w, c]
    avg_fps = float(vr.get_avg_fps())
    length = len(vr)
    if dynamic_fps:
        max_interval_frame_candidate = length // sample_n_frames
        max_interval_frame_candidate = min(avg_fps // 6, max_interval_frame_candidate)
        interval_frame = random.randint(1, max_interval_frame_candidate)
    else:
        if target_fps > 0:
            # target_fps = self.target_fps
            interval_frame = max(1, int(round(avg_fps / target_fps)))
        else:
            # init_interval_frame = self.interval_frame
            init_interval_frame = None
            if init_interval_frame <= 0:
                init_interval_frame = random.randint(2, 7)
            assert (
                length >= sample_n_frames
            ), "Too short video... The minimum number of frames should be {}, but got {}".format(
                sample_n_frames, length
            )
            interval_frame = min(init_interval_frame, int(length // sample_n_frames))

    segment_length = interval_frame * sample_n_frames
    assert length >= segment_length, "Too short video... The minimum number of frames should be {}, but got {}".format(
        segment_length, length
    )
    if task == "mask":
        bg_frame_id = 0
    else:
        bg_frame_id = random.randint(0, length - segment_length)
    frame_ids = list(range(bg_frame_id, bg_frame_id + segment_length, interval_frame))
    # pixel_values = np.array([vr[frame_id].asnumpy() for frame_id in frame_ids]) # [b, h, w, c], [0, 255]
    # pixel_values = numpy_to_pt(pixel_values) # [b, c, h, w], [0, 1]
    pixel_values = [vr[frame_id].asnumpy() for frame_id in frame_ids]
    vid_width = pixel_values[0].shape[1]
    vid_height = pixel_values[0].shape[0]
    w0, h0, crop_width, crop_height = compute_crop_params(vid_width, vid_height, height, width)

    for i in range(len(pixel_values)):
        pixel_values[i] = pixel_values[i][h0 : h0 + crop_height, w0 : w0 + crop_width, :]

    def write_video(frames, output_video_path):
        output_fps = target_fps

        def make_frame(t):
            return frames[int(t * output_fps)]

        clip = mpy.VideoClip(make_frame, duration=len(frames) / output_fps)
        clip.write_videofile(output_video_path, fps=output_fps)

    output_video_path = video_path.replace(".mp4", "_processed.mp4")
    write_video(pixel_values, output_video_path)
    print("processed video is written to ", output_video_path)
    return output_video_path


def process_reference_image_pose(image_path, height, width):
    image = cv2.imread(image_path)
    vid_width = image.shape[1]
    vid_height = image.shape[0]
    w0, h0, crop_width, crop_height = compute_crop_params(vid_width, vid_height, height, width)
    print(w0, h0, crop_width, crop_height)
    image = image[h0 : h0 + crop_height, w0 : w0 + crop_width, :]
    output_image_path = image_path.replace(".jpg", "_processed.jpg").replace(".png", "_process.png")
    cv2.imwrite(output_image_path, image)
    return output_image_path

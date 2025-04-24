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

import audioread
import av
import librosa
import numpy as np

from paddlemix.processors.qwen2_5_vl_processing import process_vision_info


def _check_if_video_has_audio(video_path):
    container = av.open(video_path)
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        return False
    return True


def process_audio_info(conversations: list[dict] | list[list[dict]], use_audio_in_video: bool):
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if "audio" in ele:
                        path = ele["audio"]
                        if path.startswith("http://") or path.startswith("https://"):
                            audios.append(librosa.load(audioread.ffdec.FFmpegAudioFile(path), sr=16000)[0])
                        elif isinstance(path, np.ndarray):
                            if path.ndim > 1:
                                raise ValueError("Support only mono audio")
                            audios.append(path)
                        elif path.startswith("file://"):
                            audios.append(librosa.load(path[len("file://") :], sr=16000)[0])
                        else:
                            audios.append(librosa.load(path, sr=16000)[0])
                    else:
                        raise ValueError("Unknown audio {}".format(ele))
                if use_audio_in_video and ele["type"] == "video":
                    if "video" in ele:
                        path = ele["video"]
                        assert _check_if_video_has_audio(
                            path
                        ), "Video must has audio track when use_audio_in_video=True"
                        if path.startswith("http://") or path.startswith("https://"):
                            audios.append(librosa.load(audioread.ffdec.FFmpegAudioFile(path), sr=16000)[0])
                        elif path.startswith("file://"):
                            audios.append(librosa.load(path[len("file://") :], sr=16000)[0])
                        else:
                            audios.append(librosa.load(path, sr=16000)[0])
                    else:
                        raise ValueError("Unknown video {}".format(ele))
    if len(audios) == 0:
        audios = None
    return audios


def process_mm_info(conversations, use_audio_in_video, return_video_kwargs=False):
    audios = process_audio_info(conversations, use_audio_in_video)
    vision = process_vision_info(conversations)
    return (audios,) + vision

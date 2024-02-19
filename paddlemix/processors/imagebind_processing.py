# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The Salesforce Team Authors and The HuggingFace Team. All rights reserved.
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
"""
Processor class for ImageBind.
"""

import logging
from abc import ABC, abstractmethod
from fractions import Fraction

# from paddlevideo.data.clip_sampling import ConstantClipsPerVideoSampler
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union  # noqa

import paddle
from paddle.vision.transforms import transforms as T
from paddlenlp.transformers.tokenizer_utils_base import BatchEncoding

from .base_processing import ProcessorMixin
from .processing_utils import BaseAudioProcessor

__all__ = ["ImageBindProcessor", "ImageBindAudioProcessor"]

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10


class ImageBindProcessor(ProcessorMixin):

    # attributes = ["image_processor", "text_processor","audio_processor","imu_processor","thermalprocess","rgbdtprocess"]
    attributes = ["image_processor", "tokenizer", "audio_processor"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = "CLIPTokenizer"
    audio_processor_class = "ImageBindAudioProcessor"

    def __init__(self, image_processor, tokenizer, audio_processor):
        super().__init__(image_processor, tokenizer, audio_processor)

    def __call__(self, text=None, images=None, audios=None, return_tensors=None, **kwargs):

        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
            n, m = encoding["input_ids"].shape
            zero_encoding = paddle.zeros(shape=[n, self.tokenizer.max_len], dtype="int64")
            zero_encoding[:, :m] = paddle.to_tensor(data=encoding["input_ids"])
            encoding["input_ids"] = zero_encoding

        if images is not None:
            image_features = self.image_processor(images, return_tensors=return_tensors, **kwargs)

        if audios is not None:
            encoding["audio_values"] = self.audio_processor(audios, return_tensors=return_tensors, **kwargs)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features["image"]
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


class ImageBindAudioProcessor(BaseAudioProcessor):
    model_input_names = ["audio_values"]

    def __init__(
        self,
        num_mel_bins: int = 0,
        target_length: int = 0,
        sample_rate: int = 0,
        clip_duration: int = 0,
        clips_per_video: int = 0,
        mean: Optional[Union[float, List[float]]] = None,
        std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.clips_per_video = clips_per_video
        self.mean = mean
        self.std = std

    def preprocess(
        self,
        audio_path: Union[str, List[str]],
        **kwargs,
    ):
        """
        Preprocess the text with tokenization.
        """
        if audio_path is None:
            return None
        audio_outputs = []
        # breakpoint()
        clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=self.clip_duration, clips_per_video=self.clips_per_video
        )
        # for audio_path in audio_paths:
        waveform, sr = paddle.audio.load(audio_path)
        if self.sample_rate != sr:
            waveform = paddle.audio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)

        all_clips_timepoints = self.get_clip_timepoints(clip_sampler, waveform.shape[1] / self.sample_rate)
        all_clips = []

        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * self.sample_rate) : int(clip_timepoints[1] * self.sample_rate),
            ]
            waveform_melspec = self.waveform2melspec(
                waveform_clip, self.sample_rate, self.num_mel_bins, self.target_length
            )
            all_clips.append(waveform_melspec)

            normalize = T.Normalize(
                mean=self.mean
                if not isinstance(self.mean, (float, int))
                else [
                    self.mean,
                ],
                std=self.std
                if not isinstance(self.std, (float, int))
                else [
                    self.std,
                ],
            )

        all_clips = [normalize(ac) for ac in all_clips]
        all_clips = paddle.stack(x=all_clips, axis=0)
        audio_outputs.append(all_clips)
        return paddle.stack(x=audio_outputs, axis=0)

    def get_clip_timepoints(self, clip_sampler, duration):
        all_clips_timepoints = []
        is_last_clip = False
        end = 0.0
        while not is_last_clip:
            start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
            all_clips_timepoints.append((start, end))
        return all_clips_timepoints

    def waveform2melspec(self, waveform, sample_rate, num_mel_bins, target_length):
        waveform -= waveform.mean()
        fbank = paddle.audio.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=num_mel_bins,
            dither=0.0,
            frame_length=25,
            frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
        )
        x = fbank
        perm_0 = list(range(x.ndim))
        perm_0[0] = 1
        perm_0[1] = 0
        fbank = x.transpose(perm=perm_0)
        n_frames = fbank.shape[1]
        p = target_length - n_frames
        if abs(p) / n_frames > 0.2:
            logging.warning(
                "Large gap between audio n_frames(%d) and target_length (%d). Is the audio_target_length setting correct?",
                n_frames,
                target_length,
            )
        if p > 0:
            fbank = paddle.pad_from_torch(fbank, pad=(0, p), mode="constant", value=0)
        elif p < 0:
            fbank = fbank[:, 0:target_length]

        fbank = fbank.unsqueeze(axis=0)
        return fbank


class ClipInfo(NamedTuple):
    """
    Named-tuple for clip information with:
        clip_start_sec  (Union[float, Fraction]): clip start time.
        clip_end_sec (Union[float, Fraction]): clip end time.
        clip_index (int): clip index in the video.
        aug_index (int): augmentation index for the clip. Different augmentation methods
            might generate multiple views for the same clip.
        is_last_clip (bool): a bool specifying whether there are more clips to be
            sampled from the video.
    """

    clip_start_sec: Union[float, Fraction]
    clip_end_sec: Union[float, Fraction]
    clip_index: int
    aug_index: int
    is_last_clip: bool


class ClipSampler(ABC):
    """
    Interface for clip samplers that take a video time, previous sampled clip time,
    and returns a named-tuple ``ClipInfo``.
    """

    def __init__(self, clip_duration: Union[float, Fraction]) -> None:
        self._clip_duration = Fraction(clip_duration)
        self._current_clip_index = 0
        self._current_aug_index = 0

    @abstractmethod
    def __call__(
        self,
        last_clip_time: Union[float, Fraction],
        video_duration: Union[float, Fraction],
        annotation: Dict[str, Any],
    ) -> ClipInfo:
        pass

    def reset(self) -> None:
        """Resets any video-specific attributes in preparation for next video"""
        pass


class ConstantClipsPerVideoSampler(ClipSampler):
    """
    Evenly splits the video into clips_per_video increments and samples clips of size
    clip_duration at these increments.
    """

    def __init__(self, clip_duration: float, clips_per_video: int, augs_per_clip: int = 1) -> None:
        super().__init__(clip_duration)
        self._clips_per_video = clips_per_video
        self._augs_per_clip = augs_per_clip

    def __call__(self, last_clip_time: float, video_duration: float, annotation: Dict[str, Any]) -> ClipInfo:
        """
        Args:
            last_clip_time (float): Not used for ConstantClipsPerVideoSampler.
            video_duration: (float): the duration (in seconds) for the video that's
                being sampled.
            annotation (Dict): Not used by this sampler.
        Returns:
            a named-tuple `ClipInfo`: includes the clip information of (clip_start_time,
                clip_end_time, clip_index, aug_index, is_last_clip). The times are in seconds.
                is_last_clip is True after clips_per_video clips have been sampled or the end
                of the video is reached.

        """
        max_possible_clip_start = Fraction(max(video_duration - self._clip_duration, 0))
        uniform_clip = Fraction(max_possible_clip_start, self._clips_per_video)
        clip_start_sec = uniform_clip * self._current_clip_index
        clip_index = self._current_clip_index
        aug_index = self._current_aug_index

        self._current_aug_index += 1
        if self._current_aug_index >= self._augs_per_clip:
            self._current_clip_index += 1
            self._current_aug_index = 0

        # Last clip is True if sampled self._clips_per_video or if end of video is reached.
        is_last_clip = False
        if (
            self._current_clip_index >= self._clips_per_video
            or uniform_clip * self._current_clip_index > max_possible_clip_start
        ):
            self._current_clip_index = 0
            is_last_clip = True

        if is_last_clip:
            self.reset()

        return ClipInfo(
            clip_start_sec,
            clip_start_sec + self._clip_duration,
            clip_index,
            aug_index,
            is_last_clip,
        )

    def reset(self):
        self._current_clip_index = 0
        self._current_aug_index = 0

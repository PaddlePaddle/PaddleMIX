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

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.distributed as dist
from paddlenlp.generation import (
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    get_unfinished_flag,
    validate_stopping_criteria,
)
from paddlenlp.transformers.model_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    ModelOutput,
)
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.transformers.utils import cached_file

from ppdiffusers.utils import logging

from ...activations import ACT2FN
from .configuration_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoderConfig,
    Qwen2_5OmniBigVGANConfig,
    Qwen2_5OmniConfig,
    Qwen2_5OmniDiTConfig,
    Qwen2_5OmniTalkerConfig,
    Qwen2_5OmniTextConfig,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniToken2WavConfig,
    Qwen2_5OmniVisionEncoderConfig,
)
from .logits_processors_utils import SuppressTokensLogitsProcessor
from .modeling_rope_utils import ROPE_INIT_FUNCTIONS

logger = logging.get_logger(__name__)

__all__ = [
    "Qwen2_5OmniModel",
    "Qwen2_5OmniThinkerTextModel",
    "Qwen2_5OmniThinkerForConditionalGeneration",
    "Qwen2_5OmniTalkerModel",
    "Qwen2_5OmniTalkerForConditionalGeneration",
    "Qwen2_5OmniToken2WavDiTModel",
    "Qwen2_5OmniToken2WavBigVGANModel",
    "Qwen2_5OmniToken2WavModel",
    "Qwen2_5OmniPreTrainedModel",
    "Qwen2_5OmniPreTrainedModelForConditionalGeneration",
]


def compute_eager_attn(self, query, key, value, scale=None, attn_mask=None, is_causal=False):
    """
    query : paddle.Tensor [(B, H, L, D)]
    key : paddle.Tensor [(B, H, L, D)]
    value : paddle.Tensor [(B, H, L, D)]
    """
    b, h, l, d = query.shape
    if scale is None:
        scale = d**-0.5
    if attn_mask is not None:
        if attn_mask.dtype == paddle.bool:
            attn_mask = (~attn_mask).cast(dtype=query.dtype) * paddle.finfo(query.dtype).min
        attn_weights = paddle.matmul(query, key.transpose([0, 1, 3, 2])) * scale + attn_mask
    attn_weights = paddle.nn.functional.softmax(attn_weights, axis=-1)
    attn_out = paddle.matmul(attn_weights, value)
    attn_out = attn_out.transpose([0, 2, 1, 3])
    return attn_out


def compute_sdpa_attn(
    self,
    query,
    key,
    value,
    attn_mask=None,
    is_causal=False,
):
    q = query.transpose([0, 2, 1, 3])
    k = key.transpose([0, 2, 1, 3])
    v = value.transpose([0, 2, 1, 3])
    if attn_mask is not None:
        if attn_mask.dtype == paddle.bool:
            attn_mask = (~attn_mask).cast(dtype=query.dtype) * paddle.finfo(query.dtype).min
    return paddle.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)


# TODO: Support flash attention (liaojincheng)
def compute_flash_attn(
    self,
    query,
    key,
    value,
    attn_mask=None,
    is_causal=False,
):
    pass


ALL_ATTENTION_FUNCTIONS = {
    "eager": compute_eager_attn,
    "sdpa": compute_sdpa_attn,
    "flash_attention_2": compute_sdpa_attn,
}


def kaiser_window(
    window_length, periodic=True, beta=12.0, *, dtype=None, layout=None, device=None, stop_gradient=True
):
    # 计算窗函数的采样位置
    n = paddle.arange(0, window_length, dtype=dtype)
    alpha = (window_length - 1) / 2.0

    # 计算 Kaiser 窗
    x = (n - alpha) / alpha
    window = paddle.i0(beta * paddle.sqrt(1 - x**2)) / paddle.i0(paddle.to_tensor(beta, dtype=dtype))

    return window


class Qwen2_5OmniPreTrainedModel(PretrainedModel):
    config_class = Qwen2_5OmniConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    # def _init_weights(self, module):
    #     std = self.config.init_std if hasattr(self.config, 'init_std'
    #         ) else 0.02
    #     if isinstance(module, (paddle.nn.Linear, paddle.nn.Conv1D, paddle.
    #         nn.Conv3D)):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, paddle.nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()


class Qwen2_5OmniPreTrainedModelForConditionalGeneration(Qwen2_5OmniPreTrainedModel):
    def _prepare_4d_causal_attention_mask_with_cache_position(
        self,
        attention_mask: paddle.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: paddle.dtype,
        min_dtype: float,
        cache_position: paddle.Tensor,
        batch_size: int,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            min_dtype (`float`):
                The minimum value representable with the dtype `dtype`.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            causal_mask = paddle.full(shape=(sequence_length, target_length), fill_value=min_dtype, dtype=dtype)
            if sequence_length != 1:
                causal_mask = paddle.triu(x=causal_mask, diagonal=1)
            causal_mask *= paddle.arange(end=target_length) > cache_position.reshape([-1, 1])
            causal_mask = causal_mask[None, None, :, :].expand(shape=[batch_size, 1, -1, -1])
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                mask_length = tuple(attention_mask.shape)[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    mask=padding_mask, value=min_dtype
                )
        return causal_mask

    def get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: List[int],
        grid_hs: List[int],
        grid_ws: List[int],
    ):
        llm_pos_ids_list = []
        llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
        llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
        h_index = paddle.arange(end=llm_grid_h).view([1, -1, 1]).expand(shape=[len(t_index), -1, llm_grid_w]).flatten()
        w_index = paddle.arange(end=llm_grid_w).view([1, 1, -1]).expand(shape=[len(t_index), llm_grid_h, -1]).flatten()
        t_index = (
            paddle.to_tensor(data=t_index)
            .view([-1, 1])
            .expand(shape=[-1, llm_grid_h * llm_grid_w])
            .flatten()
            .astype(dtype="int64")
        )
        _llm_pos_ids = paddle.stack(x=[t_index, h_index, w_index])
        llm_pos_ids_list.append(_llm_pos_ids + start_idx)
        llm_pos_ids = paddle.concat(x=llm_pos_ids_list, axis=1)
        return llm_pos_ids

    def get_chunked_index(self, llm_pos_ids, t_ntoken_per_chunk, st_idx):
        def _iter():
            i, start_idx = 0, 0
            current_chunk = 1
            while i < tuple(llm_pos_ids.shape)[1]:
                if llm_pos_ids[0][i] - st_idx >= current_chunk * t_ntoken_per_chunk:
                    yield start_idx, i
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield start_idx, tuple(llm_pos_ids.shape)[1]

        return list(_iter())

    def get_rope_index(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,
        video_grid_thw: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        use_audio_in_video: bool = False,
        audio_seqlens: Optional[paddle.Tensor] = None,
        second_per_grids: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            use_audio_in_video (`bool`, *optional*):
                 If set to `True`, use the audio in video.
            audio_seqlens (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
            second_per_grids (`torch.LongTensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.config.image_token_index
        video_token_id = self.config.video_token_index
        audio_token_id = self.config.audio_token_index
        vision_start_token_id = self.config.vision_start_token_id
        audio_start_token_id = self.config.audio_start_token_id
        position_id_per_seconds = self.config.position_id_per_seconds
        seconds_per_chunk = self.config.seconds_per_chunk
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = paddle.ones_like(x=total_input_ids)
            position_ids = paddle.ones(
                shape=[3, tuple(input_ids.shape)[0], tuple(input_ids.shape)[1]], dtype=input_ids.dtype
            )
            image_idx, video_idx, audio_idx = 0, 0, 0
            attention_mask = attention_mask.to(total_input_ids.place)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums, audio_nums = 0, 0, 0
                vision_start_indices = paddle.nonzero(x=input_ids == vision_start_token_id).squeeze(axis=1)
                vision_tokens = input_ids[vision_start_indices + 1]
                audio_nums = paddle.sum(x=input_ids == audio_start_token_id)
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (
                    (vision_tokens == audio_start_token_id).sum()
                    if use_audio_in_video
                    else (vision_tokens == video_token_id).sum()
                )
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_audios = (image_nums, video_nums, audio_nums)
                multimodal_nums = (
                    image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums
                )
                for _ in range(multimodal_nums):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if audio_token_id in input_tokens and remain_audios > 0:
                        ed_audio = input_tokens.index(audio_token_id, st)
                    else:
                        ed_audio = len(input_tokens) + 1
                    min_ed = min(ed_image, ed_video, ed_audio)
                    if min_ed == ed_audio:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(
                                paddle.arange(end=text_len).view(1, -1).expand(shape=[3, -1]) + st_idx
                            )
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(paddle.arange(end=bos_len).view(1, -1).expand(shape=[3, -1]) + st_idx)
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                        llm_pos_ids = paddle.arange(end=audio_len).view(1, -1).expand(shape=[3, -1]) + st_idx
                        llm_pos_ids_list.append(llm_pos_ids)
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(paddle.arange(end=eos_len).view(1, -1).expand(shape=[3, -1]) + st_idx)
                        st += text_len + bos_len + audio_len + eos_len
                        audio_idx += 1
                        remain_audios -= 1
                    elif min_ed == ed_image:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(
                                paddle.arange(end=text_len).view([1, -1]).expand(shape=[3, -1]) + st_idx
                            )
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(
                            paddle.arange(end=bos_len).view([1, -1]).expand(shape=[3, -1]) + st_idx
                        )
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (paddle.arange(end=grid_t) * 1 * position_id_per_seconds).astype(dtype="int64")
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        image_len = image_grid_thw[image_idx].prod() // spatial_merge_size**2
                        llm_pos_ids_list.append(llm_pos_ids)
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(
                            paddle.arange(end=eos_len).view([1, -1]).expand(shape=[3, -1]) + st_idx
                        )
                        st += text_len + bos_len + image_len + eos_len
                        image_idx += 1
                        remain_images -= 1
                    elif min_ed == ed_video and not use_audio_in_video:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(
                                paddle.arange(end=text_len).view([1, -1]).expand(shape=[3, -1]) + st_idx
                            )
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(
                            paddle.arange(end=bos_len).view([1, -1]).expand(shape=[3, -1]) + st_idx
                        )
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        t_index = (
                            paddle.arange(end=grid_t)
                            * second_per_grids[video_idx].cpu().astype(dtype="float32")
                            * position_id_per_seconds
                        ).astype(dtype="int64")
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_len = video_grid_thw[video_idx].prod() // spatial_merge_size**2
                        llm_pos_ids_list.append(llm_pos_ids)
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(
                            paddle.arange(end=eos_len).view([1, -1]).expand(shape=[3, -1]) + st_idx
                        )
                        st += text_len + bos_len + video_len + eos_len
                        video_idx += 1
                        remain_videos -= 1
                    elif min_ed == ed_video and use_audio_in_video:
                        text_len = min_ed - st - 2
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(
                                paddle.arange(end=text_len).view([1, -1]).expand(shape=[3, -1]) + st_idx
                            )
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(
                            paddle.arange(end=bos_len).view([1, -1]).expand(shape=[3, -1]) + st_idx
                        )
                        llm_pos_ids_list.append(
                            paddle.arange(end=bos_len).view([1, -1]).expand(shape=[3, -1]) + st_idx
                        )
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                        audio_llm_pos_ids = paddle.arange(end=audio_len).view([1, -1]).expand(shape=[3, -1]) + st_idx
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        t_index = (
                            paddle.arange(end=grid_t)
                            * second_per_grids[video_idx].cpu().astype(dtype="float32")
                            * position_id_per_seconds
                        ).astype(dtype="int64")
                        video_llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                        video_chunk_indexes = self.get_chunked_index(video_llm_pos_ids, t_ntoken_per_chunk, st_idx)
                        audio_chunk_indexes = self.get_chunked_index(audio_llm_pos_ids, t_ntoken_per_chunk, st_idx)
                        sub_len = 0
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            video_chunk_index = video_chunk_indexes[j] if j < len(video_chunk_indexes) else None
                            audio_chunk_index = audio_chunk_indexes[j] if j < len(audio_chunk_indexes) else None
                            if video_chunk_index is not None:
                                sub_len += video_chunk_index[1] - video_chunk_index[0]
                                llm_pos_ids_list.append(
                                    video_llm_pos_ids[:, video_chunk_index[0] : video_chunk_index[1]]
                                )
                            if audio_chunk_index is not None:
                                sub_len += audio_chunk_index[1] - audio_chunk_index[0]
                                llm_pos_ids_list.append(
                                    audio_llm_pos_ids[:, audio_chunk_index[0] : audio_chunk_index[1]]
                                )
                        video_len = video_grid_thw[video_idx].prod() // spatial_merge_size**2
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(
                            paddle.arange(end=eos_len).view([1, -1]).expand(shape=[3, -1]) + st_idx
                        )
                        llm_pos_ids_list.append(
                            paddle.arange(end=eos_len).view([1, -1]).expand(shape=[3, -1]) + st_idx
                        )
                        st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2
                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(paddle.arange(end=text_len).view([1, -1]).expand(shape=[3, -1]) + st_idx)
                llm_positions = paddle.concat(x=llm_pos_ids_list, axis=1).reshape([3, -1])
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.place)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
            mrope_position_deltas = paddle.to_tensor(data=mrope_position_deltas, place=input_ids.place).unsqueeze(
                axis=1
            )
            return position_ids, mrope_position_deltas
        else:
            position_ids = attention_mask.astype(dtype="int64").cumsum(axis=-1) - 1
            position_ids.masked_fill_(mask=attention_mask == 0, value=1)
            position_ids = position_ids.unsqueeze(axis=0).expand(shape=[3, -1, -1]).to(attention_mask.place)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - paddle.sum(x=attention_mask, axis=-1, keepdim=True)
            return position_ids, mrope_position_deltas


@dataclass
class Qwen2_5OmniThinkerCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Qwen2.5OmniThinker causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        attention_mask (`torch.FloatTensor`, *optional*):
            Attentions mask, used to update attention mask and position_ids.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[paddle.Tensor] = None
    logits: Optional[paddle.Tensor] = None
    past_key_values: Optional[List[paddle.Tensor]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    attention_mask: Optional[paddle.Tensor] = None
    rope_deltas: Optional[paddle.Tensor] = None


class Qwen2_5OmniAudioAttention(paddle.nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        self.config = config
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = False
        self.is_causal = False
        self.k_proj = paddle.nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim, bias_attr=False)
        self.v_proj = paddle.nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim, bias_attr=True)
        self.q_proj = paddle.nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim, bias_attr=True)
        self.out_proj = paddle.nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim, bias_attr=True)

    def forward(
        self, hidden_states: paddle.Tensor, cu_seqlens: Optional[paddle.Tensor] = None
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        seq_length, _ = tuple(hidden_states.shape)
        query_states = self.q_proj(hidden_states).reshape([seq_length, self.num_heads, -1])
        key_states = self.k_proj(hidden_states).reshape([seq_length, self.num_heads, -1])
        value_states = self.v_proj(hidden_states).reshape([seq_length, self.num_heads, -1])
        query_states = query_states.transpose([1, 0, 2])

        key_states = key_states.transpose([1, 0, 2])
        value_states = value_states.transpose([1, 0, 2])
        attn_weights = paddle.matmul(x=query_states, y=key_states.transpose([0, 2, 1])) / math.sqrt(self.head_dim)

        attention_mask = paddle.full(
            shape=[1, seq_length, tuple(key_states.shape)[1]],
            fill_value=paddle.finfo(dtype=query_states.dtype).min,
            dtype=query_states.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        attn_weights = attn_weights + attention_mask
        attn_weights = paddle.nn.functional.softmax(x=attn_weights, axis=-1).astype(query_states.dtype)
        attn_output = (
            paddle.matmul(x=attn_weights, y=value_states).transpose([1, 0, 2]).reshape([seq_length, self.embed_dim])
        )
        attn_output = self.out_proj(attn_output)
        return attn_output


class Qwen2_5OmniAudioFlashAttention2(Qwen2_5OmniAudioAttention):
    """
    Qwen2.5OmniThinker flash attention module. This module inherits from `Qwen2_5OmniAudioAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = False

    def forward(
        self, hidden_states: paddle.Tensor, cu_seqlens: Optional[paddle.Tensor] = None
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        seq_length, all_dim = tuple(hidden_states.shape)
        query_states = self.q_proj(hidden_states)
        query_states = query_states.reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states)
        key_states = key_states.reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states)
        value_states = value_states.reshape(seq_length, self.num_heads, -1)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        assert (
            paddle.device.cuda.get_device_capability()[0] >= 8
        ), "Fault: Your device computational capabilities less 8"
        attn_output = paddle.nn.functional.flash_attention.flash_attn_unpadded(
            query=query_states,
            key=key_states,
            value=value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout=0.0,
            scale=paddle.utils.try_import("math").sqrt(query_states.shape[-1]),
        )[0]
        attn_output = attn_output.reshape(seq_length, all_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class Qwen2_5OmniAudioSdpaAttention(Qwen2_5OmniAudioAttention):
    def forward(
        self, hidden_states: paddle.Tensor, cu_seqlens: Optional[paddle.Tensor] = None
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        seq_length, _ = tuple(hidden_states.shape)
        query_states = self.q_proj(hidden_states).reshape([1, seq_length, self.num_heads, -1])
        key_states = self.k_proj(hidden_states).reshape([1, seq_length, self.num_heads, -1])
        value_states = self.v_proj(hidden_states).reshape([1, seq_length, self.num_heads, -1])
        attention_mask = paddle.zeros(shape=[1, 1, seq_length, tuple(key_states.shape)[1]], dtype=query_states.dtype)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = paddle.finfo(
                query_states.dtype
            ).min
        attn_output = paddle.nn.functional.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn_output = attn_output.squeeze(0)
        attn_output = attn_output.reshape([seq_length, self.embed_dim])
        attn_output = self.out_proj(attn_output)
        return attn_output


QWEN2_5_OMNI_AUDIO_ATTENTION_CLASSES = {
    "eager": Qwen2_5OmniAudioAttention,
    "flash_attention_2": Qwen2_5OmniAudioFlashAttention2,
    "sdpa": Qwen2_5OmniAudioSdpaAttention,
}


class Qwen2_5OmniAudioEncoderLayer(paddle.nn.Layer):
    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = QWEN2_5_OMNI_AUDIO_ATTENTION_CLASSES[config.get("_attn_implementation", "eager")](config)
        self.self_attn_layer_norm = paddle.nn.LayerNorm(normalized_shape=self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = paddle.nn.Linear(in_features=self.embed_dim, out_features=config.encoder_ffn_dim)
        self.fc2 = paddle.nn.Linear(in_features=config.encoder_ffn_dim, out_features=self.embed_dim)
        self.final_layer_norm = paddle.nn.LayerNorm(normalized_shape=self.embed_dim)

    def forward(self, hidden_states: paddle.Tensor, cu_seqlens: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, cu_seqlens=cu_seqlens)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == paddle.float16 and (
            paddle.isinf(x=hidden_states).astype("bool").any() or paddle.isnan(x=hidden_states).astype("bool").any()
        ):
            clamp_value = paddle.finfo(dtype=hidden_states.dtype).max - 1000
            hidden_states = paddle.clip(x=hidden_states, min=-clamp_value, max=clamp_value)
        outputs = (hidden_states,)
        return outputs


class SinusoidsPositionEmbedding(paddle.nn.Layer):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = paddle.exp(x=-log_timescale_increment * paddle.arange(end=channels // 2))
        scaled_time = paddle.arange(end=length, dtype="float32")[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            name="positional_embedding",
            tensor=paddle.concat(x=[paddle.sin(x=scaled_time), paddle.cos(x=scaled_time)], axis=1),
            persistable=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


class Qwen2_5OmniAudioEncoder(Qwen2_5OmniPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`Qwen2_5OmniAudioEncoderLayer`].

    Args:
        config: Qwen2_5OmniAudioEncoderConfig
    """

    config_class = Qwen2_5OmniAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["Qwen2_5OmniAudioEncoderLayer"]
    _supports_sdpa = True

    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.conv1 = paddle.nn.Conv1D(in_channels=self.num_mel_bins, out_channels=embed_dim, kernel_size=3, padding=1)
        self.conv2 = paddle.nn.Conv1D(
            in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=2, padding=1
        )
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.audio_bos_eos_token = paddle.nn.Embedding(num_embeddings=2, embedding_dim=config.output_dim)
        self.layers = paddle.nn.LayerList(
            sublayers=[Qwen2_5OmniAudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.ln_post = paddle.nn.LayerNorm(normalized_shape=config.d_model)
        self.avg_pooler = paddle.nn.AvgPool1D(kernel_size=2, stride=2, exclusive=False)
        self.proj = paddle.nn.Linear(in_features=config.d_model, out_features=config.output_dim)
        self.gradient_checkpointing = False
        self.dtype = config.dtype
        # self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.stop_gradient = not False
        self._requires_grad = False

    def get_input_embeddings(self) -> paddle.nn.Layer:
        return self.conv1

    def set_input_embeddings(self, value: paddle.nn.Layer):
        self.conv1 = value

    def forward(self, input_features, feature_lens=None, aftercnn_lens=None):
        chunk_num = paddle.ceil(x=feature_lens / (self.n_window * 2)).astype(dtype="int64")
        chunk_lengths = paddle.to_tensor(
            data=[self.n_window * 2] * int(chunk_num.sum()), dtype="int64", place=feature_lens.place
        )
        tail_chunk_index = paddle.nn.functional.pad(
            x=chunk_num, pad=(1, 0), value=-1, pad_from_left_axis=False
        ).cumsum(axis=0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = paddle.where(condition=chunk_lengths == 0, x=self.n_window * 2, y=chunk_lengths)
        chunk_list = input_features.split(chunk_lengths.tolist(), axis=1)
        padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0, padding_side="right"
        )
        padded_embed = paddle.nn.functional.gelu(
            x=self.conv1(padded_feature).astype("float32") * padded_mask.astype("float32")
        ).cast(dtype=self.dtype)
        padded_embed = (
            paddle.nn.functional.gelu(x=self.conv2(padded_embed).astype("float32"))
            .transpose([0, 2, 1])
            .cast(dtype=self.dtype)
        )
        padded_embed = padded_embed + self.positional_embedding.positional_embedding[
            : tuple(padded_embed.shape)[1], :
        ].unsqueeze(axis=0).to(padded_embed.dtype)
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_seqlens = paddle.concat(
            x=(paddle.zeros(shape=[1], dtype="int64"), padded_mask_after_cnn.sum(axis=1).cumsum(axis=0))
        ).astype("int32")
        tmp_hidden_states = []
        for idx, encoder_layer in enumerate(self.layers):
            to_drop = False
            if self.training:
                dropout_probability = paddle.rand(shape=[])
                if dropout_probability < self.layerdrop:
                    to_drop = True
            if to_drop:
                layer_outputs = None, None
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__, hidden_states, cu_seqlens
                    )
                else:
                    layer_outputs = encoder_layer(hidden_states, cu_seqlens)
                hidden_states = layer_outputs[0]
                tmp_hidden_states.append(hidden_states)
        hidden_states_list = hidden_states.split(aftercnn_lens.tolist(), axis=0)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = (
                self.avg_pooler(each_audio_states.transpose([1, 0]).unsqueeze(0)).squeeze(0).transpose([1, 0])
            )
            each_audio_states = self.ln_post(each_audio_states)
            each_audio_states = self.proj(each_audio_states)
            token_audio_list.append(each_audio_states)
        token_audio = paddle.concat(x=token_audio_list, axis=0)
        return BaseModelOutput(last_hidden_state=token_audio)

    def padded_and_mask_function(self, tensor_list, tensor_len, padding_value=0, padding_side="right"):
        max_len = tensor_len.max()
        dim = tuple(tensor_list[0].shape)[0]
        padded_tensor = paddle.full(shape=(len(tensor_list), dim, max_len), fill_value=padding_value, dtype=self.dtype)
        batch_mask = paddle.zeros(shape=(len(tensor_len), max_len), dtype="int64")
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]
        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = paddle.zeros(shape=(len(tensor_len), max_len_after_cnn), dtype="int64")
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return padded_tensor, batch_mask.unsqueeze(axis=1), batch_mask_after_cnn.astype(dtype="bool")

    def _get_feat_extract_output_lengths(self, input_lengths: paddle.Tensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : tuple(x.shape)[-1] // 2]
    x2 = x[..., tuple(x.shape)[-1] // 2 :]
    return paddle.concat(x=(-x2, x1), axis=-1)


def apply_rotary_pos_emb_vision(tensor: paddle.Tensor, freqs: paddle.Tensor) -> paddle.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.astype(dtype="float32")
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(axis=1).tile(repeat_times=[1, 1, 2]).unsqueeze(axis=0).astype(dtype="float32")
    sin = sin.unsqueeze(axis=1).tile(repeat_times=[1, 1, 2]).unsqueeze(axis=0).astype(dtype="float32")
    output = tensor * cos + rotate_half(tensor) * sin
    output = output.to(orig_dtype)
    return output


class Qwen2_5OmniVisionAttention(paddle.nn.Layer):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = paddle.nn.Linear(in_features=dim, out_features=dim, bias_attr=True)
        self.k = paddle.nn.Linear(in_features=dim, out_features=dim, bias_attr=True)
        self.v = paddle.nn.Linear(in_features=dim, out_features=dim, bias_attr=True)
        self.proj = paddle.nn.Linear(in_features=dim, out_features=dim)

    def forward(
        self, hidden_states: paddle.Tensor, cu_seqlens: paddle.Tensor, rotary_pos_emb: paddle.Tensor = None
    ) -> paddle.Tensor:
        seq_length = tuple(hidden_states.shape)[0]
        q = self.q(hidden_states).reshape([seq_length, self.num_heads, -1])
        k = self.k(hidden_states).reshape([seq_length, self.num_heads, -1])
        v = self.v(hidden_states).reshape([seq_length, self.num_heads, -1])
        q = apply_rotary_pos_emb_vision(q.unsqueeze(axis=0), rotary_pos_emb).squeeze(axis=0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(axis=0), rotary_pos_emb).squeeze(axis=0)
        attention_mask = paddle.full(
            shape=[1, seq_length, seq_length], fill_value=paddle.finfo(dtype=q.dtype).min, dtype=q.dtype
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        q = q.transpose([1, 0, 2])
        k = k.transpose([1, 0, 2])
        v = v.transpose([1, 0, 2])
        attn_weights = paddle.matmul(x=q, y=k.transpose([0, 2, 1])) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = paddle.nn.functional.softmax(x=attn_weights, axis=-1, dtype="float32").astype(q.dtype)
        attn_output = paddle.matmul(x=attn_weights, y=v)
        attn_output = attn_output.transpose([1, 0, 2])
        attn_output = attn_output.reshape([seq_length, -1])
        attn_output = self.proj(attn_output)
        return attn_output


# TODO: Fix FlashAttention
class Qwen2_5OmniVisionFlashAttention2(paddle.nn.Layer):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.q = paddle.nn.Linear(in_features=dim, out_features=dim, bias_attr=True)
        self.k = paddle.nn.Linear(in_features=dim, out_features=dim, bias_attr=True)
        self.v = paddle.nn.Linear(in_features=dim, out_features=dim, bias_attr=True)
        self.proj = paddle.nn.Linear(in_features=dim, out_features=dim)

    # def _apply_rotary_pos_emb_flashatt(self, tensor: paddle.Tensor, freqs: paddle.Tensor) -> paddle.Tensor:
    #     tensor_ = tensor.astype(dtype="float32")
    #     cos = freqs.cos()
    #     sin = freqs.sin()
    # apply rotary emb
    # output = flash_attn.layers.rotary.apply_rotary_emb(tensor_, cos, sin).astype(dtype=tensor.dtype)
    # return output

    def forward(
        self, hidden_states: paddle.Tensor, cu_seqlens: paddle.Tensor, rotary_pos_emb: paddle.Tensor = None
    ) -> paddle.Tensor:
        seq_length = tuple(hidden_states.shape)[0]
        q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
        q = self._apply_rotary_pos_emb_flashatt(q.unsqueeze(axis=0), rotary_pos_emb).squeeze(axis=0)
        k = self._apply_rotary_pos_emb_flashatt(k.unsqueeze(axis=0), rotary_pos_emb).squeeze(axis=0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        assert (
            paddle.device.cuda.get_device_capability()[0] >= 8
        ), "Fault: Your device computational capabilities less 8"
        attn_output = paddle.nn.functional.flash_attention.flash_attn_unpadded(
            query=q,
            key=k,
            value=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            scale=paddle.utils.try_import("math").sqrt(q.shape[-1]),
        )[0].reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5OmniVisionSdpaAttention(paddle.nn.Layer):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.q = paddle.nn.Linear(in_features=dim, out_features=dim, bias_attr=True)
        self.k = paddle.nn.Linear(in_features=dim, out_features=dim, bias_attr=True)
        self.v = paddle.nn.Linear(in_features=dim, out_features=dim, bias_attr=True)
        self.proj = paddle.nn.Linear(in_features=dim, out_features=dim)

    def forward(
        self, hidden_states: paddle.Tensor, cu_seqlens: paddle.Tensor, rotary_pos_emb: paddle.Tensor = None
    ) -> paddle.Tensor:
        seq_length = tuple(hidden_states.shape)[0]
        q = self.q(hidden_states).reshape([1, seq_length, self.num_heads, -1])
        k = self.k(hidden_states).reshape([1, seq_length, self.num_heads, -1])
        v = self.v(hidden_states).reshape([1, seq_length, self.num_heads, -1])
        q = apply_rotary_pos_emb_vision(q.unsqueeze(axis=0), rotary_pos_emb).squeeze(axis=0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(axis=0), rotary_pos_emb).squeeze(axis=0)
        attention_mask = paddle.full(
            shape=[1, 1, seq_length, seq_length], dtype=q.dtype, fill_value=paddle.finfo(q.dtype).min
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        attn_output = paddle.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=attention_mask, dropout_p=0.0
        )
        attn_output = attn_output.reshape([seq_length, -1])
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5OmniMLP(paddle.nn.Layer):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.intermediate_size, bias_attr=bias
        )
        self.up_proj = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.intermediate_size, bias_attr=bias
        )
        self.down_proj = paddle.nn.Linear(
            in_features=self.intermediate_size, out_features=self.hidden_size, bias_attr=bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Qwen2RMSNorm(paddle.nn.Layer):
    def __init__(self, hidden_size, eps=1e-06):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.ones(shape=hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype("float32")
        variance = hidden_states.pow(y=2).mean(axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(x=variance + self.variance_epsilon)
        return (self.weight * hidden_states).astype(input_dtype)

    def extra_repr(self):
        return f"{tuple(tuple(self.weight.shape))}, eps={self.variance_epsilon}"


QWEN2_5_OMNI_VISION_ATTENTION_CLASSES = {
    "eager": Qwen2_5OmniVisionAttention,
    "flash_attention_2": Qwen2_5OmniVisionFlashAttention2,
    "sdpa": Qwen2_5OmniVisionSdpaAttention,
}


class Qwen2_5OmniVisionBlock(paddle.nn.Layer):
    def __init__(self, config: Qwen2_5OmniVisionEncoderConfig) -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-06)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-06)
        self.attn = QWEN2_5_OMNI_VISION_ATTENTION_CLASSES[config.get("_attn_implementation", "eager")](
            config.hidden_size, num_heads=config.num_heads
        )
        self.mlp = Qwen2_5OmniMLP(config, bias=True)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> paddle.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5_VisionPatchEmbed(paddle.nn.Layer):
    def __init__(
        self, patch_size: int = 14, temporal_patch_size: int = 2, in_channels: int = 3, embed_dim: int = 1152
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = paddle.nn.Conv3D(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias_attr=False,
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            [-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size]
        )
        hidden_states = self.proj(hidden_states.astype(target_dtype)).view([-1, self.embed_dim])
        return hidden_states


class Qwen2_5_VisionRotaryEmbedding(paddle.nn.Layer):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / theta ** (paddle.arange(start=0, end=dim, step=2, dtype="float32") / dim)
        self.register_buffer(name="inv_freq", tensor=inv_freq, persistable=False)

    def forward(self, seqlen: int) -> paddle.Tensor:
        seq = paddle.arange(dtype=self.inv_freq.dtype, end=seqlen)
        freqs = paddle.outer(x=seq, y=self.inv_freq)
        return freqs


class Qwen2_5OmniPatchMerger(paddle.nn.Layer):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * spatial_merge_size**2
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-06)
        self.mlp = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
            paddle.nn.GELU(),
            paddle.nn.Linear(in_features=self.hidden_size, out_features=dim),
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.mlp(self.ln_q(x).view([-1, self.hidden_size]))
        return x


class Qwen2_5OmniVisionEncoder(Qwen2_5OmniPreTrainedModel):
    config_class = Qwen2_5OmniVisionEncoderConfig
    _no_split_modules = ["Qwen2_5OmniVisionBlock"]
    # TODO vision
    def __init__(self, config: Qwen2_5OmniVisionEncoderConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.blocks = paddle.nn.LayerList(sublayers=[Qwen2_5OmniVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen2_5OmniPatchMerger(
            dim=config.out_hidden_size, context_dim=config.hidden_size, spatial_merge_size=config.spatial_merge_size
        )
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = paddle.arange(end=h).unsqueeze(axis=1).expand(shape=[-1, w])
            hpos_ids = hpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            hpos_ids = hpos_ids.transpose(perm=[0, 2, 1, 3])
            hpos_ids = hpos_ids.flatten()
            wpos_ids = paddle.arange(end=w).unsqueeze(axis=0).expand(shape=[h, -1])
            wpos_ids = wpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            wpos_ids = wpos_ids.transpose(perm=[0, 2, 1, 3])
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(paddle.stack(x=[hpos_ids, wpos_ids], axis=-1).tile(repeat_times=[t, 1]))
        pos_ids = paddle.concat(x=pos_ids, axis=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(start_axis=1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (grid_h // self.spatial_merge_size, grid_w // self.spatial_merge_size)
            index = paddle.arange(end=grid_t * llm_grid_h * llm_grid_w).reshape([grid_t, llm_grid_h, llm_grid_w])
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = paddle.nn.functional.pad(
                x=index, pad=(0, pad_w, 0, pad_h), mode="constant", value=-100, pad_from_left_axis=False
            )
            index_padded = index_padded.reshape(
                [grid_t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size]
            )
            index_padded = index_padded.transpose(perm=[0, 1, 3, 2, 4]).reshape(
                [grid_t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size]
            )
            seqlens = (index_padded != -100).sum(axis=[2, 3]).reshape([-1])
            index_padded = index_padded.reshape([-1])
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(axis=0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = paddle.concat(x=window_index, axis=0)
        return window_index, cu_window_seqlens

    def forward(self, hidden_states: paddle.Tensor, grid_thw: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = paddle.to_tensor(data=cu_window_seqlens, dtype="int32", place=hidden_states.place)
        cu_window_seqlens = paddle.unique_consecutive(x=cu_window_seqlens)
        seq_len, _ = tuple(hidden_states.shape)
        hidden_states = hidden_states.reshape([seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1])
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape([seq_len, -1])
        rotary_pos_emb = rotary_pos_emb.reshape([seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1])
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape([seq_len, -1])
        cu_seqlens = paddle.repeat_interleave(x=grid_thw[:, 1] * grid_thw[:, 2], repeats=grid_thw[:, 0]).cumsum(
            axis=0, dtype="int32"
        )
        cu_seqlens = paddle.nn.functional.pad(x=cu_seqlens, pad=(1, 0), value=0, pad_from_left_axis=False)
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, rotary_pos_emb
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, rotary_pos_emb=rotary_pos_emb)
        hidden_states = self.merger(hidden_states)
        reverse_indices = paddle.argsort(x=window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

    def get_dtype(self) -> paddle.dtype:
        return self.blocks[0].mlp.gate_proj.weight.dtype

    def get_device(self) -> str:
        return self.blocks[0].mlp.gate_proj.weight.place


class Qwen2_5OmniRotaryEmbedding(paddle.nn.Layer):
    def __init__(self, config: Qwen2_5OmniThinkerConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        # TODO
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer(name="inv_freq", tensor=inv_freq, persistable=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = paddle.max(x=position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer(name="inv_freq", tensor=inv_freq, persistable=False)
            self.max_seq_len_cached = seq_len
        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
            self.register_buffer(name="inv_freq", tensor=self.original_inv_freq, persistable=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @paddle.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.place)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .astype(dtype="float32")
            .expand(shape=[3, tuple(position_ids.shape)[1], -1, 1])
        )
        position_ids_expanded = position_ids[:, :, None, :].astype(dtype="float32")

        with paddle.amp.auto_cast(enable=False):
            freqs = inv_freq_expanded.astype(dtype="float32") @ position_ids_expanded.astype(dtype="float32")
            freqs = freqs.transpose(perm=[0, 1, 3, 2])
            emb = paddle.concat(x=(freqs, freqs), axis=-1)
            cos = emb.cos()
            sin = emb.sin()
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos.astype(dtype=x.dtype), sin.astype(dtype=x.dtype)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = paddle.concat(x=[m[i % 3] for i, m in enumerate(cos.split(mrope_section, axis=-1))], axis=-1).unsqueeze(
        axis=unsqueeze_dim
    )
    sin = paddle.concat(x=[m[i % 3] for i, m in enumerate(sin.split(mrope_section, axis=-1))], axis=-1).unsqueeze(
        axis=unsqueeze_dim
    )

    q_embed = q * cos + rotate_half(q) * sin  # q : [1,28,2378,128] + k: [1,4,2376,128]
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = tuple(hidden_states.shape)
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(shape=[batch, num_key_value_heads, n_rep, slen, head_dim])
    return hidden_states.reshape([batch, num_key_value_heads * n_rep, slen, head_dim])


class Qwen2_5OmniAttention(paddle.nn.Layer):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2_5OmniConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` when creating this class."
            )
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        self.q_proj = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.num_heads * self.head_dim, bias_attr=True
        )
        self.k_proj = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.num_key_value_heads * self.head_dim, bias_attr=True
        )
        self.v_proj = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.num_key_value_heads * self.head_dim, bias_attr=True
        )
        self.o_proj = paddle.nn.Linear(
            in_features=self.num_heads * self.head_dim, out_features=self.hidden_size, bias_attr=False
        )
        self.rotary_emb = Qwen2_5OmniRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        bsz, q_len, _ = tuple(hidden_states.shape)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view([bsz, q_len, -1, self.head_dim]).transpose([0, 2, 1, 3])
        key_states = key_states.view([bsz, q_len, -1, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = value_states.view([bsz, q_len, -1, self.head_dim]).transpose([0, 2, 1, 3])
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        # update kv cache
        if use_cache and past_key_value[0][self.layer_idx] is None:
            past_key_value[0][self.layer_idx] = key_states
            past_key_value[1][self.layer_idx] = value_states
        elif use_cache and past_key_value is not None:
            # cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position':
            #     cache_position}
            # get key states and value states from cache
            past_key_states, past_value_states = past_key_value[0][self.layer_idx], past_key_value[1][self.layer_idx]
            key_states = paddle.concat([past_key_states, key_states], axis=2)  # b h l d
            value_states = paddle.concat([past_value_states, value_states], axis=2)

            past_key_value[0][self.layer_idx] = key_states
            past_key_value[1][self.layer_idx] = value_states

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        for i in range(q_len):
            attention_mask[..., i, : i + 1] = 0
        attn_weights = paddle.matmul(x=query_states, y=key_states.transpose(perm=[0, 1, 3, 2])) / math.sqrt(
            self.head_dim
        )
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : tuple(key_states.shape)[-2]]
            attn_weights = attn_weights + causal_mask
        if query_states.dtype == paddle.float16:
            attn_weights = paddle.where(
                condition=paddle.isinf(x=attn_weights), x=paddle.zeros_like(x=attn_weights), y=attn_weights
            )
        attn_weights = paddle.nn.functional.softmax(x=attn_weights, axis=-1, dtype="float32").astype(
            query_states.dtype
        )
        attn_weights = paddle.nn.functional.dropout(x=attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = paddle.matmul(x=attn_weights, y=value_states)
        if tuple(attn_output.shape) != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {bsz, self.num_heads, q_len, self.head_dim}, but is {tuple(attn_output.shape)}"
            )
        attn_output = attn_output.transpose(perm=[0, 2, 1, 3]).contiguous()
        attn_output = attn_output.reshape([bsz, q_len, -1])
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class Qwen2MLP(paddle.nn.Layer):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.intermediate_size, bias_attr=bias
        )
        self.up_proj = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.intermediate_size, bias_attr=bias
        )
        self.down_proj = paddle.nn.Linear(
            in_features=self.intermediate_size, out_features=self.hidden_size, bias_attr=bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Qwen2_5OmniFlashAttention2(Qwen2_5OmniAttention):
    """
    Qwen2_5Omni flash attention module, following Qwen2_5Omni attention module. This module inherits from `Qwen2_5OmniAttention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = False

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
    ):
        bsz, q_len, _ = tuple(hidden_states.shape)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, -1, self.head_dim)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim)
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        # update kv cache
        if use_cache and past_key_value[0][self.layer_idx] is None:
            past_key_value[0][self.layer_idx] = key_states
            past_key_value[1][self.layer_idx] = value_states
        elif use_cache and past_key_value is not None:
            # cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position':
            #     cache_position}
            # get key states and value states from cache
            past_key_states, past_value_states = past_key_value[0][self.layer_idx], past_key_value[1][self.layer_idx]
            key_states = paddle.concat([past_key_states, key_states], axis=2)  # b h l d
            value_states = paddle.concat([past_value_states, value_states], axis=2)

            past_key_value[0][self.layer_idx] = key_states
            past_key_value[1][self.layer_idx] = value_states

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout
        input_dtype = query_states.dtype
        if input_dtype == "float32":
            # TODO
            if paddle.is_autocast_enabled():
                target_dtype = paddle.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in {target_dtype}."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        # query_states = query_states.transpose(perm=dim2perm(query_states.ndim, 1, 2))
        # key_states = key_states.transpose(perm=dim2perm(key_states.ndim, 1, 2))
        # value_states = value_states.transpose(perm=dim2perm(value_states.ndim, 1, 2))
        # if (
        #     self.config.use_sliding_window
        #     and getattr(self.config, "sliding_window", None) is not None
        #     and self.layer_idx >= self.config.max_window_layers
        # ):
        #     sliding_window = self.config.sliding_window
        # else:
        #     sliding_window = None
        # TODO
        attn_output = None
        # attn_output = _flash_attention_forward(
        #     query_states,
        #     key_states,
        #     value_states,
        #     attention_mask,
        #     q_len,
        #     dropout=dropout_rate,
        #     sliding_window=sliding_window,
        #     is_causal=self.is_causal,
        #     use_top_left_mask=self._flash_attn_uses_top_left_mask,
        # )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class Qwen2_5OmniSdpaAttention(Qwen2_5OmniAttention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if output_attentions:
            logger.warning_once(
                'Qwen2_5OmniModel is using Qwen2_5OmniSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = tuple(hidden_states.shape)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view([bsz, q_len, -1, self.head_dim]).transpose([0, 2, 1, 3])
        key_states = key_states.view([bsz, q_len, -1, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = value_states.view([bsz, q_len, -1, self.head_dim]).transpose([0, 2, 1, 3])

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        # update kv cache
        if use_cache and past_key_value[0][self.layer_idx] is None:
            past_key_value[0][self.layer_idx] = key_states
            past_key_value[1][self.layer_idx] = value_states
        elif use_cache and past_key_value is not None:

            # get key states and value states from cache
            past_key_states, past_value_states = past_key_value[0][self.layer_idx], past_key_value[1][self.layer_idx]
            key_states = paddle.concat([past_key_states, key_states], axis=2)  # b h l d
            value_states = paddle.concat([past_value_states, value_states], axis=2)

            past_key_value[0][self.layer_idx] = key_states
            past_key_value[1][self.layer_idx] = value_states

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        causal_mask = attention_mask

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : tuple(key_states.shape)[-2]]
        is_causal = True if causal_mask is None and q_len > 1 else False

        # tranpose qkv
        query_states = query_states.transpose([0, 2, 1, 3])
        key_states = key_states.transpose([0, 2, 1, 3])
        value_states = value_states.transpose([0, 2, 1, 3])
        attn_output = paddle.nn.functional.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.view([bsz, q_len, -1])
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value


QWEN2_5_OMNI_ATTENTION_CLASSES = {
    "eager": Qwen2_5OmniAttention,
    "flash_attention_2": Qwen2_5OmniFlashAttention2,
    "sdpa": Qwen2_5OmniSdpaAttention,
}


class Qwen2_5OmniDecoderLayer(paddle.nn.Layer):
    def __init__(self, config: Qwen2_5OmniConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self._attn_implementation = config.get("_attn_implementation", "eager")
        if config.use_sliding_window and self._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{self._attn_implementation}`; unexpected results may be encountered."
            )
        self.self_attn = QWEN2_5_OMNI_ATTENTION_CLASSES[self._attn_implementation](config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[paddle.Tensor] = None,
        position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        **kwargs
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class Qwen2_5OmniThinkerTextModel(Qwen2_5OmniPreTrainedModel):
    config_class = Qwen2_5OmniTextConfig
    _no_split_modules = ["Qwen2_5OmniDecoderLayer"]

    def __init__(self, config: Qwen2_5OmniTextConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = paddle.nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, padding_idx=self.padding_idx
        )
        self.layers = paddle.nn.LayerList(
            sublayers=[Qwen2_5OmniDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config.get("_attn_implementation", "eager")
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5OmniRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
                # TODO
        if use_cache and past_key_values is None:
            past_key_values = [[None] * len(self.layers), [None] * len(self.layers)]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen_tokens = past_key_values[0][0].shape[2] if past_key_values[0][0] is not None else 0
            cache_position = paddle.arange(
                start=past_seen_tokens, end=past_seen_tokens + tuple(inputs_embeds.shape)[1]
            )
        if position_ids is None:
            position_ids = cache_position.view([1, 1, -1]).expand(shape=[3, tuple(inputs_embeds.shape)[0], -1])
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(shape=[3, tuple(position_ids.shape)[0], -1])
        # TODO
        # import pdb;pdb.set_trace()
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = [None] * len(self.layers)
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: paddle.Tensor,
        input_tensor: paddle.Tensor,
        cache_position: paddle.Tensor,
        past_key_values: Tuple[paddle.Tensor],
        output_attentions: bool = False,
    ):
        if self._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != tuple(input_tensor.shape)[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right' this may lead to unexpected behaviour for Flash Attention version of Qwen25OmniThinkerText. Make sure to  call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = past_key_values[0][0].shape[2] if past_key_values[0][0] is not None else 0

        # TODO Fix Cache
        using_static_cache = False
        using_sliding_window_cache = False
        # using_static_cache = isinstance(past_key_values, StaticCache)
        # using_sliding_window_cache = isinstance(past_key_values,
        #     SlidingWindowCache)
        if (
            self._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if self._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None
        dtype, _ = input_tensor.dtype, input_tensor.place
        min_dtype = paddle.finfo(dtype=dtype).min
        sequence_length = tuple(input_tensor.shape)[1]
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                tuple(attention_mask.shape)[-1]
                if isinstance(attention_mask, paddle.Tensor)
                else past_seen_tokens + sequence_length + 1
            )
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=tuple(input_tensor.shape)[0],
            config=self.config,
            past_key_values=past_key_values,
        )
        if self._attn_implementation == "sdpa" and attention_mask is not None and not output_attentions:
            causal_mask = self._unmask_unattended(causal_mask, min_dtype)
        return causal_mask

    def _ignore_causal_mask_sdpa(
        self,
        attention_mask: Optional[paddle.Tensor],
        inputs_embeds: paddle.Tensor,
        past_key_values_length: int,
        sliding_window: Optional[int] = None,
        is_training: bool = False,
    ) -> bool:
        """
        Detects whether the optional user-specified attention_mask & the automatically created causal mask can be
        ignored in case PyTorch's SDPA is used, rather relying on SDPA's `is_causal` argument.

        In case no token is masked in the `attention_mask` argument, if `query_length == 1` or
        `key_value_length == query_length`, we rather rely on SDPA `is_causal` argument to use causal/non-causal masks,
        allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is
        passed).
        """

        _, query_length = inputs_embeds.shape[0], inputs_embeds.shape[1]
        key_value_length = query_length + past_key_values_length
        ignore_causal_mask = False

        if attention_mask is None:
            if (
                is_training
                and (query_length == 1 or key_value_length == query_length)
                and (sliding_window is None or key_value_length < sliding_window)
            ):
                ignore_causal_mask = True
        elif sliding_window is None or key_value_length < sliding_window:
            if len(attention_mask.shape) == 4:
                return False
            elif paddle.all(attention_mask == 1):
                if query_length == 1 or key_value_length == query_length:
                    # For query_length == 1, causal attention and bi-directional attention are the same.
                    ignore_causal_mask = True

                # Unfortunately, for query_length > 1 and key_value_length != query_length, we cannot generally ignore
                # the attention mask, as SDPA causal mask generation may be wrong. We will set `is_causal=False` in
                # SDPA and rely on Transformers attention_mask instead, hence not setting it to None here.
                # Reference: https://github.com/pytorch/pytorch/issues/108108
                # TODO: maybe revisit this with https://github.com/pytorch/pytorch/pull/114823 in PyTorch 2.3.

        return ignore_causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: paddle.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: paddle.dtype,
        cache_position: paddle.Tensor,
        batch_size: int,
        config: Qwen2_5OmniConfig,
        past_key_values: Tuple[paddle.Tensor],
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen25OmniThinkerTextConfig`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = paddle.finfo(dtype=dtype).min
            causal_mask = paddle.full(shape=(sequence_length, target_length), fill_value=min_dtype, dtype=dtype)
            diagonal_attend_mask = paddle.arange(end=target_length) > cache_position.reshape([-1, 1])
            if config.sliding_window is not None:
                # TODO (liaojincheng)
                # if not isinstance(past_key_values, SlidingWindowCache
                # ) or sequence_length > target_length:
                if sequence_length > target_length:
                    sliding_attend_mask = (
                        paddle.arange(end=target_length) <= cache_position.reshape([-1, 1]) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(y=sliding_attend_mask)
            causal_mask *= diagonal_attend_mask.astype(causal_mask.dtype)
            causal_mask = causal_mask[None, None, :, :].expand(shape=[batch_size, 1, -1, -1])
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if tuple(attention_mask.shape)[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = tuple(attention_mask.shape)[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.place
                ).astype(causal_mask.dtype)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    mask=padding_mask, value=min_dtype
                )
        return causal_mask

    def _unmask_unattended(
        self,
        expanded_mask: paddle.Tensor,
        min_dtype: float,
    ):
        if expanded_mask.dtype == paddle.bool:
            raise ValueError(
                "AttentionMaskConverter._unmask_unattended expects a float `expanded_mask`, got a BoolTensor."
            )

        return expanded_mask.multiply(
            (~paddle.all(expanded_mask == min_dtype, axis=-1, keepdim=True)).astype(expanded_mask.dtype)
        )


class Qwen2_5OmniThinkerForConditionalGeneration(Qwen2_5OmniPreTrainedModelForConditionalGeneration):
    config_class = Qwen2_5OmniThinkerConfig
    _no_split_modules = ["Qwen2_5OmniAudioEncoder", "Qwen2_5OmniVisionEncoder"]

    def __init__(self, config: Qwen2_5OmniThinkerConfig):
        super().__init__(config)
        # set dtype
        config.audio_config.dtype = config.dtype
        config.vision_config.dtype = config.dtype
        config.text_config.dtype = config.dtype

        self.audio_tower = Qwen2_5OmniAudioEncoder._from_config(
            config.audio_config, attn_implementation=config.get("_attn_implementation", "eager")
        )
        self.visual = Qwen2_5OmniVisionEncoder._from_config(
            config.vision_config, attn_implementation=config.get("_attn_implementation", "eager")
        )
        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen2_5OmniThinkerTextModel._from_config(
            config.text_config, attn_implementation=config.get("_attn_implementation", "eager")
        )
        self.lm_head = paddle.nn.Linear(
            in_features=config.text_config.hidden_size, out_features=config.text_config.vocab_size, bias_attr=False
        )
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.rope_deltas = None
        # self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        input_features: Optional[paddle.Tensor] = None,
        pixel_values: Optional[paddle.Tensor] = None,
        pixel_values_videos: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,
        video_grid_thw: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        feature_attention_mask: Optional[paddle.Tensor] = None,
        audio_feature_lengths: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        rope_deltas: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        cache_position: Optional[paddle.Tensor] = None,
        video_second_per_grid: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from qwen_vl_utils import process_vision_info
        >>> from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        >>> thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        >>> processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

        >>> conversations = [
        >>>         {'role': 'system', 'content': 'You are a helpful voice chat bot, and please respond to me in a casual conversation manner using random voice.'},
        >>>         {"role": "user", "content": [
        >>>             {"type": "image", "image_url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
        >>>             {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
        >>>         ]},
        >>> ]

        >>> text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        >>> audios = [ librosa.load(BytesIO(urlopen( conversations[1]['content'][1]['audio_url'] ).read()), sr=self.processor.feature_extractor.sampling_rate) ]
        >>> images, videos = process_vision_info(conversations)
        >>> inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)

        >>> # Generate
        >>> inputs['use_audio_in_video'] = `True` or `False`
        >>> generation = thinker.generate(**inputs, max_new_tokens=2048)
        >>> generate_ids = generation[:, inputs.input_ids.size(1):]

        >>> response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if feature_attention_mask is not None:
            audio_feature_lengths = paddle.sum(x=feature_attention_mask, axis=1)
            input_features = input_features.transpose(perm=[0, 2, 1])[
                feature_attention_mask.astype(dtype="bool")
            ].transpose(perm=[1, 0])
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or cache_position is not None
                and cache_position[0] == 0
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(axis=-1).unsqueeze(axis=1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = tuple(input_ids.shape)
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = paddle.arange(end=seq_length)
                position_ids = position_ids.view([1, -1]).expand(shape=[batch_size, -1])
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(axis=0).expand(shape=[3, -1, -1])
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        embeds_to_talker = inputs_embeds.clone()
        if tuple(input_ids.shape)[1] != 1:
            if input_features is not None:
                audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                    audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(axis=-1)
                )
                feature_lens = (
                    audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(axis=-1)
                )
                audio_outputs = self.audio_tower(
                    input_features, feature_lens=feature_lens, aftercnn_lens=audio_feat_lengths
                )
                audio_features = audio_outputs.last_hidden_state
                if tuple(audio_features.shape)[0] != sum(audio_output_lengths.tolist()):
                    raise ValueError("length of audio_features should match audio_output_lengths")
                audio_mask = (
                    (input_ids == self.config.audio_token_index)
                    .unsqueeze(axis=-1)
                    .expand_as(y=inputs_embeds)
                    .to(inputs_embeds.place)
                )
                audio_features = audio_features.to(inputs_embeds.place).astype(inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
                embeds_to_talker = embeds_to_talker.masked_scatter(audio_mask, paddle.zeros_like(x=audio_features))
            if pixel_values is not None:
                pixel_values = pixel_values.astype(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (
                    (input_ids == self.config.image_token_index)
                    .unsqueeze(axis=-1)
                    .expand_as(y=inputs_embeds)
                    .to(inputs_embeds.place)
                )
                image_embeds = image_embeds.astype(inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                embeds_to_talker = embeds_to_talker.masked_scatter(image_mask, paddle.zeros_like(x=image_embeds))
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.astype(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (
                    (input_ids == self.config.video_token_index)
                    .unsqueeze(axis=-1)
                    .expand_as(y=inputs_embeds)
                    .to(inputs_embeds.place)
                )
                video_embeds = video_embeds.to(inputs_embeds.place).astype(inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                embeds_to_talker = embeds_to_talker.masked_scatter(video_mask, paddle.zeros_like(x=video_embeds))
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.place)
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
        if not return_dict:
            output = (logits,) + (embeds_to_talker, outputs[0]) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=(embeds_to_talker, outputs.hidden_states),
            attentions=outputs.attentions,
            attention_mask=attention_mask,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        return_dict=False,
        **kwargs
    ):

        if use_cache:
            if past_key_values is None or past_key_values[0][0] is None:
                cache_position = paddle.arange(0, len(input_ids))

            # trunc input when use kv cache and set cache_postion
            if past_key_values is not None and past_key_values[0][0] is not None:
                cache_position = paddle.arange(past_key_values[0][0].shape[2], past_key_values[0][0].shape[2] + 1)
                trim_input_ids = input_ids[:, -1:]
            else:
                trim_input_ids = input_ids
        else:
            trim_input_ids = input_ids
        model_inputs = dict(
            input_ids=trim_input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            return_dict=return_dict,
        )
        # kwargs.pop("return_dict_in_generate")
        if kwargs.get("output_hidden_states") is not None:
            model_inputs["output_hidden_states"] = kwargs["output_hidden_states"]
        model_inputs["position_ids"] = None
        if cache_position is not None and cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
        return model_inputs

    def update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:

        # set present input_ids for next token generation
        # model_kwargs["input_ids"] = next_tokens

        if getattr(outputs, "attention_mask", None) is not None:
            model_kwargs["attention_mask"] = outputs.attention_mask

        model_kwargs = super().update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder)

        past_key_values = None
        # kv cache
        if model_kwargs["use_cache"] and isinstance(outputs, ModelOutput):
            if getattr(outputs, "past_key_values", None) is not None:
                past_key_values = outputs.past_key_values

        elif isinstance(outputs, Tuple):
            if model_kwargs.get("output_hidden_states", False):
                past_key_values = outputs[-2]
            else:
                past_key_values = outputs[-1]

        model_kwargs["past_key_values"] = past_key_values
        return model_kwargs

    def greedy_search(
        self,
        input_ids,
        logits_processors,
        max_length,
        pad_token_id,
        eos_token_id,
        stopping_criteria=None,
        streamer=None,
        fast_ptq_sampling=False,
        trunc_input=True,
        synced_gpus=False,
        **model_kwargs
    ):
        model_kwargs["use_cache"] = model_kwargs.get("use_cache", True)
        return_dict_in_generate = model_kwargs.get("return_dict_in_generate", False)
        output_hidden_states = model_kwargs.get("output_hidden_states", False)
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        batch_size, cur_len = input_ids.shape
        origin_len = cur_len
        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())
        generate_end = False
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = paddle.to_tensor(0.0 if generate_end else 1.0)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs & get model output
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs)

            if synced_gpus and generate_end:
                continue  # don't waste resources running the code we don't need

            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, ModelOutput):
                logits = outputs.logits
            else:
                logits = outputs
            # [batch_size, vocab_size]
            next_token_logits = logits[:, -1, :]

            # pre-process distribution
            next_token_logits = self.adjust_logits_during_generation(next_token_logits)
            probs = logits_processors(input_ids, next_token_logits)

            # Store hidden_states when required
            if return_dict_in_generate:
                if output_hidden_states:
                    if isinstance(outputs, tuple):
                        decoder_hidden_states += (outputs[1], outputs[2])
                    elif isinstance(outputs, ModelOutput):
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )
                    else:
                        raise ValueError(f"outputs type {type(outputs)} is not supported")
            # greedy
            next_tokens = paddle.argmax(probs, axis=-1).unsqueeze(-1)
            next_scores = paddle.index_sample(probs, next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(unfinished_flag, next_tokens, paddle.full_like(next_tokens, pad_token_id))

            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)
            cur_len += 1

            input_ids = paddle.concat([input_ids, next_tokens], axis=1)
            if streamer is not None:
                if self.config.tensor_parallel_rank == 0:
                    streamer.put(next_tokens.cpu())

            if stopping_criteria(input_ids, scores):
                generate_end = True

            if eos_token_id is not None:
                unfinished_flag = get_unfinished_flag(input_ids, unfinished_flag, eos_token_id)
                if not paddle.any(unfinished_flag):
                    generate_end = True

            # Stop when there is a </s> in all sentences
            if generate_end and not synced_gpus:
                break

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if fast_ptq_sampling:
                break

        if streamer is not None:
            streamer.end()
        if return_dict_in_generate:
            return ModelOutput(
                logits=logits,
                sequences=input_ids[:, origin_len:] if trunc_input else input_ids,
                past_key_values=model_kwargs["past_key_values"],
                hidden_states=decoder_hidden_states,
            )
        else:
            return input_ids[:, origin_len:] if trunc_input else input_ids, scores


@dataclass
class Qwen2_5OmniTalkerCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Qwen2.5OmniTalker causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        attention_mask (`torch.FloatTensor`, *optional*):
            Attentions mask, used to update attention mask and position_ids.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[paddle.Tensor] = None
    logits: paddle.float32 = None
    past_key_values: Optional[List[paddle.Tensor]] = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None
    attention_mask: Optional[paddle.Tensor] = None
    rope_deltas: Optional[paddle.Tensor] = None
    thinker_reply_part: paddle.float32 = None


class Qwen2_5OmniTalkerModel(Qwen2_5OmniPreTrainedModel):
    config_class = Qwen2_5OmniTalkerConfig
    _no_split_modules = ["Qwen2_5OmniTalkerDecoderLayer"]

    def __init__(self, config: Qwen2_5OmniTalkerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = paddle.nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.embedding_size, padding_idx=self.padding_idx
        )
        self.layers = paddle.nn.LayerList(
            sublayers=[Qwen2_5OmniDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config.get("_attn_implementation", "eager")
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5OmniRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        if use_cache and past_key_values is None:
            past_key_values = [[None] * len(self.layers), [None] * len(self.layers)]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen_tokens = past_key_values[0][0].shape[2] if past_key_values[0][0] is not None else 0
            cache_position = paddle.arange(
                start=past_seen_tokens, end=past_seen_tokens + tuple(inputs_embeds.shape)[1]
            )
        if position_ids is None:
            position_ids = cache_position.view([1, 1, -1]).expand(shape=[3, tuple(inputs_embeds.shape)[0], -1])
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(shape=[3, tuple(position_ids.shape)[0], -1])
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # TODO Fix recompute (liaojincheng)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _unmask_unattended(
        self,
        expanded_mask: paddle.Tensor,
        min_dtype: float,
    ):
        # fmt: off
        # fmt: on
        if expanded_mask.dtype == paddle.bool:
            raise ValueError(
                "AttentionMaskConverter._unmask_unattended expects a float `expanded_mask`, got a BoolTensor."
            )

        return expanded_mask.multiply(
            (~paddle.all(expanded_mask == min_dtype, axis=-1, keepdim=True)).astype(expanded_mask.dtype)
        )

    def _update_causal_mask(
        self,
        attention_mask: paddle.Tensor,
        input_tensor: paddle.Tensor,
        cache_position: paddle.Tensor,
        past_key_values: Tuple[paddle.Tensor],
        output_attentions: bool = False,
    ):
        if self._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != tuple(input_tensor.shape)[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right' this may lead to unexpected behaviour for Flash Attention version of Qwen25OmniTalker. Make sure to  call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        # past_seen_tokens = past_key_values.get_seq_length(
        #     ) if past_key_values is not None else 0
        past_seen_tokens = past_key_values[0][0].shape[2] if past_key_values[0][0] is not None else 0
        using_static_cache = False
        using_sliding_window_cache = False
        if (
            self._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if self._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None
        dtype, _ = input_tensor.dtype, input_tensor.place
        min_dtype = paddle.finfo(dtype=dtype).min
        sequence_length = tuple(input_tensor.shape)[1]

        # TODO Fix (liaojincheng)
        if using_sliding_window_cache or using_static_cache:
            # target_length = past_key_values.get_max_cache_shape()
            target_length = None
        else:
            target_length = (
                tuple(attention_mask.shape)[-1]
                if isinstance(attention_mask, paddle.Tensor)
                else past_seen_tokens + sequence_length + 1
            )
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=tuple(input_tensor.shape)[0],
            config=self.config,
            past_key_values=past_key_values,
        )
        if self._attn_implementation == "sdpa" and attention_mask is not None and not output_attentions:
            causal_mask = self._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: paddle.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: paddle.dtype,
        cache_position: paddle.Tensor,
        batch_size: int,
        config: Qwen2_5OmniConfig,
        past_key_values: Tuple[paddle.Tensor],
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen25OmniTalkerConfig`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = paddle.finfo(dtype=dtype).min
            causal_mask = paddle.full(shape=(sequence_length, target_length), fill_value=min_dtype, dtype=dtype)
            diagonal_attend_mask = paddle.arange(end=target_length) > cache_position.reshape([-1, 1])
            if config.sliding_window is not None:
                # TODO
                # if not isinstance(past_key_values, SlidingWindowCache
                #     ) or sequence_length > target_length:
                if sequence_length > target_length:
                    sliding_attend_mask = (
                        paddle.arange(end=target_length) <= cache_position.reshape([-1, 1]) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(y=sliding_attend_mask)

            causal_mask *= diagonal_attend_mask.astype(causal_mask.dtype)
            causal_mask = causal_mask[None, None, :, :].expand(shape=[batch_size, 1, -1, -1])
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if tuple(attention_mask.shape)[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = tuple(attention_mask.shape)[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].astype(
                    causal_mask.dtype
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    mask=padding_mask, value=min_dtype
                )
        return causal_mask

    @staticmethod
    def _ignore_causal_mask_sdpa(
        attention_mask: Optional[paddle.Tensor],
        inputs_embeds: paddle.Tensor,
        past_key_values_length: int,
        sliding_window: Optional[int] = None,
        is_training: bool = False,
    ) -> bool:
        """
        Detects whether the optional user-specified attention_mask & the automatically created causal mask can be
        ignored in case PyTorch's SDPA is used, rather relying on SDPA's `is_causal` argument.

        In case no token is masked in the `attention_mask` argument, if `query_length == 1` or
        `key_value_length == query_length`, we rather rely on SDPA `is_causal` argument to use causal/non-causal masks,
        allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is
        passed).
        """

        _, query_length = inputs_embeds.shape[0], inputs_embeds.shape[1]
        key_value_length = query_length + past_key_values_length
        ignore_causal_mask = False

        if attention_mask is None:
            if (
                is_training
                and (query_length == 1 or key_value_length == query_length)
                and (sliding_window is None or key_value_length < sliding_window)
            ):
                ignore_causal_mask = True
        elif sliding_window is None or key_value_length < sliding_window:
            if len(attention_mask.shape) == 4:
                return False
            elif paddle.all(attention_mask == 1):
                if query_length == 1 or key_value_length == query_length:
                    # For query_length == 1, causal attention and bi-directional attention are the same.
                    ignore_causal_mask = True

                # Unfortunately, for query_length > 1 and key_value_length != query_length, we cannot generally ignore
                # the attention mask, as SDPA causal mask generation may be wrong. We will set `is_causal=False` in
                # SDPA and rely on Transformers attention_mask instead, hence not setting it to None here.
                # Reference: https://github.com/pytorch/pytorch/issues/108108
                # TODO: maybe revisit this with https://github.com/pytorch/pytorch/pull/114823 in PyTorch 2.3.

        return ignore_causal_mask


class Qwen2_5OmniTalkerForConditionalGeneration(Qwen2_5OmniPreTrainedModelForConditionalGeneration):
    config_class = Qwen2_5OmniTalkerConfig

    def __init__(self, config: Qwen2_5OmniTalkerConfig):
        super().__init__(config)
        self.thinker_to_talker_proj = paddle.nn.Linear(
            in_features=config.embedding_size, out_features=config.hidden_size
        )
        self.model = Qwen2_5OmniTalkerModel(config)
        self.codebook_size = config.vocab_size
        self.codec_head = paddle.nn.Linear(
            in_features=config.hidden_size, out_features=self.codebook_size, bias_attr=False
        )
        self.codec_bos_token = config.tts_codec_start_token_id
        self.codec_eos_token = config.tts_codec_end_token_id
        self.codec_pad_token = config.tts_codec_pad_token_id
        self.codec_mask_token = config.tts_codec_mask_token_id
        self.text_bos_token = config.tts_text_start_token_id
        self.text_eos_token = config.tts_text_end_token_id
        self.text_pad_token = config.tts_text_pad_token_id
        self.spatial_merge_size = self.config.spatial_merge_size
        self.rope_deltas = None
        # self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        thinker_reply_part: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        rope_deltas: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[paddle.Tensor] = None,
        input_text_ids: Optional[paddle.Tensor] = None,
        image_grid_thw: Optional[paddle.Tensor] = None,
        video_grid_thw: Optional[paddle.Tensor] = None,
        use_audio_in_video: Optional[bool] = None,
        audio_feature_lengths: Optional[paddle.Tensor] = None,
        video_second_per_grid: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Qwen2_5OmniTalkerCausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from io import BytesIO
        >>> from urllib.request import urlopen
        >>> import librosa
        >>> from transformers import AutoProcessor, Qwen2_5OmniTalkerForConditionalGeneration

        >>> model = Qwen2_5OmniTalkerForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B")

        >>> prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
        >>> url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
        >>> audio, _ = librosa.load(BytesIO(urlopen(url).read()), sr=self.processor.feature_extractor.sampling_rate)

        >>> inputs = processor(text=prompt, audios=audio, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Generate the caption in English: Glass is breaking."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # import pdb;pdb.set_trace()
        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or cache_position is not None
                and cache_position[0] == 0
                or self.rope_deltas is None
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_text_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                inputs_embeds[:, -1, :] += self.get_input_embeddings()(
                    paddle.to_tensor(data=[self.codec_bos_token], dtype="int64", place=inputs_embeds.place)
                )
                inputs_embeds[:, -2, :] += self.get_input_embeddings()(
                    paddle.to_tensor(data=[self.codec_pad_token], dtype="int64", place=inputs_embeds.place)
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = tuple(input_ids.shape)
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = paddle.arange(end=seq_length)
                position_ids = position_ids.view([1, -1]).expand(shape=[batch_size, -1])
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(axis=0).expand(shape=[3, -1, -1])
        if inputs_embeds is None:
            codec_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds = codec_embeds + thinker_reply_part[:, :1, :]
            if tuple(thinker_reply_part.shape)[1] > 1:
                thinker_reply_part = thinker_reply_part[:, 1:, :]
        talker_lm_input = self.thinker_to_talker_proj(inputs_embeds)
        # import pdb;pdb.set_trace()
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=talker_lm_input,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.codec_head(hidden_states)
        logits = logits.astype(dtype="float32")
        loss = None
        if not return_dict:
            output = (logits, thinker_reply_part) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return Qwen2_5OmniTalkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            attention_mask=attention_mask,
            rope_deltas=self.rope_deltas,
            thinker_reply_part=thinker_reply_part,
        )

    def _get_initial_cache_position(self, input_ids, model_kwargs):
        inputs_embeds = model_kwargs.pop("inputs_embeds")
        model_kwargs = super()._get_initial_cache_position(input_ids, model_kwargs)
        model_kwargs["inputs_embeds"] = inputs_embeds
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        input_text_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        thinker_reply_part=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_audio_features=None,
        audio_feature_attention_mask=None,
        audio_feature_lengths=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs
    ):
        if use_cache:
            if past_key_values is None or past_key_values[0][0] is None:
                cache_position = paddle.arange(0, len(input_ids))

            # trunc input when use kv cache and set cache_postion
            if past_key_values is not None and past_key_values[0][0] is not None:
                cache_position = paddle.arange(past_key_values[0][0].shape[2], past_key_values[0][0].shape[2] + 1)
                trim_input_ids = input_ids[:, -1:]
            else:
                trim_input_ids = input_ids
        else:
            trim_input_ids = input_ids
        model_inputs = dict(
            input_ids=trim_input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=use_cache,
            thinker_reply_part=thinker_reply_part,
            input_text_ids=input_text_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_audio_in_video=use_audio_in_video,
            audio_feature_lengths=audio_feature_lengths,
            video_second_per_grid=video_second_per_grid,
            position_ids=position_ids,
        )
        model_inputs["position_ids"] = None

        return model_inputs

    def update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        if getattr(outputs, "attention_mask", None) is not None:
            model_kwargs["attention_mask"] = outputs.attention_mask

        model_kwargs = super().update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder)

        past_key_values = None
        # kv cache
        if model_kwargs["use_cache"] and isinstance(outputs, ModelOutput):
            if getattr(outputs, "past_key_values", None) is not None:
                past_key_values = outputs.past_key_values

        elif isinstance(outputs, Tuple):
            if model_kwargs.get("output_hidden_states", False):
                past_key_values = outputs[-2]
            else:
                past_key_values = outputs[-1]

        model_kwargs["past_key_values"] = past_key_values

        if isinstance(outputs, ModelOutput) and getattr(outputs, "thinker_reply_part", None) is not None:
            model_kwargs["thinker_reply_part"] = outputs.thinker_reply_part
        elif isinstance(outputs, Tuple):
            model_kwargs["thinker_reply_part"] = outputs[1]

        # clean inputs_embeds after first token generated
        if model_kwargs.get("inputs_embeds", None) is not None:
            model_kwargs["inputs_embeds"] = None
        return model_kwargs

    def get_logits_processor(
        self,
        min_length=None,
        max_length=None,
        eos_token_id=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        num_beams=1,
        num_beam_groups=1,
        diversity_rate=0.0,
        repetition_penalty=None,
        no_repeat_ngram_size=None,
        logits_processors=None,
    ):
        processors = super().get_logits_processor(
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_bos_token_id,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_rate=diversity_rate,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            logits_processors=None,
        )
        for p in logits_processors._processors.values():
            processors.append(p)
        return processors


class RotaryEmbedding(paddle.nn.Layer):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / base ** (paddle.arange(start=0, end=dim, step=2).astype(dtype="float32") / dim)
        self.register_buffer(name="inv_freq", tensor=inv_freq)

    def forward(self, x):
        batch_size, seq_len = tuple(x.shape)[0], tuple(x.shape)[1]
        t = paddle.arange(dtype=self.inv_freq.dtype, end=seq_len)
        freqs = paddle.einsum("i , j -> i j", t.astype(dtype=self.inv_freq.dtype), self.inv_freq)
        freqs = paddle.stack(x=(freqs, freqs), axis=-1)
        freqs = freqs.reshape(freqs.shape[:-2] + [-1])
        freqs = freqs.tile(repeat_times=[batch_size, *([1] * freqs.dim())])
        return freqs.cos(), freqs.sin()


class TDNNBlock(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = paddle.nn.Conv1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
            padding_mode="zeros",
        )  # reflect -> zeros
        self.activation = paddle.nn.ReLU()

    def forward(self, x):
        return self.activation(self.conv(x))


class Res2NetBlock(paddle.nn.Layer):
    """An implementation of Res2NetBlock w/ dilation.

    Arguments
    ---------
    in_channels : int
        The number of channels expected in the input.
    out_channels : int
        The number of output channels.
    scale : int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the Res2Net block.
    dilation : int
        The dilation of the Res2Net block.
    """

    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.blocks = paddle.nn.LayerList(
            sublayers=[
                TDNNBlock(in_channel, hidden_channel, kernel_size=kernel_size, dilation=dilation)
                for i in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, hidden):
        outputs = []
        for i, hidden_part in enumerate(paddle.chunk(x=hidden, chunks=self.scale, axis=1)):
            if i == 0:
                output_part = hidden_part
            elif i == 1:
                output_part = self.blocks[i - 1](hidden_part)
            else:
                output_part = self.blocks[i - 1](hidden_part + output_part)
            outputs.append(output_part)
        output = paddle.concat(x=outputs, axis=1)
        return output


class SEBlock(paddle.nn.Layer):
    """An implementation of squeeze-and-excitation block.

    Arguments
    ---------
    in_channels : int
        The number of input channels.
    se_channels : int
        The number of output channels after squeeze.
    out_channels : int
        The number of output channels.
    """

    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()
        self.conv1 = paddle.nn.Conv1D(
            in_channels=in_channels, out_channels=se_channels, kernel_size=1, padding="same", padding_mode="zeros"
        )
        self.relu = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv1D(
            in_channels=se_channels, out_channels=out_channels, kernel_size=1, padding="same", padding_mode="zeros"
        )
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, x):
        s = x.mean(axis=2, keepdim=True)
        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))
        return s * x


class AttentiveStatisticsPooling(paddle.nn.Layer):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.

    Arguments
    ---------
    channels: int
        The number of input channels.
    attention_channels: int
        The number of attention channels.
    """

    def __init__(self, channels, attention_channels=128):
        super().__init__()
        self.eps = 1e-12
        self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = paddle.nn.Tanh()
        self.conv = paddle.nn.Conv1D(
            in_channels=attention_channels, out_channels=channels, kernel_size=1, padding="same", padding_mode="zeros"
        )

    def _length_to_mask(self, length, max_len=None, dtype=None, device=None):
        """Creates a binary mask for each sequence.

        Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3

        Arguments
        ---------
        length : torch.LongTensor
            Containing the length of each sequence in the batch. Must be 1D.
        max_len : int
            Max length for the mask, also the size of the second dimension.
        dtype : torch.dtype, default: None
            The dtype of the generated mask.
        device: torch.device, default: None
            The device to put the mask variable.

        Returns
        -------
        mask : tensor
            The binary mask.
        """
        if max_len is None:
            max_len = length.max().astype(dtype="int64").item()
        mask = paddle.arange(dtype=length.dtype, end=max_len).expand(shape=[len(length), max_len]) < length.unsqueeze(
            axis=1
        )
        mask = paddle.to_tensor(data=mask, dtype=dtype, place=device)
        return mask

    def _compute_statistics(self, x, m, dim=2):
        mean = (m * x).sum(axis=dim)
        std = paddle.sqrt(x=(m * (x - mean.unsqueeze(axis=dim)).pow(y=2)).sum(axis=dim).clip(min=self.eps))
        return mean, std

    def forward(self, x):
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape [N, C, L].
        """
        L = tuple(x.shape)[-1]
        lengths = paddle.ones(shape=tuple(x.shape)[0])
        mask = self._length_to_mask(lengths * L, max_len=L, dtype=x.dtype, device=x.place)
        mask = mask.unsqueeze(axis=1)
        total = mask.sum(axis=2, keepdim=True)
        mean, std = self._compute_statistics(x, mask / total)
        mean = mean.unsqueeze(axis=2).tile(repeat_times=[1, 1, L])
        std = std.unsqueeze(axis=2).tile(repeat_times=[1, 1, L])
        attn = paddle.concat(x=[x, mean, std], axis=1)
        attn = self.conv(self.tanh(self.tdnn(attn)))
        attn = attn.masked_fill(mask=mask == 0, value=float("-inf"))
        attn = paddle.nn.functional.softmax(x=attn, axis=2)
        mean, std = self._compute_statistics(x, attn)
        pooled_stats = paddle.concat(x=(mean, std), axis=1)
        pooled_stats = pooled_stats.unsqueeze(axis=2)
        return pooled_stats


class SERes2NetBlock(paddle.nn.Layer):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SEBlock.

    Arguments
    ----------
    out_channels: int
        The number of output channels.
    res2net_scale: int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the TDNN blocks.
    dilation: int
        The dilation of the Res2Net block.
    activation : torch class
        A class for constructing the activation layers.
    """

    def __init__(self, in_channels, out_channels, res2net_scale=8, se_channels=128, kernel_size=1, dilation=1):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(in_channels, out_channels, kernel_size=1, dilation=1)
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TDNNBlock(out_channels, out_channels, kernel_size=1, dilation=1)
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

    def forward(self, x):
        residual = x
        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x)
        return x + residual


class ECAPA_TDNN(paddle.nn.Layer):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).

    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    activation : torch class
        A class for constructing the activation layers.
    channels : list of ints
        Output channels for TDNN/SERes2Net layer.
    kernel_sizes : list of ints
        List of kernel sizes for each layer.
    dilations : list of ints
        List of dilations for kernels in each layer.
    lin_neurons : int
        Number of neurons in linear layers.
    """

    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__()
        if len(config.enc_channels) != len(config.enc_kernel_sizes) or len(config.enc_channels) != len(
            config.enc_dilations
        ):
            raise ValueError("enc_channels, enc_kernel_sizes and enc_dilations should have same length")
        self.channels = config.enc_channels
        self.blocks = paddle.nn.LayerList()
        self.blocks.append(
            TDNNBlock(config.mel_dim, config.enc_channels[0], config.enc_kernel_sizes[0], config.enc_dilations[0])
        )
        for i in range(1, len(config.enc_channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    config.enc_channels[i - 1],
                    config.enc_channels[i],
                    res2net_scale=config.enc_res2net_scale,
                    se_channels=config.enc_se_channels,
                    kernel_size=config.enc_kernel_sizes[i],
                    dilation=config.enc_dilations[i],
                )
            )
        self.mfa = TDNNBlock(
            config.enc_channels[-1], config.enc_channels[-1], config.enc_kernel_sizes[-1], config.enc_dilations[-1]
        )
        self.asp = AttentiveStatisticsPooling(
            config.enc_channels[-1], attention_channels=config.enc_attention_channels
        )
        self.fc = paddle.nn.Conv1D(
            in_channels=config.enc_channels[-1] * 2,
            out_channels=config.enc_dim,
            kernel_size=1,
            padding="same",
            padding_mode="zeros",
        )

    def forward(self, x):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        x = x.transpose(perm=[0, 2, 1])
        xl = []
        for layer in self.blocks:
            x = layer(x)
            xl.append(x)
        x = paddle.concat(x=xl[1:], axis=1)
        x = self.mfa(x)
        x = self.asp(x)
        x = self.fc(x)
        x = x.squeeze(axis=-1)
        return x


class InputEmbedding(paddle.nn.Layer):
    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__()
        self.proj = paddle.nn.Linear(
            in_features=config.mel_dim + config.enc_dim + config.enc_emb_dim + config.emb_dim,
            out_features=config.hidden_size,
        )
        self.spk_encoder = ECAPA_TDNN(config)

    def forward(self, x, spk, cond, code_embed, drop_audio_cond=False, code_embed_uncond=None, cfg=True):
        if cfg:
            x = paddle.concat(x=[x, x], axis=0)
            spk = paddle.concat(x=[spk, paddle.zeros_like(x=spk)], axis=0)
            cond = paddle.concat(x=[cond, paddle.zeros_like(x=cond)], axis=0)
            code_embed = paddle.concat(x=[code_embed, code_embed_uncond], axis=0)
        elif drop_audio_cond:
            cond = paddle.zeros_like(x=cond)
            spk = paddle.zeros_like(x=spk)
        cond = self.spk_encoder(cond).unsqueeze(axis=1).tile(repeat_times=[1, x.shape[1], 1])
        x = self.proj(paddle.concat(x=[x, cond, code_embed, spk], axis=-1))
        return x


class CodecEmbedding(paddle.nn.Layer):
    def __init__(self, codec_num_embeds, codec_dim, repeats):
        super().__init__()
        self.repeats = repeats
        self.codec_embed = paddle.nn.Embedding(num_embeddings=codec_num_embeds + 1, embedding_dim=codec_dim)

    def forward(self, code, drop_code=False):
        if drop_code:
            code = paddle.zeros_like(x=code)
        code_embed = self.codec_embed(code)
        code_embed = paddle.repeat_interleave(x=code_embed, repeats=self.repeats, axis=1)
        return code_embed


class AdaLayerNormZero(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.silu = paddle.nn.Silu()
        self.linear = paddle.nn.Linear(in_features=dim, out_features=dim * 6)
        self.norm = paddle.nn.LayerNorm(normalized_shape=dim, weight_attr=False, bias_attr=False, epsilon=1e-06)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = paddle.chunk(x=emb, chunks=6, axis=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZero_Final(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.silu = paddle.nn.Silu()
        self.linear = paddle.nn.Linear(in_features=dim, out_features=dim * 2)
        self.norm = paddle.nn.LayerNorm(normalized_shape=dim, weight_attr=False, bias_attr=False, epsilon=1e-06)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = paddle.chunk(x=emb, chunks=2, axis=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class FeedForward(paddle.nn.Layer):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.ff = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.Linear(in_features=dim, out_features=inner_dim),
                paddle.nn.GELU(approximate=True),
                paddle.nn.Dropout(p=dropout),
                paddle.nn.Linear(in_features=inner_dim, out_features=dim),
            ]
        )

    def forward(self, x):
        for layer in self.ff:
            x = layer(x)
        return x


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    def rotate_half_codec(x):
        x = x.reshape([*tuple(x.shape)[:-1], -1, 2])
        x1, x2 = x.unbind(axis=-1)
        x = paddle.stack(x=(-x2, x1), axis=-1)
        return x.reshape([*tuple(x.shape)[:-2], -1])

    cos = cos.unsqueeze(axis=unsqueeze_dim)
    sin = sin.unsqueeze(axis=unsqueeze_dim)
    q_embed = q * cos + rotate_half_codec(q) * sin
    k_embed = k * cos + rotate_half_codec(k) * sin
    return q_embed, k_embed


class DiTAttention(paddle.nn.Layer):
    def __init__(self, config: Qwen2_5OmniDiTConfig):
        super().__init__()
        self.config = config
        self.dim = config.hidden_size
        self.heads = config.num_attention_heads
        self.inner_dim = config.head_dim * config.num_attention_heads
        self.dropout = config.dropout
        self._attn_implementation = config.get("_attn_implementation", "eager")
        self.is_causal = False
        self.to_q = paddle.nn.Linear(in_features=config.hidden_size, out_features=self.inner_dim)
        self.to_k = paddle.nn.Linear(in_features=config.hidden_size, out_features=self.inner_dim)
        self.to_v = paddle.nn.Linear(in_features=config.hidden_size, out_features=self.inner_dim)
        self.to_out = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.Linear(in_features=self.inner_dim, out_features=config.hidden_size),
                paddle.nn.Dropout(p=config.dropout),
            ]
        )

    def forward(self, x, rope=None, mask=None) -> paddle.Tensor:
        batch_size = tuple(x.shape)[0]
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)
        inner_dim = tuple(key.shape)[-1]
        head_dim = inner_dim // self.heads
        query = query.view([batch_size, -1, self.heads, head_dim]).transpose([0, 2, 1, 3])
        key = key.view([batch_size, -1, self.heads, head_dim]).transpose([0, 2, 1, 3])
        value = value.view([batch_size, -1, self.heads, head_dim]).transpose([0, 2, 1, 3])  # b h l d

        cos, sin = rope
        query[:, :1], key[:, :1] = apply_rotary_pos_emb(query[:, :1], key[:, :1], cos, sin)

        # TODO: Fix attention func (liaojincheng)
        attention_interface = ALL_ATTENTION_FUNCTIONS[self._attn_implementation]
        x = attention_interface(self, query, key, value, attn_mask=mask, is_causal=False)

        x = x.reshape([batch_size, -1, self.heads * head_dim])
        x = x.astype(query.dtype)
        x = self.to_out[0](x)
        x = self.to_out[1](x)
        return x


class SinusPositionEmbedding(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = paddle.exp(x=paddle.arange(end=half_dim).astype(dtype="float32") * -emb)
        emb = scale * x.unsqueeze(axis=1) * emb.unsqueeze(axis=0)
        emb = paddle.concat(x=(emb.sin(), emb.cos()), axis=-1)
        return emb.astype(dtype=x.dtype)


class TimestepEmbedding(paddle.nn.Layer):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.Linear(in_features=freq_embed_dim, out_features=dim),
                paddle.nn.Silu(),
                paddle.nn.Linear(in_features=dim, out_features=dim),
            ]
        )

    def forward(self, timestep):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.astype(timestep.dtype)
        for layer in self.time_mlp:
            time_hidden = layer(time_hidden)
        return time_hidden


class DiTBlock(paddle.nn.Layer):
    def __init__(self, config: Qwen2_5OmniDiTConfig, look_ahead_block=0, look_backward_block=0):
        super().__init__()
        self.attn_norm = AdaLayerNormZero(config.hidden_size)
        self.attn = DiTAttention(config)
        self.look_ahead_block = look_ahead_block
        self.look_backward_block = look_backward_block
        self.ff_norm = paddle.nn.LayerNorm(
            normalized_shape=config.hidden_size, weight_attr=False, bias_attr=False, epsilon=1e-06
        )
        self.ff = FeedForward(dim=config.hidden_size, mult=config.ff_mult, dropout=config.dropout)

    def forward(self, x, t, rope=None, block_diff=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(
            x=norm,
            rope=rope,
            mask=(block_diff >= -float(self.look_backward_block)) & (block_diff <= float(self.look_ahead_block)),
        )
        x = x + gate_msa.unsqueeze(axis=1) * attn_output
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(axis=1) * ff_output
        return x


class SnakeBeta(paddle.nn.Layer):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    """

    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.alpha = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.zeros(shape=in_features) * alpha)
        self.beta = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.zeros(shape=in_features) * alpha)
        self.no_div_by_zero = 1e-09

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(axis=0).unsqueeze(axis=-1)
        beta = self.beta.unsqueeze(axis=0).unsqueeze(axis=-1)
        alpha = paddle.exp(x=alpha)
        beta = paddle.exp(x=beta)
        x = x + 1.0 / (beta + self.no_div_by_zero) * paddle.pow(x=paddle.sin(x=x * alpha), y=2)
        return x


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
        # TODO 算子缺失
    window = kaiser_window(kernel_size, beta=beta, periodic=False, dtype="float32")
    if even:
        time = paddle.arange(start=-half_size, end=half_size) + 0.5
    else:
        time = paddle.arange(end=kernel_size) - half_size
    if cutoff == 0:
        filter_ = paddle.zeros_like(x=time)
    else:
        filter_ = 2 * cutoff * window * paddle.sinc(x=2 * cutoff * time)
        filter_ /= filter_.sum()
        filter = filter_.view([1, 1, kernel_size])
    return filter


class UpSample1d(paddle.nn.Layer):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer(name="filter", tensor=filter, persistable=False)

    def forward(self, x):
        _, C, _ = tuple(x.shape)
        x = paddle.nn.functional.pad(x=x, pad=(self.pad, self.pad), mode="replicate", pad_from_left_axis=False)
        x = self.ratio * paddle.nn.functional.conv1d_transpose(
            x=x, weight=self.filter.expand(shape=[C, -1, -1]), stride=self.stride, groups=C
        )
        x = x[..., self.pad_left : -self.pad_right]
        return x


class DownSample1d(paddle.nn.Layer):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = ratio
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer(name="filter", tensor=filter, persistable=False)

    def forward(self, x):
        _, C, _ = tuple(x.shape)
        x = paddle.nn.functional.pad(
            x=x, pad=(self.pad_left, self.pad_right), mode="replicate", pad_from_left_axis=False
        )
        out = paddle.nn.functional.conv1d(
            x=x, weight=self.filter.expand(shape=[C, -1, -1]), stride=self.stride, groups=C
        )
        return out


class TorchActivation1d(paddle.nn.Layer):
    def __init__(
        self, activation, up_ratio: int = 2, down_ratio: int = 2, up_kernel_size: int = 12, down_kernel_size: int = 12
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


class AMPBlock(paddle.nn.Layer):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.Conv1D(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[0],
                    padding=self._get_padding(kernel_size, dilation[0]),
                ),
                paddle.nn.Conv1D(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[1],
                    padding=self._get_padding(kernel_size, dilation[1]),
                ),
                paddle.nn.Conv1D(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[2],
                    padding=self._get_padding(kernel_size, dilation[2]),
                ),
            ]
        )
        self.convs2 = paddle.nn.LayerList(
            sublayers=[
                paddle.nn.Conv1D(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1),
                ),
                paddle.nn.Conv1D(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1),
                ),
                paddle.nn.Conv1D(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1),
                ),
            ]
        )
        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = paddle.nn.LayerList(
            sublayers=[TorchActivation1d(activation=SnakeBeta(channels)) for _ in range(self.num_layers)]
        )

    def _get_padding(self, kernel_size, dilation=1):
        return int((kernel_size * dilation - dilation) / 2)

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x


class Qwen2_5OmniToken2WavBigVGANModel(Qwen2_5OmniPreTrainedModel):
    config_class = Qwen2_5OmniBigVGANConfig

    def __init__(self, config: Qwen2_5OmniBigVGANConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = paddle.nn.Conv1D(
            in_channels=config.mel_dim,
            out_channels=config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        self.ups = paddle.nn.LayerList()
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.ups.append(
                paddle.nn.LayerList(
                    sublayers=[
                        paddle.nn.Conv1DTranspose(
                            in_channels=config.upsample_initial_channel // 2**i,
                            out_channels=config.upsample_initial_channel // 2 ** (i + 1),
                            kernel_size=k,
                            stride=u,
                            padding=(k - u) // 2,
                        )
                    ]
                )
            )
        self.resblocks = paddle.nn.LayerList()
        for i in range(len(self.ups)):
            ch = config.upsample_initial_channel // 2 ** (i + 1)
            for j, (k, d) in enumerate(zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)):
                self.resblocks.append(AMPBlock(ch, k, d))
        self.activation_post = TorchActivation1d(activation=SnakeBeta(ch))
        self.conv_post = paddle.nn.Conv1D(
            in_channels=ch, out_channels=1, kernel_size=7, stride=1, padding=3, bias_attr=False
        )

    def _normalize(self, S, max_abs_value, min_db):
        return paddle.clip(
            x=2 * max_abs_value * ((S - min_db) / -min_db) - max_abs_value, min=-max_abs_value, max=max_abs_value
        )

    def _amp_to_db(self, x, min_level_db):
        min_level = np.exp(min_level_db / 20 * np.log(10))
        min_level = paddle.ones_like(x=x) * min_level
        return 20 * paddle.log10(x=paddle.maximum(x=min_level, y=x))

    def apm_to_db(self, apm_mel):
        mel_spec = paddle.exp(x=apm_mel)
        mel_spec = self._amp_to_db(mel_spec, -115) - 20
        mel_spec = self._normalize(mel_spec, 1, -115)
        return mel_spec

    def forward(self, apm_mel):
        mel_spec = self.apm_to_db(apm_mel)
        hidden = self.conv_pre(mel_spec)
        for i in range(self.num_upsamples):
            for i_up in range(len(self.ups[i])):
                hidden = self.ups[i][i_up](hidden)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](hidden)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](hidden)
            hidden = xs / self.num_kernels
        hidden = self.activation_post(hidden)
        hidden = self.conv_post(hidden)
        audio = paddle.clip(x=hidden, min=-1.0, max=1.0)
        return audio.squeeze().cpu()


class ODESolverRK4:
    def __init__(self, func, y0):
        self.func = func
        self.y0 = y0
        self._one_third = 1 / 3
        self._two_thirds = 2 / 3

    def _rk4_alt_step_func(self, func, t0, dt, t1, y0, f0=None):
        k1 = f0
        if k1 is None:
            k1 = func(t0, y0)
        k2 = func(t0 + dt * self._one_third, y0 + dt * k1 * self._one_third)
        k3 = func(t0 + dt * self._two_thirds, y0 + dt * (k2 - k1 * self._one_third))
        k4 = func(t1, y0 + dt * (k1 - k2 + k3))
        return (k1 + 3 * (k2 + k3) + k4) * dt * 0.125

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0)
        return self._rk4_alt_step_func(func, t0, dt, t1, y0, f0=f0), f0

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)

    def integrate(self, t):
        solution = paddle.empty(shape=[len(t), *tuple(self.y0.shape)], dtype=self.y0.dtype)
        solution[0] = self.y0
        j = 1
        y0 = self.y0
        for t0, t1 in zip(t[:-1], t[1:]):
            dt = t1 - t0
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy
            while j < len(t) and t1 >= t[j]:
                solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                j += 1
            y0 = y1

            # del y1
            # del f0
            # paddle.device.cuda.empty_cache()
        return solution


class Qwen2_5OmniToken2WavDiTModel(Qwen2_5OmniPreTrainedModel):
    config_class = Qwen2_5OmniDiTConfig
    _no_split_modules = ["DiTBlock"]

    def __init__(self, config: Qwen2_5OmniDiTConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.mel_dim = config.mel_dim
        self.repeats = config.repeats
        self.time_embed = TimestepEmbedding(config.hidden_size)
        self.text_embed = CodecEmbedding(config.num_embeds, config.emb_dim, config.repeats)
        self.input_embed = InputEmbedding(config)
        self.rotary_embed = RotaryEmbedding(config.head_dim)
        self.hidden_size = config.hidden_size
        self.layers = config.num_hidden_layers
        self.block_size = config.block_size
        self.num_attention_heads = config.num_attention_heads
        self.transformer_blocks = paddle.nn.LayerList()
        for i in range(config.num_hidden_layers):
            self.transformer_blocks.append(
                DiTBlock(
                    config,
                    look_ahead_block=1 if i in config.look_ahead_layers else 0,
                    look_backward_block=1 if i in config.look_backward_layers else 0,
                )
            )
        self.norm_out = AdaLayerNormZero_Final(config.hidden_size)
        self.proj_out = paddle.nn.Linear(in_features=config.hidden_size, out_features=config.mel_dim)

    def _create_block_diff(self, x):
        batch, seq_len = tuple(x.shape)[0], tuple(x.shape)[1]
        block_indices = paddle.arange(end=seq_len) // self.block_size
        block_i = block_indices.unsqueeze(axis=1)
        block_j = block_indices.unsqueeze(axis=0)
        block_diff = block_j - block_i
        return block_diff.expand(shape=[batch, self.num_attention_heads, seq_len, seq_len])

    def forward(self, x, cond, spk, code, time, drop_audio_cond=False, drop_code=False, cfg=True):
        batch = tuple(x.shape)[0]
        if time.ndim == 0:
            time = time.tile(repeat_times=batch)
        t = self.time_embed(time)
        code_embed = self.text_embed(code, drop_code=False if cfg else drop_code)
        code_embed_uncond = self.text_embed(code, drop_code=True) if cfg else None
        hidden = self.input_embed(
            x, spk, cond, code_embed, drop_audio_cond=drop_audio_cond, code_embed_uncond=code_embed_uncond, cfg=cfg
        )
        rope = self.rotary_embed(hidden)
        block_diff = self._create_block_diff(hidden)
        for block in self.transformer_blocks:
            hidden = block(hidden, t, rope=rope, block_diff=block_diff)
        hidden = self.norm_out(hidden, t)
        output = self.proj_out(hidden)
        return output

    @paddle.no_grad()
    def sample(self, cond, ref_mel, code, steps=10, cfg_strength=0.5, sway_sampling_coef=-1.0):
        y_all = paddle.randn(shape=[1, 30000, self.mel_dim], dtype=ref_mel.dtype)
        max_duration = tuple(code.shape)[1] * self.repeats
        y0 = y_all[:, :max_duration]
        batch = tuple(ref_mel.shape)[0]
        cond = cond.unsqueeze(axis=1).tile(repeat_times=[1, max_duration, 1])
        if batch != 1:
            raise ValueError("only support batch size = 1 currently")

        def fn(t, x):
            if cfg_strength < 1e-05:
                pred = self(x=x, spk=cond, cond=ref_mel, code=code, time=t, drop_audio_cond=False, drop_code=False)
                return pred
            out_put = self(x=x, code=code, spk=cond, cond=ref_mel, time=t, cfg=True)
            pred, null_pred = paddle.chunk(x=out_put, chunks=2, axis=0)
            return pred + (pred - null_pred) * cfg_strength

        t_start = 0
        t = paddle.linspace(start=t_start, stop=1, num=steps, dtype=cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (paddle.cos(x=np.pi / 2 * t) - 1 + t)
        solver = ODESolverRK4(func=fn, y0=y0)
        trajectory = solver.integrate(t)
        generated = trajectory[-1]
        generated_mel_spec = generated.transpose(perm=[0, 2, 1])
        return generated_mel_spec


class Qwen2_5OmniToken2WavModel(Qwen2_5OmniPreTrainedModel):
    config_class = Qwen2_5OmniToken2WavConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen2_5OmniToken2WavDiTModel", "Qwen2_5OmniToken2WavBigVGANModel"]

    def __init__(self, config: Qwen2_5OmniToken2WavConfig):
        super().__init__(config)
        attn_impl = config.get("_attn_implementation", "eager")
        config.dit_config.dtype = "float32"  # TODO Check Dtype
        config.bigvgan_config.dtype = "float32"  # TODO Check Dtype

        self.code2wav_dit_model = Qwen2_5OmniToken2WavDiTModel._from_config(
            config.dit_config, attn_implementation=attn_impl
        )
        self.code2wav_bigvgan_model = Qwen2_5OmniToken2WavBigVGANModel._from_config(
            config.bigvgan_config, attn_implementation=attn_impl
        )

    def forward(self, code, cond, ref_mel, steps=10, cfg_strength=0.5, sway_sampling_coef=-1.0, **kwargs):
        generated_mel = self.code2wav_dit_model.sample(
            cond, ref_mel, code, steps=steps, cfg_strength=cfg_strength, sway_sampling_coef=sway_sampling_coef
        )
        waveform = self.code2wav_bigvgan_model(generated_mel)
        return waveform


class Qwen2_5OmniModel(Qwen2_5OmniPreTrainedModel):
    config_class = Qwen2_5OmniConfig
    _no_split_modules = ["Qwen2_5OmniTalkerForConditionalGeneration", "Qwen2_5OmniToken2WavModel"]

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        # merge attn_implementation
        if kwargs.get("attn_implementation", None) is not None:
            config.thinker_config.vision_config._attn_implementation = kwargs["attn_implementation"]
            config.thinker_config.text_config._attn_implementation = kwargs["attn_implementation"]
            config.thinker_config.audio_config._attn_implementation = kwargs["attn_implementation"]
            config.talker_config._attn_implementation = kwargs["attn_implementation"]
            config.token2wav_config.bigvgan_config._attn_implementation = kwargs["attn_implementation"]

        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration(config.thinker_config)
        self.has_talker = config.enable_audio_output
        self.speaker_map = {}
        if kwargs.get("enable_audio_output", True) and config.enable_audio_output:
            self.enable_talker()

    def enable_talker(self):
        self.talker = Qwen2_5OmniTalkerForConditionalGeneration(self.config.talker_config)
        self.token2wav = Qwen2_5OmniToken2WavModel(self.config.token2wav_config)
        self.token2wav.astype(dtype="float32")
        self.has_talker = True

    def load_speakers(self, path):
        for key, value in paddle.load(path=str(path)).items():
            self.speaker_map[key] = value
        logger.info("Speaker {} loaded".format(list(self.speaker_map.keys())))

    def disable_talker(self):
        if hasattr(self, "talker"):
            del self.talker
        if hasattr(self, "token2wav"):
            del self.token2wav
        self.has_talker = False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        cache_dir=None,
        ignore_mismatched_sizes=False,
        force_download=False,
        local_files_only=False,
        token=None,
        revision="main",
        use_safetensors=None,
        weights_only=True,
        **kwargs
    ):
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs,
        )
        spk_path = cached_file(
            pretrained_model_name_or_path,
            "spk_dict.pdparams",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
        )
        if spk_path is None:
            raise ValueError(f"{pretrained_model_name_or_path}/{spk_path} not exists")
        model.load_speakers(spk_path)
        return model

    @paddle.no_grad()
    def generate(
        self,
        input_ids: Optional[paddle.to_tensor] = None,
        spk: str = "Chelsie",
        use_audio_in_video: bool = False,
        return_audio: Optional[bool] = None,
        thinker_max_new_tokens: int = 1024,
        talker_max_new_tokens: int = 4096,
        talker_do_sample: bool = True,
        talker_top_k: int = 40,
        talker_top_p: float = 0.8,
        talker_temperature: float = 0.9,
        talker_eos_token_id: list[int] = [8292, 8294],
        talker_repetition_penalty: float = 1.05,
        **kwargs
    ):
        """
        Generate text response and audio from input.

        Args:
            input_ids (`Optional[torch.Tensor]`, *optional*):
                Input ids, should obtain from processor.
            spk (`str` , defaults to "Chelsie"):
                Which speaker should be used in audio response.
            use_audio_in_video (`bool`, defaults to False):
                Whether or not use audio track in video, should same as the parameter in `process_audio_info`.
            return_audio (`Optional[bool]`, *optional*):
                Whether or not return response in audio format. When `return_audio=None`, this parameter is same as `config.enable_audio_output`.
            kwargs (*optional*):
                - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model.
                - With a *thinker_*, *talker_*, *token2wav_* prefix, they will be input for the `generate` method of the
                thinker, talker and token2wav respectively. It has the priority over the keywords without a prefix.
        Returns:
            When `return_audio=False`:
                - **Text** (`torch.Tensor`): Generated text token sequence.
            When `return_audio=True`:
                - **Text** (`torch.Tensor`): Generated text token sequence.
                - **Audio waveform** (`torch.Tensor`): Generated audio waveform.
        """
        if spk not in self.speaker_map:
            raise ValueError(f"{spk} is not availible, availible speakers: {self.speaker_map.keys()}")
        if return_audio and not self.has_talker:
            raise ValueError(
                "Cannot use talker when talker module not initalized. Use `enable_talker` method or set enable_talker in config to enable talker."
            )
        if return_audio is None:
            return_audio = self.has_talker
        if tuple(input_ids.shape)[0] != 1 and return_audio:
            raise NotImplementedError("Qwen2.5-Omni currently does not support batched inference with audio output")
        shared_kwargs = {"use_audio_in_video": use_audio_in_video}
        thinker_kwargs = {"output_hidden_states": True, "return_dict_in_generate": True, "return_dict": True}
        # thinker_max_new_tokens = 32
        # talker_max_new_tokens = 8
        talker_generation_config = GenerationConfig(
            max_new_tokens=talker_max_new_tokens,
            do_sample=talker_do_sample,
            top_k=talker_top_k,
            top_p=talker_top_p,
            temperature=talker_temperature,
            eos_token_id=talker_eos_token_id,
            repetition_penalty=talker_repetition_penalty,
            use_cache=kwargs.get("use_cache", True),
            bos_token_id=self.talker.codec_bos_token,
        )
        talker_kwargs = {}
        token2wav_kwargs = {}
        for key, value in kwargs.items():
            if key.startswith("thinker_"):
                thinker_kwargs[key[len("thinker_") :]] = value
            elif key.startswith("talker_"):
                talker_kwargs[key[len("talker_") :]] = value
            elif key.startswith("token2wav_"):
                token2wav_kwargs[key[len("token2wav_") :]] = value
            elif key == "feature_attention_mask":
                thinker_kwargs[key] = value
                talker_kwargs["audio_feature_lengths"] = paddle.sum(x=value, axis=1)
            elif key == "input_features" or key == "attention_mask":
                thinker_kwargs[key] = value
            else:
                shared_kwargs[key] = value

        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value
            if key not in talker_kwargs:
                talker_kwargs[key] = value
            if key not in token2wav_kwargs:
                token2wav_kwargs[key] = value

        speaker_params = self.speaker_map[spk]
        # diff thinker_kwargs['input_features'].sum()
        # thinker_kwargs['pixel_values_videos'].sum()
        thinker_result = self.thinker.generate(
            input_ids=input_ids,
            **thinker_kwargs,
            generation_config=GenerationConfig(
                eos_token_id=self.thinker.config.eos_token_id,
                max_new_tokens=thinker_max_new_tokens,
                use_cache=kwargs.get("use_cache", True),
            ),
        )
        if not (return_audio and self.has_talker):
            return thinker_result.sequences

        # generate output dict
        thinker_generate_ids = thinker_result.sequences
        thinker_token_embeds = [x[0] for x in thinker_result.hidden_states]
        thinker_hidden_states = [x[1][-1] for x in thinker_result.hidden_states]
        talker_text_bos_token = speaker_params["bos_token"]
        talker_input_text_ids = paddle.concat(
            x=[
                input_ids,
                paddle.to_tensor(data=[[talker_text_bos_token]], dtype="int64"),
                thinker_generate_ids[:, :1],
            ],
            axis=-1,
        )
        talker_input_ids = paddle.concat(
            x=[
                paddle.full_like(x=input_ids, fill_value=self.talker.codec_mask_token),
                paddle.to_tensor(data=[[self.talker.codec_pad_token]], dtype="int64"),
                paddle.to_tensor(data=[[self.talker.codec_bos_token]], dtype="int64"),
            ],
            axis=1,
        )
        thinker_reply_part = paddle.concat(x=thinker_hidden_states[1:], axis=1) + paddle.concat(
            x=thinker_token_embeds[1:], axis=1
        )
        talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
        talker_inputs_embeds = paddle.concat(
            x=[
                talker_inputs_embeds,
                self.thinker.get_input_embeddings()(paddle.to_tensor(data=[[talker_text_bos_token]], dtype="int64")),
                thinker_reply_part[:, :1, :],
            ],
            axis=1,
        )
        thinker_reply_part = paddle.concat(
            x=[
                thinker_reply_part[:, 1:, :],
                self.thinker.get_input_embeddings()(
                    paddle.to_tensor(data=[[self.talker.text_eos_token]], dtype="int64")
                ),
                self.thinker.get_input_embeddings()(
                    paddle.to_tensor(data=[[self.talker.text_pad_token]], dtype="int64")
                ),
            ],
            axis=1,
        )
        talker_attention_mask = paddle.concat(
            x=[kwargs["attention_mask"], paddle.ones(shape=(1, 2), dtype=kwargs["attention_mask"].dtype)], axis=1
        )
        # import pdb;pdb.set_trace()

        talker_result = self.talker.generate(
            input_ids=talker_input_ids,
            input_text_ids=talker_input_text_ids,
            thinker_reply_part=thinker_reply_part,  # sum: 2128
            inputs_embeds=talker_inputs_embeds,  # sum: 254976
            attention_mask=talker_attention_mask,
            **{k: (v if paddle.is_tensor(x=v) else v) for k, v in talker_kwargs.items()},
            generation_config=talker_generation_config,
            logits_processors=LogitsProcessorList(
                [
                    SuppressTokensLogitsProcessor(
                        suppress_tokens=[self.talker.codec_bos_token],
                    )
                ]
            ),
        )
        talker_generate_codes = talker_result[0][:, :-1]
        wav = self.token2wav(
            talker_generate_codes,
            cond=speaker_params["cond"].astype(dtype="float32"),
            ref_mel=speaker_params["ref_mel"].astype(dtype="float32"),
            **token2wav_kwargs,
        )
        return thinker_result.sequences, wav.astype(dtype="float32")

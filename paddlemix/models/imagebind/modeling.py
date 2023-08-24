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

from functools import partial
from types import SimpleNamespace

import paddle
from paddlenlp.transformers.model_utils import register_base_model

from paddlemix.models.model_utils import MixPretrainedModel

from .configuration import ImageBindConfig
from .helpers import (
    LearnableLogitScaling,
    Normalize,
    SelectElement,
    SelectEOSAndProject,
)
from .multimodal_preprocessors import (
    AudioPreprocessor,
    IMUPreprocessor,
    PadIm2Video,
    PatchEmbedGeneric,
    RGBDTPreprocessor,
    SpatioTemporalPosEmbeddingHelper,
    TextPreprocessor,
    ThermalPreprocessor,
)
from .transformer import MultiheadAttention, SimpleTransformer

ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    AUDIO="audio",
    THERMAL="thermal",
    DEPTH="depth",
    IMU="imu",
)

__all__ = [
    "ImageBindModel",
    "ImageBindPretrainedModel",
]


class ImageBindPretrainedModel(MixPretrainedModel):
    """
    See :class:`~paddlenlp.transformers.model_utils.MixPretrainedModel` for more details.
    """

    model_config_file = "config.json"
    config_class = ImageBindConfig
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "ImageBind"


@register_base_model
class ImageBindModel(ImageBindPretrainedModel):
    def __init__(self, config: ImageBindConfig):
        super(ImageBindModel, self).__init__(config)

        # vision_config
        video_frames = config.vision_config.video_frames
        kernel_size = config.vision_config.kernel_size
        out_embed_dim = config.vision_config.out_embed_dim
        vision_embed_dim = config.vision_config.vision_embed_dim
        vision_num_blocks = config.vision_config.vision_num_blocks
        vision_num_heads = config.vision_config.vision_num_heads

        # audio_config
        audio_kernel_size = config.audio_config.audio_kernel_size
        audio_stride = config.audio_config.audio_stride
        audio_embed_dim = config.audio_config.audio_embed_dim
        audio_num_blocks = config.audio_config.audio_num_blocks
        audio_num_heads = config.audio_config.audio_num_heads
        audio_num_mel_bins = config.audio_config.audio_num_mel_bins
        audio_target_len = config.audio_config.audio_target_len
        audio_drop_path = config.audio_config.audio_drop_path

        # text_config
        text_embed_dim = config.text_config.text_embed_dim
        text_num_blocks = config.text_config.text_num_blocks
        text_num_heads = config.text_config.text_num_heads
        # context_length = config.text_config.context_length
        # vocab_size = config.text_config.vocab_size

        # depth_config
        depth_embed_dim = config.depth_config.depth_embed_dim
        depth_kernel_size = config.depth_config.depth_kernel_size
        depth_num_blocks = config.depth_config.depth_num_blocks
        depth_num_heads = config.depth_config.depth_num_heads
        depth_drop_path = config.depth_config.depth_drop_path

        # thermal_config
        thermal_embed_dim = config.thermal_config.thermal_embed_dim
        thermal_kernel_size = config.thermal_config.thermal_kernel_size
        thermal_num_blocks = config.thermal_config.thermal_num_blocks
        thermal_num_heads = config.thermal_config.thermal_num_heads
        thermal_drop_path = config.thermal_config.thermal_drop_path

        # imu_config
        imu_embed_dim = config.imu_config.imu_embed_dim
        # imu_kernel_size = config.imu_config.imu_kernel_size
        imu_num_blocks = config.imu_config.imu_num_blocks
        imu_num_heads = config.imu_config.imu_num_heads
        imu_drop_path = config.imu_config.imu_drop_path

        self.modality_preprocessors = self._create_modality_preprocessors(
            video_frames,
            vision_embed_dim,
            kernel_size,
            text_embed_dim,
            audio_embed_dim,
            audio_kernel_size,
            audio_stride,
            audio_num_mel_bins,
            audio_target_len,
            depth_embed_dim,
            depth_kernel_size,
            thermal_embed_dim,
            thermal_kernel_size,
            imu_embed_dim,
        )
        self.modality_trunks = self._create_modality_trunks(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            audio_drop_path,
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            depth_drop_path,
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            thermal_drop_path,
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            imu_drop_path,
        )
        self.modality_heads = self._create_modality_heads(
            out_embed_dim,
            vision_embed_dim,
            text_embed_dim,
            audio_embed_dim,
            depth_embed_dim,
            thermal_embed_dim,
            imu_embed_dim,
        )
        self.modality_postprocessors = self._create_modality_postprocessors(out_embed_dim)

    def _create_modality_preprocessors(
        self,
        video_frames,
        vision_embed_dim,
        kernel_size,
        text_embed_dim,
        audio_embed_dim,
        audio_kernel_size,
        audio_stride,
        audio_num_mel_bins,
        audio_target_len,
        depth_embed_dim,
        depth_kernel_size,
        thermal_embed_dim,
        thermal_kernel_size,
        imu_embed_dim,
    ):
        rgbt_stem = PatchEmbedGeneric(
            proj_stem=[
                PadIm2Video(pad_type="repeat", ntimes=2),
                paddle.nn.Conv3D(
                    in_channels=3,
                    kernel_size=kernel_size,
                    out_channels=vision_embed_dim,
                    stride=kernel_size,
                    bias_attr=False,
                ),
            ]
        )
        rgbt_preprocessor = RGBDTPreprocessor(
            img_size=[3, video_frames, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=rgbt_stem,
            depth_stem=None,
        )
        text_preprocessor = TextPreprocessor(
            context_length=77,
            vocab_size=49408,
            embed_dim=text_embed_dim,
            causal_masking=True,
        )
        audio_stem = PatchEmbedGeneric(
            proj_stem=[
                paddle.nn.Conv2D(
                    in_channels=1,
                    kernel_size=audio_kernel_size,
                    stride=audio_stride,
                    out_channels=audio_embed_dim,
                    bias_attr=False,
                )
            ],
            norm_layer=paddle.nn.LayerNorm(
                normalized_shape=audio_embed_dim,
                epsilon=1e-05,
                weight_attr=None,
                bias_attr=None,
            ),
        )
        audio_preprocessor = AudioPreprocessor(
            img_size=[1, audio_num_mel_bins, audio_target_len],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            audio_stem=audio_stem,
        )
        depth_stem = PatchEmbedGeneric(
            [
                paddle.nn.Conv2D(
                    kernel_size=depth_kernel_size,
                    in_channels=1,
                    out_channels=depth_embed_dim,
                    stride=depth_kernel_size,
                    bias_attr=False,
                )
            ],
            norm_layer=paddle.nn.LayerNorm(
                normalized_shape=depth_embed_dim,
                epsilon=1e-05,
                weight_attr=None,
                bias_attr=None,
            ),
        )
        depth_preprocessor = RGBDTPreprocessor(
            img_size=[1, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=None,
            depth_stem=depth_stem,
        )
        thermal_stem = PatchEmbedGeneric(
            [
                paddle.nn.Conv2D(
                    kernel_size=thermal_kernel_size,
                    in_channels=1,
                    out_channels=thermal_embed_dim,
                    stride=thermal_kernel_size,
                    bias_attr=False,
                )
            ],
            norm_layer=paddle.nn.LayerNorm(
                normalized_shape=thermal_embed_dim,
                epsilon=1e-05,
                weight_attr=None,
                bias_attr=None,
            ),
        )
        thermal_preprocessor = ThermalPreprocessor(
            img_size=[1, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            thermal_stem=thermal_stem,
        )
        imu_stem = PatchEmbedGeneric(
            [paddle.nn.Linear(in_features=48, out_features=imu_embed_dim, bias_attr=False)],
            norm_layer=paddle.nn.LayerNorm(
                normalized_shape=imu_embed_dim,
                epsilon=1e-05,
                weight_attr=None,
                bias_attr=None,
            ),
        )
        imu_preprocessor = IMUPreprocessor(
            img_size=[6, 2000],
            num_cls_tokens=1,
            kernel_size=8,
            embed_dim=imu_embed_dim,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            imu_stem=imu_stem,
        )
        modality_preprocessors = {
            ModalityType.VISION: rgbt_preprocessor,
            ModalityType.TEXT: text_preprocessor,
            ModalityType.AUDIO: audio_preprocessor,
            ModalityType.DEPTH: depth_preprocessor,
            ModalityType.THERMAL: thermal_preprocessor,
            ModalityType.IMU: imu_preprocessor,
        }
        return paddle.nn.LayerDict(sublayers=modality_preprocessors)

    def _create_modality_trunks(
        self,
        vision_embed_dim,
        vision_num_blocks,
        vision_num_heads,
        text_embed_dim,
        text_num_blocks,
        text_num_heads,
        audio_embed_dim,
        audio_num_blocks,
        audio_num_heads,
        audio_drop_path,
        depth_embed_dim,
        depth_num_blocks,
        depth_num_heads,
        depth_drop_path,
        thermal_embed_dim,
        thermal_num_blocks,
        thermal_num_heads,
        thermal_drop_path,
        imu_embed_dim,
        imu_num_blocks,
        imu_num_heads,
        imu_drop_path,
    ):
        def instantiate_trunk(embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    add_bias_kv=add_bias_kv,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias_attr=True,
                ),
                pre_transformer_layer=paddle.nn.Sequential(
                    paddle.nn.LayerNorm(
                        normalized_shape=embed_dim,
                        epsilon=1e-06,
                        weight_attr=None,
                        bias_attr=None,
                    )
                    if pre_transformer_ln
                    else paddle.nn.Identity(),
                    # EinOpsRearrange('b l d -> l b d')
                ),
                # post_transformer_layer=EinOpsRearrange('l b d -> b l d')
            )

        modality_trunks = {}
        modality_trunks[ModalityType.VISION] = instantiate_trunk(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            pre_transformer_ln=True,
            add_bias_kv=False,
            drop_path=0.0,
        )
        modality_trunks[ModalityType.TEXT] = instantiate_trunk(
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=False,
            drop_path=0.0,
        )
        modality_trunks[ModalityType.AUDIO] = instantiate_trunk(
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=audio_drop_path,
        )
        modality_trunks[ModalityType.DEPTH] = instantiate_trunk(
            depth_embed_dim,
            depth_num_blocks,
            depth_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=depth_drop_path,
        )
        modality_trunks[ModalityType.THERMAL] = instantiate_trunk(
            thermal_embed_dim,
            thermal_num_blocks,
            thermal_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=thermal_drop_path,
        )
        modality_trunks[ModalityType.IMU] = instantiate_trunk(
            imu_embed_dim,
            imu_num_blocks,
            imu_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=imu_drop_path,
        )
        return paddle.nn.LayerDict(sublayers=modality_trunks)

    def _create_modality_heads(
        self,
        out_embed_dim,
        vision_embed_dim,
        text_embed_dim,
        audio_embed_dim,
        depth_embed_dim,
        thermal_embed_dim,
        imu_embed_dim,
    ):
        modality_heads = {}
        modality_heads[ModalityType.VISION] = paddle.nn.Sequential(
            paddle.nn.LayerNorm(
                normalized_shape=vision_embed_dim,
                epsilon=1e-06,
                weight_attr=None,
                bias_attr=None,
            ),
            SelectElement(index=0),
            paddle.nn.Linear(
                in_features=vision_embed_dim,
                out_features=out_embed_dim,
                bias_attr=False,
            ),
        )
        modality_heads[ModalityType.TEXT] = SelectEOSAndProject(
            proj=paddle.nn.Sequential(
                paddle.nn.LayerNorm(
                    normalized_shape=text_embed_dim,
                    epsilon=1e-06,
                    weight_attr=None,
                    bias_attr=None,
                ),
                paddle.nn.Linear(
                    in_features=text_embed_dim,
                    out_features=out_embed_dim,
                    bias_attr=False,
                ),
            )
        )
        modality_heads[ModalityType.AUDIO] = paddle.nn.Sequential(
            paddle.nn.LayerNorm(
                normalized_shape=audio_embed_dim,
                epsilon=1e-06,
                weight_attr=None,
                bias_attr=None,
            ),
            SelectElement(index=0),
            paddle.nn.Linear(in_features=audio_embed_dim, out_features=out_embed_dim, bias_attr=False),
        )
        modality_heads[ModalityType.DEPTH] = paddle.nn.Sequential(
            paddle.nn.LayerNorm(
                normalized_shape=depth_embed_dim,
                epsilon=1e-06,
                weight_attr=None,
                bias_attr=None,
            ),
            SelectElement(index=0),
            paddle.nn.Linear(in_features=depth_embed_dim, out_features=out_embed_dim, bias_attr=False),
        )
        modality_heads[ModalityType.THERMAL] = paddle.nn.Sequential(
            paddle.nn.LayerNorm(
                normalized_shape=thermal_embed_dim,
                epsilon=1e-06,
                weight_attr=None,
                bias_attr=None,
            ),
            SelectElement(index=0),
            paddle.nn.Linear(
                in_features=thermal_embed_dim,
                out_features=out_embed_dim,
                bias_attr=False,
            ),
        )
        modality_heads[ModalityType.IMU] = paddle.nn.Sequential(
            paddle.nn.LayerNorm(
                normalized_shape=imu_embed_dim,
                epsilon=1e-06,
                weight_attr=None,
                bias_attr=None,
            ),
            SelectElement(index=0),
            paddle.nn.Dropout(p=0.5),
            paddle.nn.Linear(in_features=imu_embed_dim, out_features=out_embed_dim, bias_attr=False),
        )
        return paddle.nn.LayerDict(sublayers=modality_heads)

    def _create_modality_postprocessors(self, out_embed_dim):
        modality_postprocessors = {}
        modality_postprocessors[ModalityType.VISION] = Normalize(dim=-1)
        modality_postprocessors[ModalityType.TEXT] = paddle.nn.Sequential(
            Normalize(dim=-1), LearnableLogitScaling(learnable=True)
        )
        modality_postprocessors[ModalityType.AUDIO] = paddle.nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=20.0, learnable=False),
        )
        modality_postprocessors[ModalityType.DEPTH] = paddle.nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )
        modality_postprocessors[ModalityType.THERMAL] = paddle.nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=10.0, learnable=False),
        )
        modality_postprocessors[ModalityType.IMU] = paddle.nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=False),
        )
        return paddle.nn.LayerDict(sublayers=modality_postprocessors)

    def forward(self, inputs):
        outputs = {}
        for modality_key, modality_value in inputs.items():
            reduce_list = modality_value.ndim >= 5
            if reduce_list:
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(B * S, *modality_value.shape[2:])
            if modality_value is not None:
                modality_value = self.modality_preprocessors[modality_key](**{modality_key: modality_value})
                print(
                    f"modal: {modality_key}   paddle_modality_value['trunk']['tokens'].mean(): {modality_value['trunk']['tokens'].mean().item()}"
                )

                trunk_inputs = modality_value["trunk"]
                head_inputs = modality_value["head"]
                modality_value = self.modality_trunks[modality_key](**trunk_inputs)
                modality_value = self.modality_heads[modality_key](modality_value, **head_inputs)
                modality_value = self.modality_postprocessors[modality_key](modality_value)
                if modality_key == "audio" or modality_key == "depth" or modality_key == "thermal":
                    modality_value = self.modality_postprocessors[modality_key](modality_value)
                if reduce_list:
                    modality_value = modality_value.reshape(B, S, -1)
                    modality_value = modality_value.mean(axis=1)
                outputs[modality_key] = modality_value
        return outputs

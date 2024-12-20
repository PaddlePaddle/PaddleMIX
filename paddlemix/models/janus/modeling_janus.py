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

import paddle
from einops import rearrange
from paddlenlp.transformers import LlamaForCausalLM, PretrainedModel

from .clip_encoder import CLIPVisionTower
from .configuration_janus import JanusFlowMultiModalityConfig, MultiModalityConfig
from .projector import MlpProjector
from .uvit import LlamaRMSNorm, ShallowUViTDecoder, ShallowUViTEncoder
from .vq_model import ModelArgs, VQModel

__all__ = [
    "JanusMultiModalityPreTrainedModel",
    "JanusMultiModalityCausalLM",
    "JanusFlowMultiModalityPreTrainedModel",
    "JanusFlowMultiModalityCausalLM",
]


class vision_head(paddle.nn.Layer):
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = paddle.nn.Linear(
            in_features=params["n_embed"], out_features=params["image_token_embed"]
        )
        self.vision_activation = paddle.nn.GELU()
        self.vision_head = paddle.nn.Linear(
            in_features=params["image_token_embed"], out_features=params["image_token_size"]
        )

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x


def model_name_to_cls(cls_name):
    if "MlpProjector" in cls_name:
        cls = MlpProjector
    elif "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower
    elif "VQ" in cls_name:

        def VQ_16(**kwargs):
            return VQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

        VQ_models = {"VQ-16": VQ_16}
        cls = VQ_models[cls_name]
    elif "vision_head" in cls_name:
        cls = vision_head
    elif "ShallowUViTEncoder" in cls_name:
        cls = ShallowUViTEncoder
    elif "ShallowUViTDecoder" in cls_name:
        cls = ShallowUViTDecoder
    else:
        raise ValueError(f"class_name {cls_name} is invalid.")
    return cls


class JanusMultiModalityPreTrainedModel(PretrainedModel):
    config_class = MultiModalityConfig
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class JanusMultiModalityCausalLM(JanusMultiModalityPreTrainedModel):
    config_class = MultiModalityConfig

    def __init__(self, config: MultiModalityConfig):
        super().__init__(config)
        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)
        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)
        gen_vision_config = config.gen_vision_config
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
        self.gen_vision_model = gen_vision_cls()
        gen_aligner_config = config.gen_aligner_config
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)
        gen_head_config = config.gen_head_config
        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)
        self.gen_embed = paddle.nn.Embedding(
            num_embeddings=gen_vision_config.params["image_token_size"],
            embedding_dim=gen_vision_config.params["n_embed"],
        )
        language_config = config.language_config
        self.language_model = LlamaForCausalLM(language_config)

    def prepare_inputs_embeds(
        self,
        input_ids: paddle.Tensor,
        pixel_values: paddle.Tensor,
        images_seq_mask: paddle.Tensor,
        images_emb_mask: paddle.Tensor,
        **kwargs
    ):
        """

        Args:
            input_ids (paddle.Tensor): [b, T]
            pixel_values (paddle.Tensor):   [b, n_images, 3, h, w]
            images_seq_mask (paddle.Tensor): [b, T]
            images_emb_mask (paddle.Tensor): [b, n_images, n_image_tokens]

            assert paddle.sum(images_seq_mask) == paddle.sum(images_emb_mask)

        Returns:
            input_embeds (paddle.Tensor): [b, T, D]
        """
        bs, n = tuple(pixel_values.shape)[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        images_embeds = self.aligner(self.vision_model(images))
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")
        input_ids[input_ids < 0] = 0
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds

    def prepare_gen_img_embeds(self, image_ids: paddle.Tensor):
        return self.gen_aligner(self.gen_embed(image_ids))


# Janus Flow
class JanusFlowMultiModalityPreTrainedModel(PretrainedModel):
    config_class = JanusFlowMultiModalityConfig
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class JanusFlowMultiModalityCausalLM(JanusFlowMultiModalityPreTrainedModel):
    config_class = JanusFlowMultiModalityConfig

    def __init__(self, config: JanusFlowMultiModalityConfig):
        super().__init__(config)
        vision_und_enc_config = config.vision_und_enc_config
        vision_und_enc_cls = model_name_to_cls(vision_und_enc_config.cls)
        self.vision_und_enc_model = vision_und_enc_cls(**vision_und_enc_config.params)
        self.vision_und_enc_aligner = paddle.nn.Linear(in_features=1024, out_features=2048, bias_attr=True)
        self.beg_of_und_embed = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.zeros(shape=[1, 2048]))
        vision_gen_enc_config = config.vision_gen_enc_config
        vision_gen_enc_cls = model_name_to_cls(vision_gen_enc_config.cls)
        self.vision_gen_enc_model = vision_gen_enc_cls(**vision_gen_enc_config.params)
        self.vision_gen_enc_aligner = paddle.nn.Linear(in_features=768, out_features=2048, bias_attr=True)
        vision_gen_dec_config = config.vision_gen_dec_config
        vision_gen_dec_cls = model_name_to_cls(vision_gen_dec_config.cls)
        self.vision_gen_dec_model = vision_gen_dec_cls(**vision_gen_dec_config.params)
        language_config = config.language_config
        self.language_model = LlamaForCausalLM(language_config)
        self.vision_gen_dec_aligner_norm = LlamaRMSNorm(2048, eps=language_config.rms_norm_eps)
        self.vision_gen_dec_aligner = paddle.nn.Linear(in_features=2048, out_features=768, bias_attr=True)

    def prepare_inputs_embeds(
        self,
        input_ids: paddle.Tensor,
        pixel_values: paddle.Tensor,
        images_seq_mask: paddle.Tensor,
        images_emb_mask: paddle.Tensor,
        **kwargs
    ):
        """

        Args:
            input_ids (paddle.Tensor): [b, T]
            pixel_values (paddle.Tensor):   [b, n_images, 3, h, w]
            images_seq_mask (paddle.Tensor): [b, T]
            images_emb_mask (paddle.Tensor): [b, n_images, n_image_tokens]

            assert paddle.sum(images_seq_mask) == paddle.sum(images_emb_mask)

        Returns:
            input_embeds (paddle.Tensor): [b, T, D]
        """
        bs, n = tuple(pixel_values.shape)[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        images_embeds = self.vision_und_enc_model(images)
        images_embeds = self.vision_und_enc_aligner(images_embeds)
        beg_of_und_embed = self.beg_of_und_embed[0].detach().clone()
        images_embeds = paddle.concat(
            x=[
                beg_of_und_embed.view(1, 1, -1).tile(repeat_times=[tuple(images_embeds.shape)[0], 1, 1]),
                images_embeds,
            ],
            axis=1,
        )
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")
        input_ids[input_ids < 0] = 0
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]
        return inputs_embeds

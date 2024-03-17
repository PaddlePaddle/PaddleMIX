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

from ppdiffusers.transformers.clip import (
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

VISION_CONFIG_DICT = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768,
    "num_positions": 512,
}


class MLP(paddle.nn.Layer):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = paddle.nn.LayerNorm(normalized_shape=in_dim)
        self.fc1 = paddle.nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.fc2 = paddle.nn.Linear(in_features=hidden_dim, out_features=out_dim)
        self.use_residual = use_residual
        self.act_fn = paddle.nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class FuseModule(paddle.nn.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = paddle.nn.LayerNorm(normalized_shape=embed_dim)

    def fuse_fn(self, prompt_embeds, id_embeds):
        stacked_id_embeds = paddle.concat(x=[prompt_embeds, id_embeds], axis=-1)
        stacked_id_embeds = self.mlp1(stacked_id_embeds) + prompt_embeds
        stacked_id_embeds = self.mlp2(stacked_id_embeds)
        stacked_id_embeds = self.layer_norm(stacked_id_embeds)
        return stacked_id_embeds

    def forward(self, prompt_embeds, id_embeds, class_tokens_mask) -> paddle.Tensor:
        num_inputs = class_tokens_mask.sum().unsqueeze(axis=0)
        batch_size, max_num_inputs = id_embeds.shape[:2]
        seq_length = prompt_embeds.shape[1]

        flat_id_embeds = id_embeds.reshape([-1, id_embeds.shape[-2], id_embeds.shape[-1]])
        valid_id_mask = paddle.arange(end=max_num_inputs)[None, :] < num_inputs[:, None]
        valid_id_embeds = flat_id_embeds[valid_id_mask.flatten()]
        prompt_embeds = prompt_embeds.reshape([-1, prompt_embeds.shape[-1]])
        class_tokens_mask = class_tokens_mask.reshape([-1])
        valid_id_embeds = valid_id_embeds.reshape([-1, valid_id_embeds.shape[-1]])

        image_token_embeds = prompt_embeds[class_tokens_mask]
        stacked_id_embeds = self.fuse_fn(image_token_embeds, valid_id_embeds)
        assert (
            class_tokens_mask.sum() == stacked_id_embeds.shape[0]
        ), f"{class_tokens_mask.sum()} != {stacked_id_embeds.shape[0]}"

        paddle_version = float(paddle.__version__[:3])
        if (paddle_version == 0.0) or (paddle_version > 2.6):
            prompt_embeds.masked_scatter_(class_tokens_mask[:, None], stacked_id_embeds)
        else:
            zeros_like_x = paddle.zeros_like(prompt_embeds, dtype=int)
            mask = paddle.add(paddle.cast(class_tokens_mask[:, None], dtype="int"), zeros_like_x)
            mask_prefix = paddle.clip(mask.cumsum() - 1, min=0)
            value = stacked_id_embeds.flatten()[mask_prefix].reshape(mask.shape)
            mask = paddle.logical_not(mask)
            prompt_embeds = paddle.where(mask, prompt_embeds, value)

        updated_prompt_embeds = prompt_embeds.reshape([batch_size, seq_length, -1])

        return updated_prompt_embeds


class PhotoMakerIDEncoder(CLIPVisionModelWithProjection):
    def __init__(self):
        super().__init__(CLIPVisionConfig(**VISION_CONFIG_DICT))
        self.visual_projection_2 = paddle.nn.Linear(in_features=1024, out_features=1280, bias_attr=False)
        self.fuse_module = FuseModule(2048)

    def forward(self, id_pixel_values, prompt_embeds, class_tokens_mask):
        b, num_inputs, c, h, w = id_pixel_values.shape
        id_pixel_values = id_pixel_values.reshape([b * num_inputs, c, h, w])

        shared_id_embeds = self.vision_model(id_pixel_values)[1]
        id_embeds = self.visual_projection(shared_id_embeds)
        id_embeds_2 = self.visual_projection_2(shared_id_embeds)
        id_embeds = id_embeds.reshape([b, num_inputs, 1, -1])
        id_embeds_2 = id_embeds_2.reshape([b, num_inputs, 1, -1])

        id_embeds = paddle.concat(x=(id_embeds, id_embeds_2), axis=-1)
        updated_prompt_embeds = self.fuse_module(prompt_embeds, id_embeds, class_tokens_mask)
        return updated_prompt_embeds


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(axis=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(axis=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


if __name__ == "__main__":
    PhotoMakerIDEncoder()

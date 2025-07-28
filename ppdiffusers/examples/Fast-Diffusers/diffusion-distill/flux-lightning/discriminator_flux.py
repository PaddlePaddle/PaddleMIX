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

from typing import Any, Dict, Optional, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute

from ppdiffusers.models.transformer_2d import Transformer2DModelOutput
from ppdiffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def modified_forward(
    self,
    hidden_states: paddle.Tensor,
    encoder_hidden_states: paddle.Tensor = None,
    pooled_projections: paddle.Tensor = None,
    timestep: paddle.Tensor = None,
    guidance: Optional[paddle.Tensor] = None,
    txt_ids: Optional[paddle.Tensor] = None,
    img_ids: Optional[paddle.Tensor] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    half_num_heads: bool = False,
    return_dict: bool = True,
) -> Union[paddle.Tensor, Transformer2DModelOutput]:
    """
    The [`FluxTransformer2DModel`] forward method.
    Args:
        hidden_states (`paddle.Tensor` of shape `(batch size, channel, height, width)`):
            Input `hidden_states`.
        encoder_hidden_states (`paddle.Tensor` of shape `(batch size, sequence_len, embed_dims)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`paddle.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
            from the embeddings of input conditions.
        timestep ( `paddle.LongTensor`):
            Used to indicate denoising step.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.
    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d paddle.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d paddle Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d paddle.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d paddle Tensor"
        )
        img_ids = img_ids[0]

    ids = paddle.concat((txt_ids, img_ids), axis=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    output_features = []
    for index_block, block in enumerate(self.transformer_blocks):
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    self.disable_adapters()
                    if return_dict is not None:
                        output = module(*inputs, return_dict=return_dict)
                    else:
                        output = module(*inputs)
                    self.enable_adapters()
                    return output

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False, "preserve_rng_state": True}
            encoder_hidden_states, hidden_states = recompute(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                **ckpt_kwargs,
            )
            output_features.append(hidden_states)
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            output_features.append(hidden_states)
    hidden_states = paddle.concat([encoder_hidden_states, hidden_states], axis=1)

    for index_block, block in enumerate(self.single_transformer_blocks):
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    self.disable_adapters()
                    if return_dict is not None:
                        output = module(*inputs, return_dict=return_dict)
                    else:
                        output = module(*inputs)
                    self.enable_adapters()
                    return output

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False, "preserve_rng_state": True}
            hidden_states = recompute(
                create_custom_forward(block),
                hidden_states,
                temb,
                image_rotary_emb,
                **ckpt_kwargs,
            )
            output_features.append(hidden_states[:, encoder_hidden_states.shape[1] :, ...])
        else:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            output_features.append(hidden_states[:, encoder_hidden_states.shape[1] :, ...])

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if half_num_heads:
        return output_features[::2]
    else:
        return output_features


class DiscriminatorHead(nn.Layer):
    def __init__(self, resolution, input_channel, output_channel=1):
        super().__init__()
        self.resolution = resolution
        self.conv1 = nn.Sequential(
            nn.Conv2D(input_channel, input_channel, 1, 1, 0),
            nn.GroupNorm(32, input_channel),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2D(input_channel, input_channel, 1, 1, 0),
            nn.GroupNorm(32, input_channel),
            nn.LeakyReLU(),
        )

        self.conv_out = nn.Conv2D(input_channel, output_channel, 1, 1, 0)

    def forward(self, x):
        b, wh, c = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape([b, c, int(self.resolution / 16), int(self.resolution / 16)])
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv_out(x)
        return x


class TransformerBasedDiscriminatorHead(nn.Layer):
    def __init__(self, input_dim=64, output_channel=1):
        super().__init__()
        self.proj_out = nn.Linear(3072, input_dim, bias_attr=True)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.LayerNorm(input_dim), nn.GELU(), nn.Linear(input_dim, output_channel)
        )
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        x_proj = self.proj_out(x)
        x_reduced = x_proj.mean(axis=-2)
        residual = x_reduced
        x_out = self.mlp(x_reduced)
        x_out = self.layer_norm(x_out + residual)
        return x_out


class Discriminator(nn.Layer):
    def __init__(
        self,
        transformer,
        resolution=1024,
        half_num_heads=False,
        num_h_per_head=1,
        adapter_channel_dims=[3072] * 57,
    ):
        super().__init__()
        self.transformer = transformer
        self.half_num_heads = half_num_heads
        self.num_h_per_head = num_h_per_head
        self.head_num = len(adapter_channel_dims)
        self.heads = nn.LayerList(
            [
                nn.LayerList(
                    [
                        DiscriminatorHead(resolution, adapter_channel)
                        # TransformerBasedDiscriminatorHead()
                        for _ in range(self.num_h_per_head)
                    ]
                )
                for adapter_channel in adapter_channel_dims
            ]
        )

    def _forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        pooled_encoder_hidden_states,
        guidance,
        text_ids,
        latent_image_ids,
    ):
        self.transformer.disable_adapters()
        features = modified_forward(
            self.transformer,
            hidden_states=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_encoder_hidden_states,
            guidance=guidance,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            half_num_heads=self.half_num_heads,
        )
        self.transformer.enable_adapters()

        assert self.head_num == len(features)

        outputs = []
        for feature, head in zip(features, self.heads):
            for h in head:
                outputs.append(h(feature))
        return outputs

    def forward(self, flag, *args):
        if flag == "d_loss":
            return self.d_loss(*args)
        elif flag == "g_loss":
            return self.g_loss(*args)
        else:
            assert 0, "not supported"

    def d_loss(
        self,
        sample_fake,
        sample_real,
        timestep,
        encoder_hidden_states,
        pooled_encoder_hidden_states,
        guidance,
        text_ids,
        latent_image_ids,
        weight,
        adv_loss_type,
    ):
        loss = 0.0
        fake_outputs = self._forward(
            sample_fake.detach(),
            timestep,
            encoder_hidden_states,
            pooled_encoder_hidden_states,
            guidance,
            text_ids,
            latent_image_ids,
        )
        real_outputs = self._forward(
            sample_real.detach(),
            timestep,
            encoder_hidden_states,
            pooled_encoder_hidden_states,
            guidance,
            text_ids,
            latent_image_ids,
        )
        for fake_output, real_output in zip(fake_outputs, real_outputs):
            if adv_loss_type == "pcm":
                loss += (
                    paddle.mean(weight * F.relu(fake_output.astype(dtype="float32") + 1))
                    + paddle.mean(weight * F.relu(1 - real_output.astype(dtype="float32")))
                ) / (self.head_num * self.num_h_per_head)
            elif adv_loss_type == "lsgan":
                valid = paddle.ones([fake_output.shape[0], 1])
                fake = paddle.zeros([fake_output.shape[0], 1])
                loss += (
                    weight * F.mse_loss(F.sigmoid(real_output.astype(dtype="float32")), valid)
                    + weight * F.mse_loss(F.sigmoid(fake_output.astype(dtype="float32")), fake)
                ) / (self.head_num * self.num_h_per_head)
            elif adv_loss_type == "wgan":
                loss += (
                    weight * -real_output.astype(dtype="float32").mean()
                    + weight * fake_output.astype(dtype="float32").mean()
                ) / (self.head_num * self.num_h_per_head)
            else:
                raise NotImplementedError
        return loss

    def g_loss(
        self,
        sample_fake,
        timestep,
        encoder_hidden_states,
        pooled_encoder_hidden_states,
        guidance,
        text_ids,
        latent_image_ids,
        weight,
        adv_loss_type,
    ):
        loss = 0.0
        fake_outputs = self._forward(
            sample_fake,
            timestep,
            encoder_hidden_states,
            pooled_encoder_hidden_states,
            guidance,
            text_ids,
            latent_image_ids,
        )
        for fake_output in fake_outputs:
            if adv_loss_type == "pcm":
                loss += paddle.mean(weight * F.relu(1 - fake_output.astype(dtype="float32"))) / (
                    self.head_num * self.num_h_per_head
                )
            elif adv_loss_type == "lsgan":
                valid = paddle.ones([fake_output.shape[0], 1])
                loss += (
                    weight
                    * F.mse_loss(F.sigmoid(fake_output.astype(dtype="float32")), valid)
                    / (self.head_num * self.num_h_per_head)
                )
            elif adv_loss_type == "wgan":
                loss += weight * -fake_output.astype(dtype="float32").mean() / (self.head_num * self.num_h_per_head)
        return loss

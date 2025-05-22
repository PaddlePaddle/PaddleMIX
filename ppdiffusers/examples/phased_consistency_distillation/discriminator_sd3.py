from typing import Any, Dict, Optional, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdiffusers.configuration_utils import ConfigMixin, register_to_config
from ppdiffusers.models.attention import JointTransformerBlock
from ppdiffusers.models.attention_processor import Attention, AttentionProcessor
from ppdiffusers.models.modeling_utils import ModelMixin
from ppdiffusers.models.normalization import AdaLayerNormContinuous
from ppdiffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from ppdiffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from ppdiffusers.models.transformer_2d import Transformer2DModelOutput
from ppdiffusers import SD3Transformer2DModel

def modified_forward(
    self,
    hidden_states: paddle.Tensor,
    encoder_hidden_states: paddle.Tensor = None,
    pooled_projections: paddle.Tensor = None,
    timestep: paddle.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[paddle.Tensor, Transformer2DModelOutput]:
    """
    The [`SD3Transformer2DModel`] forward method.
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
        if (
            joint_attention_kwargs is not None
            and joint_attention_kwargs.get("scale", None) is not None
        ):
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )
    height, width = hidden_states.shape[-2:]
    hidden_states = self.pos_embed(
        hidden_states
    )  # takes care of adding positional embeddings too.
    temb = self.time_text_embed(timestep, pooled_projections)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)
    output_features = []
    for block in self.transformer_blocks:
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
            hidden_states = paddle.distributed.fleet.utils.recompute(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                temb,
                **ckpt_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )
        output_features.append(hidden_states)
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)
    # unpatchify
    patch_size = self.config.patch_size
    height = height // patch_size
    width = width // patch_size
    hidden_states = hidden_states.reshape(
        shape=(
            hidden_states.shape[0],
            height,
            width,
            patch_size,
            patch_size,
            self.out_channels,
        )
    )
    hidden_states = paddle.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        shape=(
            hidden_states.shape[0],
            self.out_channels,
            height * patch_size,
            width * patch_size,
        )
    )
    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)
    return output_features


class DiscriminatorHead(nn.Layer):
    def __init__(self, input_channel, output_channel=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2D(input_channel, input_channel, 1, 1, 0),
            nn.GroupNorm(32, input_channel),
            nn.LeakyReLU(),  # use LeakyReLu instead of GELU shown in the paper to save memory
        )
        self.conv2 = nn.Sequential(
            nn.Conv2D(input_channel, input_channel, 1, 1, 0),
            nn.GroupNorm(32, input_channel),
            nn.LeakyReLU(),  # use LeakyReLu instead of GELU shown in the paper to save memory
        )

        self.conv_out = nn.Conv2D(input_channel, output_channel, 1, 1, 0)

    def forward(self, x):
        b, wh, c = x.shape
        x = x.permute(0, 2, 1)
        x = x.view([b, c, 64, 64])
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv_out(x)
        return x


class Discriminator(nn.Layer):

    def __init__(
        self,
        unet,
        num_h_per_head=1,
        adapter_channel_dims=[1536] * 24,
    ):
        super().__init__()
        self.unet = unet
        self.num_h_per_head = num_h_per_head
        self.head_num = len(adapter_channel_dims)
        self.heads = nn.LayerList(
            [
                nn.LayerList(
                    [
                        DiscriminatorHead(adapter_channel)
                        for _ in range(self.num_h_per_head)
                    ]
                )
                for adapter_channel in adapter_channel_dims
            ]
        )

    def _forward(
        self, sample, timestep, encoder_hidden_states, pooled_encoder_hidden_states
    ):
        features = modified_forward(
            self.unet,
            hidden_states=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_encoder_hidden_states,
        )
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
        weight,
    ):
        loss = 0.0
        fake_outputs = self._forward(
            sample_fake.detach(),
            timestep,
            encoder_hidden_states,
            pooled_encoder_hidden_states,
        )
        real_outputs = self._forward(
            sample_real.detach(),
            timestep,
            encoder_hidden_states,
            pooled_encoder_hidden_states,
        )
        for fake_output, real_output in zip(fake_outputs, real_outputs):
            loss += (
                paddle.mean(weight * F.relu(fake_output.astype(dtype="float32") + 1))
                + paddle.mean(weight * F.relu(1 - real_output.astype(dtype="float32")))
            ) / (self.head_num * self.num_h_per_head)
        return loss

    def g_loss(
        self,
        sample_fake,
        timestep,
        encoder_hidden_states,
        pooled_encoder_hidden_states,
        weight,
    ):
        loss = 0.0
        fake_outputs = self._forward(
            sample_fake, timestep, encoder_hidden_states, pooled_encoder_hidden_states
        )
        for fake_output in fake_outputs:
            loss += paddle.mean(weight * F.relu(1 - fake_output.astype(dtype="float32"))) / (
                self.head_num * self.num_h_per_head
            )
        return loss

    def feature_loss(
        self, sample_fake, sample_real, timestep, encoder_hidden_states, weight
    ):
        loss = 0.0
        features_fake = modified_forward(
            self.unet, sample_fake, timestep, encoder_hidden_states
        )
        features_real = modified_forward(
            self.unet, sample_real.detach(), timestep, encoder_hidden_states
        )
        for feature_fake, feature_real in zip(features_fake, features_real):
            loss += paddle.mean((feature_fake - feature_real) ** 2) / (self.head_num)
        return loss


if __name__ == "__main__":
    teacher_unet = SD3Transformer2DModel.from_pretrained(
        "models/sd3",
        subfolder="transformer",
    )
    teacher_unet.cuda()
    discriminator = Discriminator(teacher_unet).cuda()

    sample = paddle.randn((1, 16, 128, 128)).cuda()
    encoder_hidden_states = paddle.randn((1, 154, 4096)).cuda()
    pooled_encoder_hidden_states = paddle.randn((1, 2048)).cuda()
    timesteps = paddle.randn((1,)).cuda()

    features = discriminator._forward(
        sample, timesteps, encoder_hidden_states, pooled_encoder_hidden_states
    )
    for feature in features:
        print(feature.shape)

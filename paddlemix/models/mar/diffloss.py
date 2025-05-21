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

import paddle

from .diffusion import create_diffusion


def split(x, num_or_sections, axis=0):
    if isinstance(num_or_sections, int):
        return paddle.split(x, x.shape[axis] // num_or_sections, axis)
    else:
        return paddle.split(x, num_or_sections, axis)


def view(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list, str)):
            return paddle.view(self, args[0])
        else:
            return paddle.view(self, list(args))
    elif kwargs:
        return paddle.view(self, shape_or_dtype=list(kwargs.values())[0])


class DiffLoss(paddle.nn.Layer):
    """Diffusion Loss"""

    def __init__(
        self,
        target_channels,
        z_channels,
        depth,
        width,
        num_sampling_steps,
        grad_checkpointing=False,
    ):
        super(DiffLoss, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2,
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
        )
        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")

    def forward(self, target, z, mask=None):
        t = paddle.randint(
            low=0,
            high=self.train_diffusion.num_timesteps,
            shape=(tuple(target.shape)[0],),
        )
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.0):
        if not cfg == 1.0:
            noise = paddle.randn(shape=[tuple(z.shape)[0] // 2, self.in_channels]).cuda(blocking=True)
            noise = paddle.concat(x=[noise, noise], axis=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = paddle.randn(shape=[tuple(z.shape)[0], self.in_channels]).cuda(blocking=True)
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward
        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn,
            tuple(noise.shape),
            noise,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            temperature=temperature,
        )
        return sampled_token_latent


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(paddle.nn.Layer):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = paddle.nn.Sequential(
            paddle.nn.Linear(
                in_features=frequency_embedding_size,
                out_features=hidden_size,
                bias_attr=True,
            ),
            paddle.nn.Silu(),
            paddle.nn.Linear(in_features=hidden_size, out_features=hidden_size, bias_attr=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = paddle.exp(x=-math.log(max_period) * paddle.arange(start=0, end=half, dtype="float32") / half).to(
            device=t.place
        )
        args = t[:, None].astype(dtype="float32") * freqs[None]
        embedding = paddle.concat(x=[paddle.cos(x=args), paddle.sin(x=args)], axis=-1)
        if dim % 2:
            embedding = paddle.concat(x=[embedding, paddle.zeros_like(x=embedding[:, :1])], axis=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(paddle.nn.Layer):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.in_ln = paddle.nn.LayerNorm(normalized_shape=channels, epsilon=1e-06)
        self.mlp = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=channels, out_features=channels, bias_attr=True),
            paddle.nn.Silu(),
            paddle.nn.Linear(in_features=channels, out_features=channels, bias_attr=True),
        )
        self.adaLN_modulation = paddle.nn.Sequential(
            paddle.nn.Silu(),
            paddle.nn.Linear(in_features=channels, out_features=3 * channels, bias_attr=True),
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(chunks=3, axis=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(paddle.nn.Layer):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = paddle.nn.LayerNorm(
            normalized_shape=model_channels,
            weight_attr=False,
            bias_attr=False,
            epsilon=1e-06,
        )
        self.linear = paddle.nn.Linear(in_features=model_channels, out_features=out_channels, bias_attr=True)
        self.adaLN_modulation = paddle.nn.Sequential(
            paddle.nn.Silu(),
            paddle.nn.Linear(
                in_features=model_channels,
                out_features=2 * model_channels,
                bias_attr=True,
            ),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(chunks=2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(paddle.nn.Layer):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = paddle.nn.Linear(in_features=z_channels, out_features=model_channels)
        self.input_proj = paddle.nn.Linear(in_features=in_channels, out_features=model_channels)
        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels))
        self.res_blocks = paddle.nn.LayerList(sublayers=res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, paddle.nn.Linear):
                init_XavierUniform = paddle.nn.initializer.XavierUniform()
                init_XavierUniform(module.weight)
                if module.bias is not None:
                    init_Constant = paddle.nn.initializer.Constant(value=0)
                    init_Constant(module.bias)

        self.apply(_basic_init)
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.time_embed.mlp[0].weight)
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.time_embed.mlp[2].weight)
        for block in self.res_blocks:
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(block.adaLN_modulation[-1].weight)
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(block.adaLN_modulation[-1].bias)
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.final_layer.adaLN_modulation[-1].weight)
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.final_layer.adaLN_modulation[-1].bias)
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.final_layer.linear.weight)
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.final_layer.linear.bias)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)
        y = t + c
        if self.grad_checkpointing and not paddle.in_dynamic_mode():
            for block in self.res_blocks:
                x = paddle.distributed.fleet.utils.recompute(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)
        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = paddle.concat(x=[half, half], axis=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = split(x=eps, num_or_sections=len(eps) // 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = paddle.concat(x=[half_eps, half_eps], axis=0)
        return paddle.concat(x=[eps, rest], axis=1)

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
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
import scipy.stats as stats
from tqdm import tqdm

from .APIs.self_use_atten_block import Block
from .APIs.self_use_scatter import scatter
from .diffloss import DiffLoss


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = paddle.zeros(shape=[bsz, seq_len]).cuda(blocking=True)
    masking = paddle.put_along_axis(
        arr=masking,
        axis=-1,
        indices=order[:, : mask_len.astype(dtype="int64")],
        values=paddle.ones(shape=[bsz, seq_len]).cuda(blocking=True),
        broadcast=False,
    ).astype(dtype="bool")
    return masking


class MAR(nn.Layer):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=256,
        vae_stride=16,
        patch_size=1,
        encoder_embed_dim=1024,
        encoder_depth=16,
        encoder_num_heads=16,
        decoder_embed_dim=1024,
        decoder_depth=16,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=paddle.nn.LayerNorm,
        vae_embed_dim=16,
        mask_ratio_min=0.7,
        label_drop_prob=0.1,
        class_num=1000,
        attn_dropout=0.1,
        proj_dropout=0.1,
        buffer_size=64,
        diffloss_d=3,
        diffloss_w=1024,
        num_sampling_steps="100",
        diffusion_batch_mul=4,
        grad_checkpointing=False,
    ):
        super().__init__()
        self.vae_embed_dim = vae_embed_dim
        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.grad_checkpointing = grad_checkpointing
        self.num_classes = class_num
        self.class_emb = paddle.nn.Embedding(num_embeddings=1000, embedding_dim=encoder_embed_dim)
        self.label_drop_prob = label_drop_prob

        self.fake_latent = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=[1, encoder_embed_dim])
        )
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        self.z_proj = paddle.nn.Linear(
            in_features=self.token_embed_dim,
            out_features=encoder_embed_dim,
            bias_attr=True,
        )

        self.z_proj_ln = paddle.nn.LayerNorm(normalized_shape=encoder_embed_dim, epsilon=1e-06)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=[1, self.seq_len + self.buffer_size, encoder_embed_dim])
        )
        self.encoder_blocks = paddle.nn.LayerList(
            sublayers=[
                Block(
                    encoder_embed_dim,
                    encoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_dropout,
                    attn_drop=attn_dropout,
                )
                for _ in range(encoder_depth)
            ]
        )
        self.encoder_norm = norm_layer(encoder_embed_dim)
        self.decoder_embed = paddle.nn.Linear(
            in_features=encoder_embed_dim,
            out_features=decoder_embed_dim,
            bias_attr=True,
        )
        self.mask_token = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=[1, 1, decoder_embed_dim])
        )
        self.decoder_pos_embed_learned = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=[1, self.seq_len + self.buffer_size, decoder_embed_dim])
        )
        self.decoder_blocks = paddle.nn.LayerList(
            sublayers=[
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_dropout,
                    attn_drop=attn_dropout,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=[1, self.seq_len, decoder_embed_dim])
        )
        self.initialize_weights()
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing,
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.class_emb.weight)
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.fake_latent)
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.mask_token)
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.encoder_pos_embed_learned)
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.decoder_pos_embed_learned)
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.diffusion_pos_embed_learned)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            init_XavierUniform = paddle.nn.initializer.XavierUniform()
            init_XavierUniform(m.weight)
            if isinstance(m, paddle.nn.Linear) and m.bias is not None:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)
        elif isinstance(m, paddle.nn.LayerNorm):
            if m.bias is not None:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)
            if m.weight is not None:
                init_Constant = paddle.nn.initializer.Constant(value=1.0)
                init_Constant(m.weight)

    def patchify(self, x):
        bsz, c, h, w = tuple(x.shape)
        p = self.patch_size
        h_, w_ = h // p, w // p
        x = x.reshape(bsz, c, h_, p, w_, p)
        x = paddle.einsum("nchpwq->nhwcpq", x)
        x = x.reshape(bsz, h_ * w_, c * p**2)
        return x

    def unpatchify(self, x):
        bsz = tuple(x.shape)[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w
        x = x.reshape((bsz, h_, w_, c, p, p))
        x = paddle.einsum("nhwcpq->nchpwq", x)
        x = x.reshape((bsz, c, h_ * p, w_ * p))
        return x

    def sample_orders(self, bsz):
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = paddle.to_tensor(data=np.array(orders), dtype="float32").cuda(blocking=True).astype(dtype="int64")
        return orders

    def random_masking(self, x, orders):
        print(orders.shape)
        bsz, seq_len, embed_dim = tuple(x.shape)
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(paddle.ceil(paddle.to_tensor(seq_len * mask_rate)))
        mask = paddle.zeros(shape=[bsz, seq_len])
        mask = scatter(
            mask,
            dim=-1,
            index=orders[:, :num_masked_tokens],
            src=paddle.ones(shape=[bsz, seq_len]),
        )
        return mask

    def forward_mae_encoder(self, x, mask, class_embedding):

        x = self.z_proj(x)
        bsz, seq_len, embed_dim = tuple(x.shape)
        x = paddle.concat(x=[paddle.zeros(shape=[bsz, self.buffer_size, embed_dim]), x], axis=1)
        if mask.dtype == paddle.bool:
            mask_with_buffer = paddle.concat(
                x=[paddle.zeros(shape=[x.shape[0], self.buffer_size], dtype="bool"), mask], axis=1
            )
        else:
            mask_with_buffer = paddle.concat(x=[paddle.zeros(shape=[x.shape[0], self.buffer_size]), mask], axis=1)

        if self.training:
            drop_latent_mask = paddle.rand(shape=bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(axis=-1).cuda(blocking=True).to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding
        x[:, : self.buffer_size] = class_embedding.unsqueeze(axis=1)
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)
        x = x[(1 - mask_with_buffer).nonzero(as_tuple=True)].reshape((bsz, -1, embed_dim))
        if self.grad_checkpointing and not paddle.in_dynamic_mode():
            for block in self.encoder_blocks:
                x = paddle.distributed.fleet.utils.recompute(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)
        return x

    def forward_mae_decoder(self, x, mask):

        x = self.decoder_embed(x)
        if mask.dtype == paddle.bool:
            mask_with_buffer = paddle.concat(
                x=[paddle.zeros(shape=[x.shape[0], self.buffer_size], dtype="bool"), mask], axis=1
            )
        else:
            mask_with_buffer = paddle.concat(x=[paddle.zeros(shape=[x.shape[0], self.buffer_size]), mask], axis=1)

        mask_tokens = self.mask_token.tile(
            repeat_times=[mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1]
        ).astype(x.dtype)

        x_after_pad = mask_tokens.clone()
        mask_with_buffer_bool = (1 - mask_with_buffer).astype("bool")
        non_zero_indices = mask_with_buffer_bool.nonzero()
        if non_zero_indices.shape[0] == 0:
            rows = paddle.to_tensor([], dtype="int64")
            cols = paddle.to_tensor([], dtype="int64")
        else:
            rows = non_zero_indices[:, 0]
            cols = non_zero_indices[:, 1]

        x_reshaped = x.reshape([-1, x.shape[2]])
        x_after_pad[rows, cols] = x_reshaped

        x = x_after_pad + self.decoder_pos_embed_learned

        if self.grad_checkpointing and not paddle.in_dynamic_mode():
            for block in self.decoder_blocks:
                x = paddle.distributed.fleet.utils.recompute(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)

        x = self.decoder_norm(x)
        x = x[:, self.buffer_size :]
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = tuple(target.shape)
        target = target.reshape(bsz * seq_len, -1).tile(repeat_times=[self.diffusion_batch_mul, 1])
        z = z.reshape(bsz * seq_len, -1).tile(repeat_times=[self.diffusion_batch_mul, 1])
        mask = mask.reshape(bsz * seq_len).tile(repeat_times=self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def forward(self, imgs, labels):

        class_embedding = self.class_emb(labels)

        x = self.patchify(imgs)

        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.shape[0])
        mask = self.random_masking(x, orders)

        x = self.forward_mae_encoder(x, mask, class_embedding)

        z = self.forward_mae_decoder(x, mask)
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)
        return loss

    def sample_tokens(
        self,
        bsz,
        num_iter=64,
        cfg=1.0,
        cfg_schedule="linear",
        labels=None,
        temperature=1.0,
        progress=False,
    ):
        mask = paddle.ones(shape=[bsz, self.seq_len]).cuda(blocking=True)
        tokens = paddle.zeros(shape=[bsz, self.seq_len, self.token_embed_dim]).cuda(blocking=True)

        orders = self.sample_orders(bsz)
        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        for step in indices:
            cur_tokens = tokens.clone()
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.tile(repeat_times=[bsz, 1])
            if not cfg == 1.0:
                tokens = paddle.concat(x=[tokens, tokens], axis=0)
                class_embedding = paddle.concat(
                    x=[class_embedding, self.fake_latent.tile(repeat_times=[bsz, 1])],
                    axis=0,
                )
                mask = paddle.concat(x=[mask, mask], axis=0)
            x = self.forward_mae_encoder(tokens, mask, class_embedding)
            z = self.forward_mae_decoder(x, mask)

            mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
            mask_len = paddle.to_tensor(
                [np.floor(mask_ratio * self.seq_len)], dtype="float32", place=paddle.CUDAPlace(0)
            )

            mask_len = paddle.maximum(
                paddle.to_tensor([1.0], dtype="float32", place=paddle.CUDAPlace(0)),
                paddle.minimum((paddle.sum(mask, axis=-1, keepdim=True) - 1).astype("float32"), mask_len),
            )

            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].astype(dtype="bool")
            else:
                mask_to_pred = paddle.logical_xor(x=mask[:bsz].astype(dtype="bool"), y=mask_next.astype(dtype="bool"))
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = paddle.concat(x=[mask_to_pred, mask_to_pred], axis=0)

            nonzero_indices = mask_to_pred.nonzero(as_tuple=True)

            z = z[nonzero_indices].squeeze(1)
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError

            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(chunks=2, axis=0)
                mask_to_pred, _ = mask_to_pred.chunk(chunks=2, axis=0)

            flat_indices = paddle.nonzero(mask_to_pred.flatten()).flatten()
            cur_tokens = cur_tokens.reshape([-1, self.token_embed_dim])
            cur_tokens = paddle.scatter(
                cur_tokens, flat_indices, sampled_token_latent.reshape([-1, self.token_embed_dim])
            )
            cur_tokens = cur_tokens.reshape([bsz, self.seq_len, self.token_embed_dim])
            tokens = cur_tokens.clone()
        tokens = self.unpatchify(tokens)
        return tokens


def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=768,
        decoder_depth=12,
        decoder_num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(paddle.nn.LayerNorm, epsilon=1e-06),
        **kwargs,
    )
    return model


def mar_large(**kwargs):
    model = MAR(
        encoder_embed_dim=1024,
        encoder_depth=16,
        encoder_num_heads=16,
        decoder_embed_dim=1024,
        decoder_depth=16,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(paddle.nn.LayerNorm, epsilon=1e-06),
        **kwargs,
    )
    return model


def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1280,
        encoder_depth=20,
        encoder_num_heads=16,
        decoder_embed_dim=1280,
        decoder_depth=20,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(paddle.nn.LayerNorm, epsilon=1e-06),
        **kwargs,
    )
    return model

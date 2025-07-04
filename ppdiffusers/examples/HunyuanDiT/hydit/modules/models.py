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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdiffusers.configuration_utils import ConfigMixin, register_to_config
from ppdiffusers.models import ModelMixin

from .attn_layers import (
    Attention,
    CrossAttention,
    FlashCrossMHAModified,
    FlashSelfMHAModified,
)
from .embedders import PatchEmbed, TimestepEmbedder, timestep_embedding
from .mlp import Mlp
from .norm_layers import RMSNorm
from .poolers import AttentionPool


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FP32_Layernorm(nn.LayerNorm):
    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.astype(dtype="float32"),
            self._normalized_shape,
            self.weight.astype(dtype="float32"),
            self.bias.astype(dtype="float32"),
            self._epsilon,
        ).to(dtype=origin_dtype)


class FP32_SiLU(nn.Silu):
    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        return paddle.nn.functional.silu(inputs.astype(dtype="float32")).to(dtype=inputs.dtype)


class HunYuanDiTBlock(nn.Layer):
    """
    A HunYuanDiT block with `add` conditioning.
    """

    def __init__(
        self,
        hidden_size,
        c_emb_size,
        num_heads,
        mlp_ratio=4.0,
        text_states_dim=1024,
        use_flash_attn=False,
        qk_norm=False,
        norm_type="layer",
        skip=False,
    ):
        super().__init__()
        self.use_flash_attn = use_flash_attn

        if norm_type == "layer":
            norm_layer = FP32_Layernorm
        elif norm_type == "rms":
            norm_layer = RMSNorm
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # ========================= Self-Attention =========================
        self.norm1 = norm_layer(hidden_size, epsilon=1e-6)
        if use_flash_attn:
            self.attn1 = FlashSelfMHAModified(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm)
        else:
            self.attn1 = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm)

        # ========================= FFN =========================
        self.norm2 = norm_layer(hidden_size, epsilon=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        approx_gelu = lambda: paddle.nn.GELU(approximate=True)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        # ========================= Add =========================
        # Simply use add like SDXL.
        self.default_modulation = nn.Sequential(FP32_SiLU(), nn.Linear(c_emb_size, hidden_size, bias_attr=True))

        # ========================= Cross-Attention =========================
        if use_flash_attn:
            self.attn2 = FlashCrossMHAModified(
                hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm
            )
        else:
            self.attn2 = CrossAttention(
                hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm
            )
        self.norm3 = norm_layer(hidden_size, epsilon=1e-6)

        # ========================= Skip Connection =========================
        if skip:
            self.skip_norm = norm_layer(2 * hidden_size, epsilon=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None

    def forward(self, x, c=None, text_states=None, freq_cis_img=None, skip=None):
        # Long Skip Connection
        if self.skip_linear is not None:
            cat = paddle.concat([x, skip], axis=-1)
            cat = self.skip_norm(cat)
            x = self.skip_linear(cat)

        # Self-Attention
        shift_msa = self.default_modulation(c).unsqueeze(axis=1)
        attn_inputs = (
            self.norm1(x) + shift_msa,
            freq_cis_img,
        )
        x = x + self.attn1(*attn_inputs)[0]

        # Cross-Attention
        cross_inputs = (self.norm3(x), text_states, freq_cis_img)
        x = x + self.attn2(*cross_inputs)[0]

        # FFN Layer
        mlp_inputs = self.norm2(x)
        x = x + self.mlp(mlp_inputs)

        return x


class FinalLayer(nn.Layer):
    """
    The final layer of HunYuanDiT.
    """

    def __init__(self, final_hidden_size, c_emb_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(final_hidden_size, weight_attr=False, bias_attr=False, epsilon=1e-6)
        self.linear = nn.Linear(final_hidden_size, patch_size * patch_size * out_channels, bias_attr=True)
        self.adaLN_modulation = nn.Sequential(
            FP32_SiLU(), nn.Linear(c_emb_size, 2 * final_hidden_size, bias_attr=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class HunYuanDiT(ModelMixin, ConfigMixin):
    """
    HunYuanDiT: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    input_size: tuple
        The size of the input image.
    patch_size: int
        The size of the patch.
    in_channels: int
        The number of input channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    depth: int
        The number of transformer blocks.
    num_heads: int
        The number of attention heads.
    mlp_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    log_fn: callable
        The logging function.
    """

    @register_to_config
    def __init__(
        self,
        args,
        input_size=(32, 32),
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        log_fn=print,
    ):
        super().__init__()
        self.args = args
        self.log_fn = log_fn
        self.depth = depth
        self.learn_sigma = args.learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if args.learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.text_states_dim = args.text_states_dim
        self.text_states_dim_t5 = args.text_states_dim_t5
        self.text_len = args.text_len
        self.text_len_t5 = args.text_len_t5
        self.norm = args.norm

        use_flash_attn = args.infer_mode == "fa"
        if use_flash_attn:
            log_fn("    Enable Flash Attention.")
        qk_norm = True  # See http://arxiv.org/abs/2302.05442 for details.

        self.mlp_t5 = nn.Sequential(
            nn.Linear(self.text_states_dim_t5, self.text_states_dim_t5 * 4, bias_attr=True),
            FP32_SiLU(),
            nn.Linear(self.text_states_dim_t5 * 4, self.text_states_dim, bias_attr=True),
        )
        # learnable replace
        # self.text_embedding_padding = nn.Parameter(
        #     torch.randn(self.text_len + self.text_len_t5, self.text_states_dim, dtype=torch.float32))
        out_7 = paddle.create_parameter(
            shape=paddle.randn(shape=[self.text_len + self.text_len_t5, self.text_states_dim], dtype="float32").shape,
            dtype=paddle.randn(shape=[self.text_len + self.text_len_t5, self.text_states_dim], dtype="float32")
            .numpy()
            .dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.randn(shape=[self.text_len + self.text_len_t5, self.text_states_dim], dtype="float32")
            ),
        )
        out_7.stop_gradient = not True
        self.text_embedding_padding = out_7
        # Attention pooling
        self.pooler = AttentionPool(self.text_len_t5, self.text_states_dim_t5, num_heads=8, output_dim=1024)

        # Here we use a default learned embedder layer for future extension.
        self.style_embedder = nn.Embedding(1, hidden_size)

        # Image size and crop size conditions
        self.extra_in_dim = 256 * 6 + hidden_size

        # Text embedding for `add`
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.extra_in_dim += 1024
        self.extra_embedder = nn.Sequential(
            nn.Linear(self.extra_in_dim, hidden_size * 4),
            FP32_SiLU(),
            nn.Linear(hidden_size * 4, hidden_size, bias_attr=True),
        )

        # Image embedding
        num_patches = self.x_embedder.num_patches
        log_fn(f"    Number of tokens: {num_patches}")

        # HUnYuanDiT Blocks
        self.blocks = nn.LayerList(
            [
                HunYuanDiTBlock(
                    hidden_size=hidden_size,
                    c_emb_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    text_states_dim=self.text_states_dim,
                    use_flash_attn=use_flash_attn,
                    qk_norm=qk_norm,
                    norm_type=self.norm,
                    skip=layer > depth // 2,
                )
                for layer in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, hidden_size, patch_size, self.out_channels)
        self.unpatchify_channels = self.out_channels

        self.initialize_weights()

    def forward(
        self,
        x,
        t,
        encoder_hidden_states=None,
        text_embedding_mask=None,
        encoder_hidden_states_t5=None,
        text_embedding_mask_t5=None,
        image_meta_size=None,
        style=None,
        cos_cis_img=None,
        sin_cis_img=None,
        return_dict=True,
    ):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x: paddle.Tensor
            (B, D, H, W)
        t: paddle.Tensor
            (B)
        encoder_hidden_states: paddle.Tensor
            CLIP text embedding, (B, L_clip, D)
        text_embedding_mask: paddle.Tensor
            CLIP text embedding mask, (B, L_clip)
        encoder_hidden_states_t5: paddle.Tensor
            T5 text embedding, (B, L_t5, D)
        text_embedding_mask_t5: paddle.Tensor
            T5 text embedding mask, (B, L_t5)
        image_meta_size: paddle.Tensor
            (B, 6)
        style: paddle.Tensor
            (B)
        cos_cis_img: paddle.Tensor
        sin_cis_img: paddle.Tensor
        return_dict: bool
            Whether to return a dictionary.
        """

        text_states = encoder_hidden_states  # 2,77,1024
        text_states_t5 = encoder_hidden_states_t5  # 2,256,2048
        text_states_mask = text_embedding_mask.astype(dtype="bool")  # 2,77
        text_states_t5_mask = text_embedding_mask_t5.astype(dtype="bool")  # 2,256
        b_t5, l_t5, c_t5 = text_states_t5.shape
        text_states_t5 = self.mlp_t5(text_states_t5.reshape([-1, c_t5]))
        text_states = paddle.concat([text_states, text_states_t5.reshape([b_t5, l_t5, -1])], axis=1)  # 2,205，1024
        clip_t5_mask = paddle.concat([text_states_mask, text_states_t5_mask], axis=-1)

        clip_t5_mask = clip_t5_mask
        text_states = paddle.where(condition=clip_t5_mask.unsqueeze(2), x=text_states, y=self.text_embedding_padding)

        _, _, oh, ow = x.shape
        th, tw = oh // self.patch_size, ow // self.patch_size

        # ========================= Build time and image embedding =========================
        t = self.t_embedder(t)
        x = self.x_embedder(x)

        # Get image RoPE embedding according to `reso`lution.
        freqs_cis_img = (cos_cis_img, sin_cis_img)

        # ========================= Concatenate all extra vectors =========================
        # Build text tokens with pooling
        extra_vec = self.pooler(encoder_hidden_states_t5)

        # Build image meta size tokens
        image_meta_size = timestep_embedding(image_meta_size.reshape([-1]), 256)  # [B * 6, 256]
        if self.args.use_fp16:
            image_meta_size = image_meta_size.astype(dtype="float16")
        image_meta_size = image_meta_size.reshape([-1, 6 * 256])
        extra_vec = paddle.concat([extra_vec, image_meta_size], axis=1)  # [B, D + 6 * 256]

        # Build style tokens
        style_embedding = self.style_embedder(style)
        extra_vec = paddle.concat([extra_vec, style_embedding], axis=1)

        # Concatenate all extra vectors
        c = t + self.extra_embedder(extra_vec)  # [B, D]

        # ========================= Forward pass through HunYuanDiT blocks =========================
        skips = []
        for layer, block in enumerate(self.blocks):
            if layer > self.depth // 2:
                skip = skips.pop()
                x = block(x, c, text_states, freqs_cis_img, skip)  # (N, L, D)
            else:
                x = block(x, c, text_states, freqs_cis_img)  # (N, L, D)

            if layer < (self.depth // 2 - 1):
                skips.append(x)

        # ========================= Final layer =========================
        x = self.final_layer(x, c)  # (N, L, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, th, tw)  # (N, out_channels, H, W)

        if return_dict:
            return {"x": x}
        return x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                init_XavierUniform = nn.initializer.XavierUniform()
                init_XavierUniform(module.weight)
                if module.bias is not None:
                    init_Constant = nn.initializer.Constant(value=0)
                    init_Constant(module.bias)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(w.reshape([tuple(w.shape)[0], -1]))
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.x_embedder.proj.bias)

        # Initialize label embedding table:
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.extra_embedder[0].weight)
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.extra_embedder[2].weight)

        # Initialize timestep embedding MLP:
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.t_embedder.mlp[0].weight)
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.t_embedder.mlp[2].weight)

        # Zero-out adaLN modulation layers in HunYuanDiT blocks:
        for block in self.blocks:
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(block.default_modulation[-1].weight)
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(block.default_modulation[-1].bias)

        # Zero-out output layers:
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.final_layer.adaLN_modulation[-1].weight)
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.final_layer.adaLN_modulation[-1].bias)
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.final_layer.linear.weight)
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.final_layer.linear.bias)

    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        p = self.x_embedder.patch_size[0]
        # h = w = int(x.shape[1] ** 0.5)
        # westfish: squeeze x to [b, s, d]
        if len(x.shape) == 4:
            x = x.squeeze(axis=0)
        assert h * w == x.shape[1], f"Input tensor shape {x.shape} is not equal to height({h}) * width({w})."

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = paddle.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs


#################################################################################
#                            HunYuanDiT Configs                                 #
#################################################################################

HUNYUAN_DIT_CONFIG = {
    "DiT-g/2": {"depth": 40, "hidden_size": 1408, "patch_size": 2, "num_heads": 16, "mlp_ratio": 4.3637},
    "DiT-XL/2": {"depth": 28, "hidden_size": 1152, "patch_size": 2, "num_heads": 16},
    "DiT-L/2": {"depth": 24, "hidden_size": 1024, "patch_size": 2, "num_heads": 16},
    "DiT-B/2": {"depth": 12, "hidden_size": 768, "patch_size": 2, "num_heads": 12},
}

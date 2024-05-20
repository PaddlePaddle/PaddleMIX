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

import numpy as np
import paddle
from einops import rearrange
from models.layers.blocks import (
    Attention,
    CaptionEmbedder,
    DropPath,
    Mlp,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    PositionEmbedding2D,
    RotaryEmbedding,
    SizeEmbedder,
    T2IFinalLayer,
    approx_gelu,
    get_2d_sincos_pos_embed,
    get_layernorm,
    t2i_modulate,
)
from paddlenlp.transformers import PretrainedConfig, PretrainedModel

from ppdiffusers.models.dit_llama import TimestepEmbedder


class STDiT2Block(paddle.nn.Layer):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        rope=None,
        qk_norm=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn

        self.attn_cls = Attention
        self.mha_cls = MultiHeadCrossAttention

        # spatial branch
        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
        )

        paddle_random = paddle.randn(shape=(6, hidden_size))

        out_2 = paddle.create_parameter(
            shape=(paddle_random / hidden_size**0.5).shape,
            dtype=(paddle_random / hidden_size**0.5).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle_random / hidden_size**0.5),
        )

        out_2.stop_gradient = not True
        self.scale_shift_table = out_2

        # cross attn
        self.cross_attn = self.mha_cls(hidden_size, num_heads)

        # mlp branch
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()

        # temporal branch
        self.norm_temp = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)  # new
        self.attn_temp = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            rope=rope,
            qk_norm=qk_norm,
        )

        paddle_random = paddle.randn(shape=(3, hidden_size))
        out_3 = paddle.create_parameter(
            shape=(paddle_random / hidden_size**0.5).shape,
            dtype=(paddle_random / hidden_size**0.5).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle_random / hidden_size**0.5),
        )

        out_3.stop_gradient = not True
        self.scale_shift_table_temporal = out_3

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = paddle.where(condition=x_mask[:, :, None, None], x=x, y=masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(self, x, y, t, t_tmp, mask=None, x_mask=None, t0=None, t0_tmp=None, T=None, S=None):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape((B, 6, -1))
        ).chunk(6, axis=1)
        shift_tmp, scale_tmp, gate_tmp = (self.scale_shift_table_temporal[None] + t_tmp.reshape((B, 3, -1))).chunk(
            3, axis=1
        )
        if x_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                self.scale_shift_table[None] + t0.reshape((B, 6, -1))
            ).chunk(6, axis=1)
            shift_tmp_zero, scale_tmp_zero, gate_tmp_zero = (
                self.scale_shift_table_temporal[None] + t0_tmp.reshape((B, 3, -1))
            ).chunk(3, axis=1)

        # modulate
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm1(x), shift_msa_zero, scale_msa_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # spatial branch
        x_s = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
        x_s = self.attn(x_s)
        x_s = rearrange(x_s, "(B T) S C -> B (T S) C", T=T, S=S)
        if x_mask is not None:
            x_s_zero = gate_msa_zero * x_s
            x_s = gate_msa * x_s
            x_s = self.t_mask_select(x_mask, x_s, x_s_zero, T, S)
        else:
            x_s = gate_msa * x_s
        x = x + self.drop_path(x_s)

        # modulate
        x_m = t2i_modulate(self.norm_temp(x), shift_tmp, scale_tmp)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm_temp(x), shift_tmp_zero, scale_tmp_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # temporal branch
        x_t = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
        x_t = self.attn_temp(x_t)
        x_t = rearrange(x_t, "(B S) T C -> B (T S) C", T=T, S=S)
        if x_mask is not None:
            x_t_zero = gate_tmp_zero * x_t
            x_t = gate_tmp * x_t
            x_t = self.t_mask_select(x_mask, x_t, x_t_zero, T, S)
        else:
            x_t = gate_tmp * x_t
        x = x + self.drop_path(x_t)

        # cross attn
        x = x + self.cross_attn(x, y, mask)

        # modulate
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero, scale_mlp_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        # mlp
        x_mlp = self.mlp(x_m)
        if x_mask is not None:
            x_mlp_zero = gate_mlp_zero * x_mlp
            x_mlp = gate_mlp * x_mlp
            x_mlp = self.t_mask_select(x_mask, x_mlp, x_mlp_zero, T, S)
        else:
            x_mlp = gate_mlp * x_mlp
        x = x + self.drop_path(x_mlp)

        return x


class STDiT2Config(PretrainedConfig):

    model_type = "STDiT2"

    def __init__(
        self,
        input_size=(None, None, None),
        input_sq_size=32,
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.0,
        no_temporal_pos_emb=False,
        caption_channels=4096,
        model_max_length=120,
        freeze=None,
        qk_norm=False,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        **kwargs,
    ):
        self.input_size = input_size
        self.input_sq_size = input_sq_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.pred_sigma = pred_sigma
        self.drop_path = drop_path
        self.no_temporal_pos_emb = no_temporal_pos_emb
        self.caption_channels = caption_channels
        self.model_max_length = model_max_length
        self.freeze = freeze
        self.qk_norm = qk_norm
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        super().__init__(**kwargs)


class STDiT2(PretrainedModel):

    config_class = STDiT2Config

    def __init__(self, config):
        super().__init__(config)
        self.pred_sigma = config.pred_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.pred_sigma else config.in_channels
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.no_temporal_pos_emb = config.no_temporal_pos_emb
        self.depth = config.depth
        self.mlp_ratio = config.mlp_ratio
        self.enable_flash_attn = config.enable_flash_attn
        self.enable_layernorm_kernel = config.enable_layernorm_kernel

        # support dynamic input
        self.patch_size = config.patch_size
        self.input_size = config.input_size
        self.input_sq_size = config.input_sq_size
        self.pos_embed = PositionEmbedding2D(config.hidden_size)

        self.x_embedder = PatchEmbed3D(config.patch_size, config.in_channels, config.hidden_size)
        self.t_embedder = TimestepEmbedder(config.hidden_size)

        self.t_block = paddle.nn.Sequential(
            paddle.nn.Silu(),
            paddle.nn.Linear(in_features=config.hidden_size, out_features=6 * config.hidden_size, bias_attr=True),
        )
        self.t_block_temp = paddle.nn.Sequential(
            paddle.nn.Silu(),
            paddle.nn.Linear(in_features=config.hidden_size, out_features=3 * config.hidden_size, bias_attr=True),
        )  # new

        self.y_embedder = CaptionEmbedder(
            in_channels=config.caption_channels,
            hidden_size=config.hidden_size,
            uncond_prob=config.class_dropout_prob,
            act_layer=approx_gelu,
            token_num=config.model_max_length,
        )

        drop_path = [x.item() for x in paddle.linspace(start=0, stop=config.drop_path, num=config.depth)]

        self.rope = RotaryEmbedding(dim=self.hidden_size // self.num_heads)  # new
        self.blocks = paddle.nn.LayerList(
            [
                STDiT2Block(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flash_attn=self.enable_flash_attn,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    rope=self.rope.rotate_queries_or_keys,
                    qk_norm=config.qk_norm,
                )
                for i in range(self.depth)
            ]
        )
        self.final_layer = T2IFinalLayer(config.hidden_size, np.prod(self.patch_size), self.out_channels)

        # multi_res
        assert self.hidden_size % 3 == 0, "hidden_size must be divisible by 3"
        self.csize_embedder = SizeEmbedder(self.hidden_size // 3)
        self.ar_embedder = SizeEmbedder(self.hidden_size // 3)
        self.fl_embedder = SizeEmbedder(self.hidden_size)  # new
        self.fps_embedder = SizeEmbedder(self.hidden_size)  # new

        # init model
        self.initialize_weights()
        self.initialize_temporal()
        if config.freeze is not None:
            assert config.freeze in ["not_temporal", "text"]
            if config.freeze == "not_temporal":
                self.freeze_not_temporal()
            elif config.freeze == "text":
                self.freeze_text()

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.shape
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def forward(
        self, x, timestep, y, mask=None, x_mask=None, num_frames=None, height=None, width=None, ar=None, fps=None
    ):
        """
        Forward pass of STDiT.
        Args:
            x (paddle.Tensor): latent representation of video; of shape [B, C, T, H, W]
            timestep (paddle.Tensor): diffusion time steps; of shape [B]
            y (paddle.Tensor): representation of prompts; of shape [B, 1, N_token, C]
            mask (paddle.Tensor): mask for selecting prompt tokens; of shape [B, N_token]

        Returns:
            x (paddle.Tensor): output latent representation; of shape [B, C, T, H, W]
        """

        B = x.shape[0]
        dtype = self.x_embedder.proj.weight.dtype

        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)

        # === process data info ===
        # 1. get dynamic size

        hw = paddle.concat(x=[height[:, None], width[:, None]], axis=1)
        rs = (height[0].item() * width[0].item()) ** 0.5
        csize = self.csize_embedder(hw, B)

        # 2. get aspect ratio
        ar = ar.unsqueeze(1)
        ar = self.ar_embedder(ar, B)

        data_info = paddle.concat(x=[csize, ar], axis=1)

        # 3. get number of frames
        fl = num_frames.unsqueeze(1)
        fps = fps.unsqueeze(1)
        fl = self.fl_embedder(fl, B)
        fl = fl + self.fps_embedder(fps, B)

        # === get dynamic shape size ===

        _, _, Tx, Hx, Wx = x.shape
        T, H, W = self.get_dynamic_size(x)
        S = H * W
        scale = rs / self.input_sq_size
        base_size = round(S**0.5)
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # embedding
        x = self.x_embedder(x)  # [B, N, C]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb
        x = rearrange(x, "B T S C -> B (T S) C")

        # prepare adaIN

        t = self.t_embedder(timestep)
        t_spc = t + data_info  # [B, C]
        t_tmp = t + fl  # [B, C]
        t_spc_mlp = self.t_block(t_spc)  # [B, 6*C]
        t_tmp_mlp = self.t_block_temp(t_tmp)  # [B, 3*C]
        if x_mask is not None:

            t0_timestep = paddle.zeros_like(x=timestep)
            t0 = self.t_embedder(t0_timestep)
            t0_spc = t0 + data_info
            t0_tmp = t0 + fl
            t0_spc_mlp = self.t_block(t0_spc)
            t0_tmp_mlp = self.t_block_temp(t0_tmp)
        else:
            t0_spc = None
            t0_tmp = None
            t0_spc_mlp = None
            t0_tmp_mlp = None

        # prepare y
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]

        if mask is not None:
            if mask.shape[0] != y.shape[0]:

                mask = mask.tile((y.shape[0] // mask.shape[0], 1))
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).reshape([1, -1, x.shape[-1]])
            y_lens = mask.sum(axis=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]

            y = y.squeeze(1).reshape([1, -1, x.shape[-1]])

        # blocks
        for _, block in enumerate(self.blocks):

            x = block(
                x,
                y,
                t_spc_mlp,
                t_tmp_mlp,
                y_lens,
                x_mask,
                t0_spc_mlp,
                t0_tmp_mlp,
                T,
                S,
            )

        # final process
        x = self.final_layer(x, t, x_mask, t0_spc, T, S)  # [B, N, C=T_p * H_p * W_p * C_out]
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)  # [B, C_out, T, H, W]

        # cast to float32 for better accuracy
        x = x.to("float32")
        return x

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (paddle.Tensor): of shape [B, N, C]

        Return:
            x (paddle.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x

    def unpatchify_old(self, x):
        c = self.out_channels
        t, h, w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        pt, ph, pw = self.patch_size

        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = rearrange(x, "n t h w r p q c -> n c t r h p w q")
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def get_spatial_pos_embed(self, H, W, scale=1.0, base_size=None):
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (H, W),
            scale=scale,
            base_size=base_size,
        )

        out_4 = paddle.to_tensor(data=pos_embed).astype(dtype="float32").unsqueeze(axis=0)
        out_4.stop_gradient = not False
        pos_embed = out_4
        return pos_embed

    def freeze_not_temporal(self):
        for n, p in self.named_parameters():
            if "attn_temp" not in n:
                # p.requires_grad = False
                p.stop_gradient = not False

    def freeze_text(self):
        for n, p in self.named_parameters():
            if "cross_attn" in n:

                p.stop_gradient = not False

    def initialize_temporal(self):
        for block in self.blocks:

            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(block.attn_temp.proj.weight)
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(block.attn_temp.proj.bias)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, paddle.nn.Linear):

                init_XavierUniform = paddle.nn.initializer.XavierUniform()
                init_XavierUniform(module.weight)
                if module.bias is not None:

                    init_Constant = paddle.nn.initializer.Constant(value=0)
                    init_Constant(module.bias)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data

        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(w.reshape([w.shape[0], -1]))

        # Initialize timestep embedding MLP:

        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.t_embedder.mlp[0].weight)
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.t_embedder.mlp[2].weight)
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.t_block[1].weight)
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.t_block_temp[1].weight)

        # Initialize caption embedding MLP:

        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.y_embedder.y_proj.fc1.weight)
        init_Normal = paddle.nn.initializer.Normal(std=0.02)
        init_Normal(self.y_embedder.y_proj.fc2.weight)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:

            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(block.cross_attn.proj.weight)
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(block.cross_attn.proj.bias)

        # Zero-out output layers:

        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.final_layer.linear.weight)
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(self.final_layer.linear.bias)

from einops import rearrange

from .modelscope_st_unet import STUNetModel, default, prob_mask_like, sinusoidal_embedding_paddle, STUNetOutput, \
    TemporalTransformer, TemporalAttentionMultiBlock, SpatialTransformer, ResBlock
from ..configuration_utils import register_to_config
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

USE_TEMPORAL_TRANSFORMER = True


class Downsample(nn.Layer):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=(2, 1)):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = nn.Conv2D(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(nn.Layer):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = nn.Conv2D(self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = x[..., 1:-1, :]
        if self.use_conv:
            x = self.conv(x)
        return x


class Vid2VidSTUNet(STUNetModel):
    @register_to_config
    def __init__(self,
                 in_channels=4,
                 out_channels=4,
                 dim=320,
                 y_dim=1024,
                 context_channels=1024,
                 dim_mult=[1, 2, 4, 4],
                 num_heads=8,
                 head_dim=64,
                 num_res_blocks=2,
                 attn_scales=[1 / 1, 1 / 2, 1 / 4],
                 use_scale_shift_norm=True,
                 dropout=0.1,
                 temporal_attn_times=1,
                 temporal_attention=True,
                 use_checkpoint=True,
                 use_image_dataset=False,
                 use_fps_condition=False,
                 use_sim_mask=False,
                 training=False,
                 inpainting=True,
                 **kwargs):
        super(Vid2VidSTUNet, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dim=dim,
            y_dim=y_dim,
            context_channels=context_channels,
            dim_mult=dim_mult,
            num_heads=num_heads,
            head_dim=head_dim,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            use_scale_shift_norm=use_scale_shift_norm,
            dropout=dropout,
            temporal_attn_times=temporal_attn_times,
            temporal_attention=temporal_attention
        )
        embed_dim = dim * 4
        num_heads = num_heads if num_heads else dim // 32
        self.in_dim = in_channels
        self.dim = dim
        self.y_dim = y_dim
        self.context_dim = context_channels
        self.embed_dim = embed_dim
        self.out_dim = out_channels
        self.dim_mult = dim_mult
        # for temporal attention
        self.num_heads = num_heads
        # for spatial attention
        self.head_dim = head_dim
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.use_scale_shift_norm = use_scale_shift_norm
        self.temporal_attn_times = temporal_attn_times
        self.temporal_attention = temporal_attention
        self.inpainting = inpainting
        self.use_fps_condition = use_fps_condition

        use_linear_in_temporal = False
        transformer_depth = 1
        disabled_sa = False

        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0

        if self.use_fps_condition:
            self.fps_embedding = nn.Sequential(
                nn.Linear(dim, embed_dim),
                nn.Silu(),
                nn.Linear(
                    embed_dim,
                    embed_dim,
                    weight_attr=nn.initializer.Constant(value=0.0),
                    bias_attr=nn.initializer.Constant(value=0.0),
                ),
            )

            # encoder
        self.input_blocks = nn.LayerList()
        init_block = nn.LayerList([nn.Conv2D(self.in_dim, dim, 3, padding=1)])
        if temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                init_block.append(
                    TemporalTransformer(
                        dim,
                        num_heads,
                        head_dim,
                        depth=transformer_depth,
                        context_dim=context_channels,
                        disable_self_attn=disabled_sa,
                        use_linear=use_linear_in_temporal,
                        multiply_zero=use_image_dataset,
                    )
                )
            else:
                init_block.append(
                    TemporalAttentionMultiBlock(
                        dim,
                        num_heads,
                        head_dim,
                        rotary_emb=self.rotary_emb,
                        temporal_attn_times=temporal_attn_times,
                        use_image_dataset=use_image_dataset))

        self.input_blocks.append(init_block)
        shortcut_dims.append(dim)
        for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                block = nn.LayerList(
                    [
                        ResBlock(
                            in_dim,
                            embed_dim,
                            dropout,
                            out_channels=out_dim,
                            use_scale_shift_norm=False,
                            use_image_dataset=use_image_dataset,
                        )
                    ]
                )
                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim,
                            out_dim // head_dim,
                            head_dim,
                            depth=1,
                            context_dim=self.context_dim,
                            disable_self_attn=False,
                            use_linear=True,
                        )
                    )
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(
                                TemporalTransformer(
                                    out_dim,
                                    out_dim // head_dim,
                                    head_dim,
                                    depth=transformer_depth,
                                    context_dim=context_channels,
                                    disable_self_attn=disabled_sa,
                                    use_linear=use_linear_in_temporal,
                                    multiply_zero=use_image_dataset,
                                )
                            )
                        else:
                            block.append(
                                TemporalAttentionMultiBlock(
                                    out_dim,
                                    num_heads,
                                    head_dim,
                                    rotary_emb=self.rotary_emb,
                                    use_image_dataset=use_image_dataset,
                                    use_sim_mask=use_sim_mask,
                                    temporal_attn_times=temporal_attn_times,
                                )
                            )
                in_dim = out_dim
                self.input_blocks.append(block)
                shortcut_dims.append(out_dim)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    downsample = Downsample(out_dim, True, dims=2, out_channels=out_dim)
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    self.input_blocks.append(downsample)

        # decoder
        self.output_blocks = nn.LayerList()
        for i, (in_dim, out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            for j in range(num_res_blocks + 1):
                block = nn.LayerList(
                    [
                        ResBlock(
                            in_dim + shortcut_dims.pop(),
                            embed_dim,
                            dropout,
                            out_dim,
                            use_scale_shift_norm=False,
                            use_image_dataset=use_image_dataset,
                        )
                    ]
                )
                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim,
                            out_dim // head_dim,
                            head_dim,
                            depth=1,
                            context_dim=1024,
                            disable_self_attn=False,
                            use_linear=True,
                        )
                    )

                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(
                                TemporalTransformer(
                                    out_dim,
                                    out_dim // head_dim,
                                    head_dim,
                                    depth=transformer_depth,
                                    context_dim=context_channels,
                                    disable_self_attn=disabled_sa,
                                    use_linear=use_linear_in_temporal,
                                    multiply_zero=use_image_dataset,
                                )
                            )
                        else:
                            block.append(
                                TemporalAttentionMultiBlock(
                                    out_dim,
                                    num_heads,
                                    head_dim,
                                    rotary_emb=self.rotary_emb,
                                    use_image_dataset=use_image_dataset,
                                    use_sim_mask=use_sim_mask,
                                    temporal_attn_times=temporal_attn_times,
                                )
                            )

                in_dim = out_dim

                # upsample
                if i != len(dim_mult) - 1 and j == num_res_blocks:
                    upsample = Upsample(out_dim, True, dims=2, out_channels=out_dim)
                    scale *= 2.0
                    block.append(upsample)
                self.output_blocks.append(block)

    def forward(self,
                x,
                t,
                y,
                x_lr=None,
                fps=None,
                video_mask=None,
                focus_present_mask=None,
                prob_focus_present=0.,
                mask_last_frame_num=0,
                return_dict: bool = True,
                **kwargs):
        batch, x_c, x_f, x_h, x_w = x.shape
        device = x.place
        self.batch = batch

        # image and video joint training, if mask_last_frame_num is set, prob_focus_present will be ignored
        if mask_last_frame_num > 0:
            focus_present_mask = None
            video_mask[-mask_last_frame_num:] = False
        else:
            focus_present_mask = default(
                focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device=device)
            )

        time_rel_pos_bias = None

        # embeddings
        e = self.time_embed(sinusoidal_embedding_paddle(t, self.dim))
        context = y

        # repeat f times for spatial e and context
        e = e.repeat_interleave(repeats=x_f, axis=0)
        context = context.repeat_interleave(repeats=x_f, axis=0)

        # always in shape (b f) c h w, except for temporal layer
        x = rearrange(x, "b c f h w -> (b f) c h w")
        # encoder
        xs = []
        for block in self.input_blocks:
            x = self._forward_single(block, x, e, context, time_rel_pos_bias, focus_present_mask, video_mask)
            xs.append(x)

        # middle
        for block in self.middle_block:
            x = self._forward_single(block, x, e, context, time_rel_pos_bias, focus_present_mask, video_mask)

        # decoder
        for block in self.output_blocks:
            x = paddle.concat([x, xs.pop()], axis=1)
            x = self._forward_single(
                block,
                x,
                e,
                context,
                time_rel_pos_bias,
                focus_present_mask,
                video_mask,
                reference=xs[-1] if len(xs) > 0 else None,
            )

        # head
        x = self.out(x)

        # reshape back to (b c f h w)
        sample = rearrange(x, "(b f) c h w -> b c f h w", b=batch)

        if not return_dict:
            return (sample,)

        return STUNetOutput(sample=sample)

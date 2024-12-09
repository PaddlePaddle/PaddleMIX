import sys
import paddle
from einops import rearrange


def swish(x):
    return x * paddle.nn.functional.sigmoid(x=x)


class ResBlock(paddle.nn.Layer):

    def __init__(self, in_filters, out_filters, use_conv_shortcut=False,
        use_agn=False) ->None:
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_conv_shortcut = use_conv_shortcut
        self.use_agn = use_agn
        if not use_agn:
            self.norm1 = paddle.nn.GroupNorm(num_groups=32, num_channels=in_filters, epsilon=1e-06)
            
        self.norm2 = paddle.nn.GroupNorm(num_groups=32, num_channels=out_filters, epsilon=1e-06)
        
        self.conv1 = paddle.nn.Conv2D(in_channels=in_filters, out_channels=
            out_filters, kernel_size=(3, 3), padding=1, bias_attr=False)
        self.conv2 = paddle.nn.Conv2D(in_channels=out_filters, out_channels
            =out_filters, kernel_size=(3, 3), padding=1, bias_attr=False)
        if in_filters != out_filters:
            if self.use_conv_shortcut:
                self.conv_shortcut = paddle.nn.Conv2D(in_channels=
                    in_filters, out_channels=out_filters, kernel_size=(3, 3
                    ), padding=1, bias_attr=False)
            else:
                self.nin_shortcut = paddle.nn.Conv2D(in_channels=in_filters,
                    out_channels=out_filters, kernel_size=(1, 1), padding=0,
                    bias_attr=False)

    def forward(self, x, **kwargs):
        residual = x
    
        if not self.use_agn:
            x = self.norm1(x)
  
        x = swish(x)

        x = self.conv1(x)
     
        x = self.norm2(x)

        x = swish(x)

        x = self.conv2(x)
 
  
        if self.in_filters != self.out_filters:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)
        return x + residual


class Encoder(paddle.nn.Layer):

    def __init__(self, *, ch, out_ch, in_channels, num_res_blocks,
        z_channels, ch_mult=(1, 2, 2, 4), resolution, double_z=False):
        super().__init__()
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(ch_mult)
        self.conv_in = paddle.nn.Conv2D(in_channels=in_channels,
            out_channels=ch, kernel_size=(3, 3), padding=1, bias_attr=False)
        self.down = paddle.nn.LayerList()
        in_ch_mult = (1,) + tuple(ch_mult)
        for i_level in range(self.num_blocks):
            block = paddle.nn.LayerList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out
            down = paddle.nn.Layer()
            down.block = block
            if i_level < self.num_blocks - 1:
                down.downsample = paddle.nn.Conv2D(in_channels=block_out,
                    out_channels=block_out, kernel_size=(3, 3), stride=(2, 
                    2), padding=1)
            self.down.append(down)
        self.mid_block = paddle.nn.LayerList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in))
        self.norm_out = paddle.nn.GroupNorm(num_groups=32, num_channels=
            block_out, epsilon=1e-06)
        self.conv_out = paddle.nn.Conv2D(in_channels=block_out,
            out_channels=z_channels, kernel_size=(1, 1))

    def forward(self, x):

        x = self.conv_in(x)
      
        for i_level in range(self.num_blocks):
            for i_block in range(self.num_res_blocks):
                x = self.down[i_level].block[i_block](x)
      
            if i_level < self.num_blocks - 1:
                x = self.down[i_level].downsample(x)
     
        for res in range(self.num_res_blocks):
            x = self.mid_block[res](x)
       
        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)
        
        return x


class Decoder(paddle.nn.Layer):

    def __init__(self, *, ch, out_ch, in_channels, num_res_blocks,
        z_channels, ch_mult=(1, 2, 2, 4), resolution, double_z=False) ->None:
        super().__init__()
        self.ch = ch
        self.num_blocks = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        block_in = ch * ch_mult[self.num_blocks - 1]
        self.conv_in = paddle.nn.Conv2D(in_channels=z_channels,
            out_channels=block_in, kernel_size=(3, 3), padding=1, bias_attr
            =True)
        self.mid_block = paddle.nn.LayerList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in))
        self.up = paddle.nn.LayerList()
        self.adaptive = paddle.nn.LayerList()
        for i_level in reversed(range(self.num_blocks)):
            block = paddle.nn.LayerList()
            block_out = ch * ch_mult[i_level]
            if len(self.adaptive) == 0:
                self.adaptive.append(AdaptiveGroupNorm(z_channels, block_in))
            else:
                self.adaptive.insert(0, AdaptiveGroupNorm(z_channels, block_in))

            for i_block in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out
            up = paddle.nn.Layer()
            up.block = block
            if i_level > 0:
                up.upsample = Upsampler(block_in)
            if len(self.up) == 0:
                self.up.append(up)
            else:
                self.up.insert(0, up)
        self.norm_out = paddle.nn.GroupNorm(num_groups=32, num_channels=
            block_in, epsilon=1e-06)
        self.conv_out = paddle.nn.Conv2D(in_channels=block_in, out_channels
            =out_ch, kernel_size=(3, 3), padding=1)

    def forward(self, z):
        style = z.clone()
        z = self.conv_in(z)
        for res in range(self.num_res_blocks):
            z = self.mid_block[res](z)
        for i_level in reversed(range(self.num_blocks)):
            z = self.adaptive[i_level](z, style)
            for i_block in range(self.num_res_blocks):
                z = self.up[i_level].block[i_block](z)
            if i_level > 0:
                z = self.up[i_level].upsample(z)
        z = self.norm_out(z)
        z = swish(z)
        z = self.conv_out(z)
        return z


def depth_to_space(x: paddle.Tensor, block_size: int) ->paddle.Tensor:
    """ Depth-to-Space DCR mode (depth-column-row) core implementation.

        Args:
            x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
            block_size (int): block side size
    """
    if x.dim() < 3:
        raise ValueError(
            f'Expecting a channels-first (*CHW) tensor of at least 3 dimensions'
            )
    c, h, w = tuple(x.shape)[-3:]
    s = block_size ** 2
    if c % s != 0:
        raise ValueError(
            f'Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels'
            )
    outer_dims = tuple(x.shape)[:-3]
    x = x.reshape([-1, block_size, block_size, c // s, h, w])
    x = x.transpose(perm=[0, 3, 4, 1, 5, 2])
    x = x.contiguous().view(*outer_dims, c // s, h * block_size, w * block_size
        )
    return x


class Upsampler(paddle.nn.Layer):

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim * 4
        self.conv1 = paddle.nn.Conv2D(in_channels=dim, out_channels=dim_out,
            kernel_size=(3, 3), padding=1)
        self.depth2space = depth_to_space

    def forward(self, x):
        """
        input_image: [B C H W]
        """
        out = self.conv1(x)
        out = self.depth2space(out, block_size=2)
        return out


class AdaptiveGroupNorm(paddle.nn.Layer):

    def __init__(self, z_channel, in_filters, num_groups=32, eps=1e-06):
        super().__init__()
        self.gn = paddle.nn.GroupNorm(num_groups=32, num_channels=
            in_filters, epsilon=eps, weight_attr=False, bias_attr=False)
        self.gamma = paddle.nn.Linear(in_features=z_channel, out_features=
            in_filters)
        self.beta = paddle.nn.Linear(in_features=z_channel, out_features=
            in_filters)
        self.eps = eps

    def forward(self, x, quantizer):
        B, C, _, _ = tuple(x.shape)
        scale = rearrange(quantizer, 'b c h w -> b c (h w)')
        scale = scale.var(axis=-1) + self.eps
        scale = scale.sqrt()
        scale = self.gamma(scale).view(B, C, 1, 1)
        bias = rearrange(quantizer, 'b c h w -> b c (h w)')
        bias = bias.mean(axis=-1)
        bias = self.beta(bias).view(B, C, 1, 1)
        x = self.gn(x)
        x = scale * x + bias
        return x

import paddle
import functools
from taming.modules.util import ActNorm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init_Normal = paddle.nn.initializer.Normal(mean=0.0, std=0.02)
        init_Normal(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        init_Normal = paddle.nn.initializer.Normal(mean=1.0, std=0.02)
        init_Normal(m.weight.data)
        init_Constant = paddle.nn.initializer.Constant(value=0)
        init_Constant(m.bias.data)


class NLayerDiscriminator(paddle.nn.Layer):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = paddle.nn.BatchNorm2D
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != paddle.nn.BatchNorm2D
        else:
            use_bias = norm_layer != paddle.nn.BatchNorm2D
        kw = 4
        padw = 1
        sequence = [paddle.nn.Conv2D(in_channels=input_nc, out_channels=ndf,
            kernel_size=kw, stride=2, padding=padw), paddle.nn.LeakyReLU(
            negative_slope=0.2)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [paddle.nn.Conv2D(in_channels=ndf * nf_mult_prev,
                out_channels=ndf * nf_mult, kernel_size=kw, stride=2,
                padding=padw, bias_attr=use_bias), norm_layer(ndf * nf_mult
                ), paddle.nn.LeakyReLU(negative_slope=0.2)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [paddle.nn.Conv2D(in_channels=ndf * nf_mult_prev,
            out_channels=ndf * nf_mult, kernel_size=kw, stride=1, padding=
            padw, bias_attr=use_bias), norm_layer(ndf * nf_mult), paddle.nn
            .LeakyReLU(negative_slope=0.2)]
        sequence += [paddle.nn.Conv2D(in_channels=ndf * nf_mult,
            out_channels=1, kernel_size=kw, stride=1, padding=padw)]
        self.main = paddle.nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)

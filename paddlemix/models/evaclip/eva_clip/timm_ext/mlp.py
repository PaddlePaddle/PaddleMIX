import paddle
""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""
from functools import partial
from .timm_ext import to_2tuple


class Mlp(paddle.nn.Layer):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=paddle.nn.GELU,
                 norm_layer=None,
                 bias=True,
                 drop=0.0,
                 use_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(
            paddle.nn.Conv2D, kernel_size=1) if use_conv else paddle.nn.Linear
        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = paddle.nn.Dropout(p=drop_probs[0])
        self.norm = norm_layer(
            hidden_features) if norm_layer is not None else paddle.nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = paddle.nn.Dropout(p=drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

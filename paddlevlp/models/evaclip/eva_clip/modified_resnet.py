import paddle
from collections import OrderedDict
from .utils import freeze_batch_norm_2d


class Bottleneck(paddle.nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=1,
            bias_attr=False)
        self.bn1 = paddle.nn.BatchNorm2D(
            num_features=planes,
            momentum=1 - 0.1,
            epsilon=1e-05,
            weight_attr=None,
            bias_attr=None,
            use_global_stats=True)
        self.act1 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            padding=1,
            bias_attr=False)
        self.bn2 = paddle.nn.BatchNorm2D(
            num_features=planes,
            momentum=1 - 0.1,
            epsilon=1e-05,
            weight_attr=None,
            bias_attr=None,
            use_global_stats=True)
        self.act2 = paddle.nn.ReLU()
        self.avgpool = paddle.nn.AvgPool2D(
            kernel_size=stride,
            exclusive=False) if stride > 1 else paddle.nn.Identity()
        self.conv3 = paddle.nn.Conv2D(
            in_channels=planes,
            out_channels=planes * self.expansion,
            kernel_size=1,
            bias_attr=False)
        self.bn3 = paddle.nn.BatchNorm2D(
            num_features=planes * self.expansion,
            momentum=1 - 0.1,
            epsilon=1e-05,
            weight_attr=None,
            bias_attr=None,
            use_global_stats=True)
        self.act3 = paddle.nn.ReLU()
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = paddle.nn.Sequential(* [(
                '-1', paddle.nn.AvgPool2D(
                    kernel_size=stride, exclusive=False)), (
                        '0', paddle.nn.Conv2D(
                            in_channels=inplanes,
                            out_channels=planes * self.expansion,
                            kernel_size=1,
                            stride=1,
                            bias_attr=False)), ('1', paddle.nn.BatchNorm2D(
                                num_features=planes * self.expansion,
                                momentum=1 - 0.1,
                                epsilon=1e-05,
                                weight_attr=None,
                                bias_attr=None,
                                use_global_stats=True))])

    def forward(self, x: paddle.Tensor):
        identity = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act3(out)
        return out


class AttentionPool2d(paddle.nn.Layer):
    def __init__(self,
                 spacial_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 output_dim: int=None):
        super().__init__()
        init_data = paddle.randn(shape=[spacial_dim**2 + 1,
                                        embed_dim]) / embed_dim**0.5
        self.positional_embedding = self.create_parameter(
            shape=[spacial_dim**2 + 1, embed_dim],
            default_initializer=paddle.nn.initializer.Assign(init_data))
        self.k_proj = paddle.nn.Linear(
            in_features=embed_dim, out_features=embed_dim)
        self.q_proj = paddle.nn.Linear(
            in_features=embed_dim, out_features=embed_dim)
        self.v_proj = paddle.nn.Linear(
            in_features=embed_dim, out_features=embed_dim)
        self.c_proj = paddle.nn.Linear(
            in_features=embed_dim, out_features=output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1],
                      x.shape[2] * x.shape[3]).transpose(perm=[2, 0, 1])
        x = paddle.concat(x=[x.mean(axis=0, keepdim=True), x], axis=0)
        if isinstance(x.dtype, paddle.dtype):
            dtype = x.dtype
        elif isinstance(x.dtype,
                        str) and x.dtype not in ['cpu', 'cuda', 'ipu', 'xpu']:
            dtype = x.dtype
        elif isinstance(x.dtype, paddle.Tensor):
            dtype = x.dtype.dtype
        else:
            dtype = self.positional_embedding[:, (None), :].dtype
        x = x + self.positional_embedding[:, (None), :].cast(dtype)
        # may never use
        # x, _ = torch.nn.functional.multi_head_attention_forward(query=x,
        #     key=x, value=x, embed_dim_to_check=x.shape[-1], num_heads=self.
        #     num_heads, q_proj_weight=self.q_proj.weight, k_proj_weight=self
        #     .k_proj.weight, v_proj_weight=self.v_proj.weight,
        #     in_proj_weight=None, in_proj_bias=paddle.concat(x=[self.q_proj.
        #     bias, self.k_proj.bias, self.v_proj.bias]), bias_k=None, bias_v
        #     =None, add_zero_attn=False, dropout_p=0.0, out_proj_weight=self
        #     .c_proj.weight, out_proj_bias=self.c_proj.bias,
        #     use_separate_proj_weight=True, training=self.training,
        #     need_weights=False)
        return x[0]


class ModifiedResNet(paddle.nn.Layer):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size
        self.conv1 = paddle.nn.Conv2D(
            in_channels=3,
            out_channels=width // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias_attr=False)
        self.bn1 = paddle.nn.BatchNorm2D(
            num_features=width // 2,
            momentum=1 - 0.1,
            epsilon=1e-05,
            weight_attr=None,
            bias_attr=None,
            use_global_stats=True)
        self.act1 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(
            in_channels=width // 2,
            out_channels=width // 2,
            kernel_size=3,
            padding=1,
            bias_attr=False)
        self.bn2 = paddle.nn.BatchNorm2D(
            num_features=width // 2,
            momentum=1 - 0.1,
            epsilon=1e-05,
            weight_attr=None,
            bias_attr=None,
            use_global_stats=True)
        self.act2 = paddle.nn.ReLU()
        self.conv3 = paddle.nn.Conv2D(
            in_channels=width // 2,
            out_channels=width,
            kernel_size=3,
            padding=1,
            bias_attr=False)
        self.bn3 = paddle.nn.BatchNorm2D(
            num_features=width,
            momentum=1 - 0.1,
            epsilon=1e-05,
            weight_attr=None,
            bias_attr=None,
            use_global_stats=True)
        self.act3 = paddle.nn.ReLU()
        self.avgpool = paddle.nn.AvgPool2D(kernel_size=2, exclusive=False)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads,
                                        output_dim)
        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return paddle.nn.Sequential(*layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features**-0.5
            # torch.nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            # torch.nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            # torch.nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            # torch.nn.init.normal_(self.attnpool.c_proj.weight, std=std)
            self.attnpool.q_proj.weight = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(std=std))
            self.attnpool.k_proj.weight = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(std=std))
            self.attnpool.v_proj.weight = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(std=std))
            self.attnpool.c_proj.weight = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(std=std))
        for resnet_block in [
                self.layer1, self.layer2, self.layer3, self.layer4
        ]:
            for name, param in resnet_block.named_parameters():
                if name.endswith('bn3.weight'):
                    # torch.nn.init.zeros_(param)
                    param = paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Constant(0))

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.stop_gradient = not False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self)

    def set_grad_checkpointing(self, enable=True):
        pass

    def stem(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x

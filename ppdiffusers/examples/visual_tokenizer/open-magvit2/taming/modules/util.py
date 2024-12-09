import sys
sys.path.append(
    '/root/paddlejob/workspace/env_run/zhouhao21/code/pp-Open-MAGVIT2/utils')
import paddle_aux
import paddle


def count_params(model):
    total_params = sum(p.size for p in model.parameters())
    return total_params


class ActNorm(paddle.nn.Layer):

    def __init__(self, num_features, logdet=False, affine=True,
        allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = paddle.base.framework.EagerParamBase.from_tensor(tensor=
            paddle.zeros(shape=[1, num_features, 1, 1]))
        self.scale = paddle.base.framework.EagerParamBase.from_tensor(tensor
            =paddle.ones(shape=[1, num_features, 1, 1]))
        self.allow_reverse_init = allow_reverse_init
        self.register_buffer(name='initialized', tensor=paddle.to_tensor(
            data=0, dtype='uint8'))

    def initialize(self, input):
        with paddle.no_grad():
            flatten = input.transpose(perm=[1, 0, 2, 3]).contiguous().view(
                tuple(input.shape)[1], -1)
            mean = flatten.mean(axis=1).unsqueeze(axis=1).unsqueeze(axis=2
                ).unsqueeze(axis=3).transpose(perm=[1, 0, 2, 3])
            std = flatten.std(axis=1).unsqueeze(axis=1).unsqueeze(axis=2
                ).unsqueeze(axis=3).transpose(perm=[1, 0, 2, 3])
            paddle.assign(-mean, output=self.loc.data)
            paddle.assign(1 / (std + 1e-06), output=self.scale.data)

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(tuple(input.shape)) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False
        _, _, height, width = tuple(input.shape)
        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(value=1)
        h = self.scale * (input + self.loc)
        if squeeze:
            h = h.squeeze(axis=-1).squeeze(axis=-1)
        if self.logdet:
            log_abs = paddle.log(x=paddle.abs(x=self.scale))
            logdet = height * width * paddle.sum(x=log_abs)
            logdet = logdet * paddle.ones(shape=tuple(input.shape)[0]).to(input
                )
            return h, logdet
        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    'Initializing ActNorm in reverse direction is disabled by default. Use allow_reverse_init=True to enable.'
                    )
            else:
                self.initialize(output)
                self.initialized.fill_(value=1)
        if len(tuple(output.shape)) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False
        h = output / self.scale - self.loc
        if squeeze:
            h = h.squeeze(axis=-1).squeeze(axis=-1)
        return h


class AbstractEncoder(paddle.nn.Layer):

    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class Labelator(AbstractEncoder):
    """Net2Net Interface for Class-Conditional Model"""

    def __init__(self, n_classes, quantize_interface=True):
        super().__init__()
        self.n_classes = n_classes
        self.quantize_interface = quantize_interface

    def encode(self, c):
        c = c[:, None]
        if self.quantize_interface:
            return c, None, [None, None, c.astype(dtype='int64')]
        return c


class SOSProvider(AbstractEncoder):

    def __init__(self, sos_token, quantize_interface=True):
        super().__init__()
        self.sos_token = sos_token
        self.quantize_interface = quantize_interface

    def encode(self, x):
        c = paddle.ones(shape=[tuple(x.shape)[0], 1]) * self.sos_token
        c = c.astype(dtype='int64').to(x.place)
        if self.quantize_interface:
            return c, None, [None, None, c]
        return c


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.stop_gradient = not flag

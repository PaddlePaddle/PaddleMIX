import paddle
import math
from itertools import repeat
import collections.abc
import re
from collections import defaultdict
from itertools import chain
from typing import Any, Callable, Dict, Iterator, Tuple, Type, Union


def params_normal_(tensor, mean=0., std=1.):
    with paddle.no_grad():
        tensor.set_value(paddle.normal(mean=mean, std=std, shape=tensor.shape))
    return tensor


def trunc_normal_(tensor, mean=0., std=1., min=-2, max=2):
    with paddle.no_grad():
        normal = paddle.normal(mean=mean, std=std, shape=tensor.shape)
        trunc = paddle.clip(normal, min=min, max=max)
        tensor.set_value(trunc)
    return tensor


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def drop_path(x,
              drop_prob: float=0.0,
              training: bool=False,
              scale_by_keep: bool=True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    bern_0 = paddle.to_tensor(
        [keep_prob], dtype=paddle.float32) if not isinstance(
            keep_prob, paddle.Tensor) else keep_prob
    random_tensor = paddle.assign(
        paddle.bernoulli(
            paddle.broadcast_to(
                bern_0, paddle.empty(
                    shape=shape, dtype=x.dtype).shape)),
        paddle.empty(
            shape=shape, dtype=x.dtype))
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor.divide(keep_prob)
    return x * random_tensor


MATCH_PREV_GROUP = 99999,


def group_with_matcher(named_objects: Iterator[Tuple[str, Any]],
                       group_matcher: Union[Dict, Callable],
                       return_values: bool=False,
                       reverse: bool=False):
    if isinstance(group_matcher, dict):
        compiled = []
        for group_ordinal, (group_name,
                            mspec) in enumerate(group_matcher.items()):
            if mspec is None:
                continue
            if isinstance(mspec, (tuple, list)):
                for sspec in mspec:
                    compiled += [(re.compile(sspec[0]), (group_ordinal, ),
                                  sspec[1])]
            else:
                compiled += [(re.compile(mspec), (group_ordinal, ), None)]
        group_matcher = compiled

    def _get_grouping(name):
        if isinstance(group_matcher, (list, tuple)):
            for match_fn, prefix, suffix in group_matcher:
                r = match_fn.match(name)
                if r:
                    parts = prefix, r.groups(), suffix
                    return tuple(
                        map(float, chain.from_iterable(filter(None, parts))))
            return float('inf'),
        else:
            ord = group_matcher(name)
            if not isinstance(ord, collections.abc.Iterable):
                return ord,
            return tuple(ord)

    grouping = defaultdict(list)
    for k, v in named_objects:
        grouping[_get_grouping(k)].append(v if return_values else k)
    layer_id_to_param = defaultdict(list)
    lid = -1
    for k in sorted(filter(lambda x: x is not None, grouping.keys())):
        if lid < 0 or k[-1] != MATCH_PREV_GROUP[0]:
            lid += 1
        layer_id_to_param[lid].extend(grouping[k])
    if reverse:
        assert not return_values, 'reverse mapping only sensible for name output'
        param_to_layer_id = {}
        for lid, lm in layer_id_to_param.items():
            for n in lm:
                param_to_layer_id[n] = lid
        return param_to_layer_id
    return layer_id_to_param


def group_parameters(module: paddle.nn.Layer,
                     group_matcher,
                     return_values: bool=False,
                     reverse: bool=False):
    return group_with_matcher(
        module.named_parameters(),
        group_matcher,
        return_values=return_values,
        reverse=reverse)


def named_modules_with_params(module: paddle.nn.Layer,
                              name: str='',
                              depth_first: bool=True,
                              include_root: bool=False):
    if module._parameters and not depth_first and include_root:
        yield name, module
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        yield from named_modules_with_params(
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True)
    if module._parameters and depth_first and include_root:
        yield name, module


def group_modules(module: paddle.nn.Layer,
                  group_matcher,
                  return_values: bool=False,
                  reverse: bool=False):
    return group_with_matcher(
        named_modules_with_params(module),
        group_matcher,
        return_values=return_values,
        reverse=reverse)


class LabelSmoothingCrossEntropy(paddle.nn.Layer):
    """ NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
        logprobs = paddle.nn.functional.log_softmax(x=x, axis=-1)
        nll_loss = -logprobs.take_along_axis(
            axis=-1, indices=target.unsqueeze(axis=1))
        nll_loss = nll_loss.squeeze(axis=1)
        smooth_loss = -logprobs.mean(axis=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

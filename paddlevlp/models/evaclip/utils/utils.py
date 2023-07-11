import datetime
import time
import argparse
import paddle
import numpy as np
import paddle.nn as nn
import math
import os
import sys
from PIL import ImageFilter, ImageOps
import random
from collections import defaultdict, deque
from paddle.optimizer.lr import LRScheduler
import pprint
from paddle.distributed import fleet


@paddle.no_grad()
def clip_scale(model):
    if (fleet.get_hybrid_communicate_group().get_model_parallel_world_size() >
            1) or isinstance(model, paddle.DataParallel):
        # model._layers.logit_scale.clip_(0, 4.6052)
        share_buffer = model._layers.logit_scale.clip(0, 4.6052)
        model._layers.logit_scale.copy_(share_buffer, True)
    else:
        # model.logit_scale.clip_(0, 4.6052)
        share_buffer = model.logit_scale.clip(0, 4.6052)
        model.logit_scale.copy_(share_buffer, True)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        NOW = datetime.datetime.now().strftime('[ %Y-%m-%d/%H:%M:%S ]')
        entries = [NOW] + entries
        pprint.pprint('##'.join(entries), width=1000000000)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with paddle.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # correct = pred.eq(target.reshape([1, -1]).expand_as(pred))
        correct = paddle.equal(pred, target.reshape([1, -1]).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape([-1]).astype(paddle.float32).sum(
                0, keepdim=True)
            # res.append(correct_k.mul_(100.0 / batch_size))
            res.append(
                paddle.multiply(
                    paddle.to_tensor(correct_k),
                    paddle.to_tensor(paddle.to_tensor(100.0 / batch_size))))
        return res


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1D, nn.BatchNorm2D, nn.BatchNorm3D,
                nn.SyncBatchNorm)
    for name, module in model.named_sublayers():
        if isinstance(module, bn_types):
            return True
    return False


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if param.stop_gradient:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{
        'params': regularized
    }, {
        'params': not_regularized,
        'weight_decay': 0.
    }]


class ClipDecay(LRScheduler):
    def __init__(self, lr_list, learning_rate=1, last_epoch=-1, verbose=False):

        self.lr_list = lr_list
        super().__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < len(self.lr_list):
            return self.lr_list[self.last_epoch]
        else:
            return self.lr_list[-1]


def cosine_scheduler(base_value,
                     final_value,
                     epochs,
                     niter_per_ep,
                     warmup_epochs=0,
                     warmup_iters=0,
                     start_warmup_value=0):
    warmup_schedule = np.array([])
    if warmup_epochs > 0:
        warmup_iters = int(warmup_epochs * niter_per_ep)
    warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array([final_value + 0.5 * (base_value - final_value) * (1 + \
        math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    scheduler = ClipDecay(schedule)
    return scheduler


def linear_scheduler(base_value,
                     final_value,
                     epochs,
                     niter_per_ep,
                     warmup_epochs=0,
                     warmup_iters=0,
                     start_warmup_value=0):
    warmup_schedule = np.array([])
    if warmup_epochs > 0:
        warmup_iters = int(warmup_epochs * niter_per_ep)
    warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    #  delta = (base_value - final_value) / (niter_per_ep * epochs - warmup_iters)
    #  schedule = np.array([base_value - delta * i if base_value - delta * i > 0.0 else 0.0 for i in iters])
    schedule = np.linspace(base_value, final_value, iters.shape[0])

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(
        schedule
    ) == epochs * niter_per_ep, "expected schedule steps {} == epochs({}) x niter_per_ep({})={}, warmup_iters:{}".format(
        len(schedule), epochs, niter_per_ep, epochs * niter_per_ep,
        warmup_iters)
    scheduler = ClipDecay(schedule)
    return scheduler


def exp_scheduler(base_value,
                  final_value,
                  epochs,
                  niter_per_ep,
                  warmup_epochs=0,
                  warmup_iters=0,
                  start_warmup_value=0):
    warmup_schedule = np.array([])
    if warmup_epochs > 0:
        warmup_iters = int(warmup_epochs * niter_per_ep)
    warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    T2 = 0
    T1 = 9.5e-5
    B = (base_value - final_value) / (1 - math.exp(-1 * (T1 + T2) * iters[-1]))
    b = (final_value - base_value * math.exp(-1 * (T1 + T2) * iters[-1])) / (
        1 - math.exp(-1 * (T1 + T2) * iters[-1]))
    schedule = np.array([B * math.exp(-1 * (T1 + T2) * i) + b for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    scheduler = ClipDecay(schedule)
    return scheduler


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.norm(2)
            norms.append(param_norm.item())
            clip_coef = paddle.to_tensor(clip / (param_norm + 1e-6))
            if clip_coef < 1:
                # p.grad.data.mul_(clip_coef)
                p.grad.set_value(paddle.multiply(p.grad, clip_coef))
    return norms


def clip_grad_norm_(parameters,
                    max_norm,
                    norm_type,
                    error_if_nonfinite: bool=False):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        print("skil clip_grad_norm")
        return paddle.to_tensor([0.])
    if norm_type == float("inf"):
        norms = [g.detach().abs().max() for g in grads]
        total_norm = norms[0] if len(norms) == 1 else paddle.max(
            paddle.stack(norms))
    else:
        total_norm = paddle.norm(
            paddle.stack([paddle.norm(g.detach(), norm_type) for g in grads]),
            norm_type)
    if error_if_nonfinite and paddle.logical_or(total_norm.isnan(),
                                                total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = paddle.clip(clip_coef, max=1.0)
    for g in grads:
        clipg = paddle.multiply(g, clip_coef_clamped)
        g.set_value(clipg)
    return total_norm


def clip_grad_norm(model,
                   max_norm,
                   norm_type=2.0,
                   error_if_nonfinite: bool=False):
    with paddle.no_grad():
        parameters = model.parameters()
        return clip_grad_norm_(parameters, max_norm, norm_type,
                               error_if_nonfinite)


def cancel_gradients_last_layer(epoch_float, model, freeze_last_layer):
    if epoch_float >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = paddle.load(ckp_path)

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key])
                print("=> loaded {} from checkpoint '{}' with msg {}".format(
                    key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded {} from checkpoint '{}'".format(key,
                                                                     ckp_path))
                except ValueError:
                    print("=> failed to load {} from checkpoint '{}'".format(
                        key, ckp_path))
        else:
            print("=> failed to load {} from checkpoint '{}'".format(key,
                                                                     ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min,
                                                           self.radius_max)))


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def load_pretrained_weights(model, pretrained_weights, checkpoint_key,
                            model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = paddle.load(pretrained_weights)
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if 'model' in state_dict:
            state_dict = state_dict['model']
        # remove `module.` prefix
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {
            k.replace("backbone.", ""): v
            for k, v in state_dict.items()
        }
        # remove `encoder_q. & encoder_k.` prefix induced by moco
        state_dict = {
            k.replace("encoder_q.", ""): v
            for k, v in state_dict.items() if 'encoder_k' not in k
        }
        msg = model.set_state_dict(state_dict)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(
            pretrained_weights, msg))
    else:
        print(
            "There is no reference weights available for this model => We use random weights."
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, paddle.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if 'gpu' in paddle.device.get_device():
            log_msg = self.delimiter.join([
                header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}',
                'time: {time}', 'data: {data}', 'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}',
                'time: {time}', 'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if 'gpu' in paddle.device.get_device():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=paddle.device.max_memory_allocated() / MB))
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

from cgi import print_arguments
import paddle
""" PyTorch Lamb optimizer w/ behaviour similar to NVIDIA FusedLamb
This optimizer code was adapted from the following (starting with latest)
* https://github.com/HabanaAI/Model-References/blob/2b435114fe8e31f159b1d3063b8280ae37af7423/PyTorch/nlp/bert/pretraining/lamb.py
* https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py
* https://github.com/cybertronai/pytorch-lamb
Use FusedLamb if you can (GPU). The reason for including this variant of Lamb is to have a version that is
similar in behaviour to APEX FusedLamb if you aren't using NVIDIA GPUs or cannot install/use APEX.
In addition to some cleanup, this Lamb impl has been modified to support PyTorch XLA and has been tested on TPU.
Original copyrights for above sources are below.
Modifications Copyright 2021 Ross Wightman
"""
import math
from typing import Dict
from .optimizer import Optimizer


class Lamb(Optimizer):
    """Implements a pure pytorch variant of FuseLAMB (NvLamb variant) optimizer from apex.optimizers.FusedLAMB
    reference: https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py
    LAMB was proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        epsilon (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_averaging (bool, optional): whether apply (1-beta2) to grad when
            calculating running averages of gradient. (default: True)
        max_grad_norm (float, optional): value used to clip global grad norm (default: 1.0)
        trust_clip (bool): enable LAMBC trust ratio clipping (default: False)
        always_adapt (boolean, optional): Apply adaptive learning rate to 0.0
            weight decay parameter (default: False)
    .. _Large Batch Optimization for Deep Learning - Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self,
                 parameters,
                 lr=0.001,
                 bias_correction=True,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-06,
                 weight_decay=0.01,
                 grad_averaging=True,
                 max_grad_norm=1.0,
                 trust_clip=False,
                 always_adapt=False):
        betas = (beta1, beta2)
        self.defaults = dict(
            learning_rate=lr,
            bias_correction=bias_correction,
            betas=betas,
            epsilon=epsilon,
            weight_decay=weight_decay,
            grad_averaging=grad_averaging,
            max_grad_norm=max_grad_norm,
            trust_clip=trust_clip,
            always_adapt=always_adapt)
        optim_defaults = dict(
            learning_rate=lr, epsilon=epsilon, weight_decay=weight_decay)
        # self.state = Dict[paddle.Tensor, Dict[str, paddle.Tensor]]
        self.state = {}
        super(Lamb, self).__init__(parameters, self.defaults)

    @paddle.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with paddle.set_grad_enabled(True):
                loss = closure()
        one_tensor = paddle.to_tensor(data=1.0)
        global_grad_norm = paddle.zeros(shape=[1])
        for group in self._param_groups:
            for p in group['params']:
                if p._grad_ivar() is None:
                    continue
                grad = p._grad_ivar()
                """Class Method: *.add_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                global_grad_norm = global_grad_norm.add(grad.pow(y=2).sum())
        global_grad_norm = paddle.sqrt(x=global_grad_norm)
        max_grad_norm = paddle.to_tensor(data=self.defaults['max_grad_norm'])
        clip_global_grad_norm = paddle.where(
            condition=global_grad_norm > max_grad_norm,
            x=global_grad_norm / max_grad_norm,
            y=one_tensor)
        for group in self._param_groups:
            bias_correction = 1 if self.defaults['bias_correction'] else 0
            beta1, beta2 = self.defaults['betas']
            grad_averaging = 1 if self.defaults['grad_averaging'] else 0
            beta3 = 1 - beta1 if grad_averaging else 1.0
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
            if bias_correction:
                bias_correction1 = 1 - beta1**group['step']
                bias_correction2 = 1 - beta2**group['step']
            else:
                bias_correction1, bias_correction2 = 1.0, 1.0
            for idx, p in enumerate(group['params']):
                if p._grad_ivar() is None:
                    continue
                grad = p._grad_ivar().divide(clip_global_grad_norm)
                if p not in self.state:
                    self.state[p] = {}
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = paddle.zeros_like(x=p)
                    state['exp_avg_sq'] = paddle.zeros_like(x=p)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                """Class Method: *.add_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                exp_avg = exp_avg.scale_(scale=beta1).add(grad * beta3)
                """Class Method: *.addcmul_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                # exp_avg_sq.scale_(scale=beta2).addcmul_(grad, grad, value=1 -
                #     beta2)
                exp_avg_sq.scale_(scale=beta2)
                exp_avg_sq = exp_avg_sq + grad * grad * (1 - beta2)
                """Class Method: *.add_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                         ).add(paddle.to_tensor(self.defaults['epsilon']))
                """Class Method: *.div_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                update = (exp_avg / bias_correction1).divide(denom)
                weight_decay = group['weight_decay']
                if weight_decay != 0:
                    """Class Method: *.add_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                    update = update.add(p * weight_decay)
                if weight_decay != 0 or self.defaults['always_adapt']:
                    w_norm = p.norm(p=2.0)
                    g_norm = update.norm(p=2.0)
                    trust_ratio = paddle.where(
                        condition=w_norm > 0,
                        x=paddle.where(
                            condition=g_norm > 0,
                            x=w_norm / g_norm,
                            y=one_tensor),
                        y=one_tensor)
                    if self.defaults['trust_clip']:
                        trust_ratio = paddle.minimum(
                            x=trust_ratio, y=one_tensor)
                    update.scale_(scale=trust_ratio)
                """Class Method: *.add_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                group['params'][idx].copy_(
                    p.add(update * (-group['learning_rate'])), False)
        return

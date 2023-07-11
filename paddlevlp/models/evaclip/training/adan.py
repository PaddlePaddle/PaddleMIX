import paddle
""" Adan Optimizer
Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models[J]. arXiv preprint arXiv:2208.06677, 2022.
    https://arxiv.org/abs/2208.06677
Implementation adapted from https://github.com/sail-sg/Adan
"""
import math


class Adan(paddle.optimizer.Optimizer):
    """
    Implements a pytorch variant of Adan
    Adan was proposed in
    Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models[J]. arXiv preprint arXiv:2208.06677, 2022.
    https://arxiv.org/abs/2208.06677
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float, flot], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.98, 0.92, 0.99))
        epsilon (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay (L2 penalty) (default: 0)
        no_prox (bool): how to perform the decoupled weight decay (default: False)
    """

    def __init__(self,
                 params,
                 lr=0.001,
                 betas=(0.98, 0.92, 0.99),
                 epsilon=1e-08,
                 weight_decay=0.0,
                 no_prox=False):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= epsilon:
            raise ValueError('Invalid epsilon value: {}'.format(epsilon))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(
                betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError('Invalid beta parameter at index 2: {}'.format(
                betas[2]))
        defaults = dict(
            lr=lr,
            betas=betas,
            epsilon=epsilon,
            weight_decay=weight_decay,
            no_prox=no_prox)
        super(Adan, self).__init__(params, defaults)

    @paddle.no_grad()
    def restart_opt(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if not p.stop_gradient:
                    state = self.state[p]
                    state['exp_avg'] = paddle.zeros_like(x=p)
                    state['exp_avg_sq'] = paddle.zeros_like(x=p)
                    state['exp_avg_diff'] = paddle.zeros_like(x=p)

    @paddle.no_grad()
    def step(self, closure=None):
        """ Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with paddle.set_grad_enabled(True):
                loss = closure()
        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
            bias_correction1 = 1.0 - beta1**group['step']
            bias_correction2 = 1.0 - beta2**group['step']
            bias_correction3 = 1.0 - beta3**group['step']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = paddle.zeros_like(x=p)
                    state['exp_avg_diff'] = paddle.zeros_like(x=p)
                    state['exp_avg_sq'] = paddle.zeros_like(x=p)
                    state['pre_grad'] = grad.clone()
                exp_avg, exp_avg_sq, exp_avg_diff = state['exp_avg'], state[
                    'exp_avg_diff'], state['exp_avg_sq']
                grad_diff = grad - state['pre_grad']
                exp_avg.lerp_(y=grad, weight=1.0 - beta1)
                exp_avg_diff.lerp_(y=grad_diff, weight=1.0 - beta2)
                update = grad + beta2 * grad_diff
                """Class Method: *.addcmul_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                # exp_avg_sq.scale_(scale=beta3).addcmul_(update, update,
                #     value=1.0 - beta3)
                exp_avg_sq.scale_(scale=beta3)
                exp_avg_sq = exp_avg_sq + update * update * (1.0 - beta3)
                """Class Method: *.add_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                denom = (exp_avg_sq.sqrt() /
                         math.sqrt(bias_correction3)).add(group['epsilon'])
                """Class Method: *.div_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                update = (exp_avg / bias_correction1 + beta2 * exp_avg_diff /
                          bias_correction2).divide(denom)
                if group['no_prox']:
                    p.data.mul_(1 - group['learning_rate'] * group[
                        'weight_decay'])
                    """Class Method: *.add_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                    p = p.add(update, alpha=-group['learning_rate'])
                else:
                    """Class Method: *.add_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                    p = p.add(update, alpha=-group['learning_rate'])
                    p.data.div_(1 + group['learning_rate'] * group[
                        'weight_decay'])
                paddle.assign(grad, output=state['pre_grad'])
        return

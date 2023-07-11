import paddle
"""PyTorch implementation of the Lion optimizer."""


class Lion(paddle.optimizer.Optimizer):
    """Implements Lion algorithm."""

    def __init__(self, params, lr=0.0001, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialize the hyperparameters.
    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
      lr (float, optional): learning rate (default: 1e-4)
      betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.99))
      weight_decay (float, optional): weight decay coefficient (default: 0)
    """
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(
                betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @paddle.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    Returns:
      the loss.
    """
        loss = None
        if closure is not None:
            with paddle.set_grad_enabled(True):
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.mul_(1 - group['learning_rate'] * group['weight_decay'])
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = paddle.zeros_like(x=p)
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                update = exp_avg * beta1 + grad * (1 - beta1)
                """Class Method: *.add_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                p = p.add(paddle.sign(x=update), alpha=-group['learning_rate'])
                """Class Method: *.add_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                exp_avg = exp_avg.scale_(scale=beta2).add(grad,
                                                          alpha=1 - beta2)
        return loss

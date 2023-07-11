import paddle


class AnyPrecisionAdamW(paddle.optimizer.Optimizer):
    def __init__(self,
                 params,
                 lr=0.001,
                 betas=(0.9, 0.999),
                 epsilon=1e-08,
                 weight_decay=0.0,
                 use_kahan_summation=True,
                 momentum_dtype='bfloat16',
                 variance_dtype='bfloat16',
                 compensation_buffer_dtype='bfloat16'):
        """
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
            epsilon (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)
            weight_decay (float, optional): weight decay coefficient (default: 1e-2)
            # Any Precision specific
            use_kahan_summation = creates auxiliary buffer to ensure high precision
            model param updates (default: False)
            momentum_dtype = dtype for momentum  (default: BFloat32)
            variance_dtype = dtype for uncentered variance (default: BFloat16)
            compensation_buffer_dtype  = dtype for Kahan summation
                                         buffer (default: BFloat16). Only used if
                                         ``use_kahan_summation=True``.
            # Usage
            This optimizer implements optimizer states, and Kahan summation
            for high precision updates, all in user controlled dtypes.
            Defaults are variance in BF16, Momentum in FP32.
            This can be run in FSDP mixed precision, amp, or full precision,
            depending on what training pipeline you wish to work with.
            Setting to use_kahan_summation = False, and changing momentum and
            variance dtypes to FP32, reverts this to a standard AdamW optimizer.
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            epsilon=epsilon,
            weight_decay=weight_decay,
            use_kahan_summation=use_kahan_summation,
            momentum_dtype=momentum_dtype,
            variance_dtype=variance_dtype,
            compensation_buffer_dtype=compensation_buffer_dtype)
        super().__init__(params, defaults)

    @paddle.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if closure is not None:
            with paddle.set_grad_enabled(True):
                closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['learning_rate']
            weight_decay = group['weight_decay']
            epsilon = group['epsilon']
            use_kahan_summation = group['use_kahan_summation']
            momentum_dtype = group['momentum_dtype']
            variance_dtype = group['variance_dtype']
            compensation_buffer_dtype = group['compensation_buffer_dtype']
            for p in group['params']:
                if p.grad is None:
                    continue
                """Tensor Attribute: torch.Tensor.is_sparse, not convert, please check whether it is torch.Tensor.* and convert manually"""
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = paddle.to_tensor(data=0.0)
                    state['exp_avg'] = paddle.zeros_like(
                        x=p).astype(momentum_dtype)
                    state['exp_avg_sq'] = paddle.zeros_like(
                        x=p).astype(variance_dtype)
                    if use_kahan_summation:
                        state['compensation'] = paddle.zeros_like(
                            x=p).astype(compensation_buffer_dtype)
                state['step'] += 1
                step = state['step']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                grad = p.grad
                if weight_decay:
                    p.data.mul_(1 - lr * weight_decay)
                """Class Method: *.add_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                exp_avg = exp_avg.scale_(scale=beta1).add(grad,
                                                          alpha=1 - beta1)
                """Class Method: *.addcmul_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                # exp_avg_sq.scale_(scale=beta2).addcmul_(grad, grad, value=1 -
                #     beta2)
                exp_avg_sq.scale_(scale=beta2)
                exp_avg_sq = exp_avg_sq + grad * grad * (1 - beta2)
                bias_correction1 = 1 - beta1**step
                step_size = lr / bias_correction1
                denom_correction = (1 - beta2**step)**0.5
                """Class Method: *.add_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                centered_variance = (exp_avg_sq.sqrt() /
                                     denom_correction).add(epsilon, alpha=1)
                if use_kahan_summation:
                    compensation = state['compensation']
                    """Class Method: *.addcdiv_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                    # compensation.addcdiv_(exp_avg, centered_variance, value
                    #     =-step_size)
                    compensation = compensation - (step_size * exp_avg
                                                   ) / centered_variance
                    temp_buffer = p.detach().clone()
                    p.data.add_(compensation)
                    """Class Method: *.sub_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                    """Class Method: *.add_, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                    compensation = compensation.add(temp_buffer.sub_(p.data))
                else:
                    p.data = p.data - (step_size * exp_avg) / centered_variance

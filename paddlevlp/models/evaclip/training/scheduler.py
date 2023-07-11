import numpy as np


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer._param_groups:
        param_group['learning_rate'] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def warmup_cosine_lr(optimizer, args, steps):
    def _lr_adjuster(step):
        for param_group in optimizer._param_groups:
            if param_group['group'] == 'text':
                base_lr = args.text_lr if args.text_lr is not None else args.lr
            elif param_group['group'] == 'visual':
                base_lr = (args.visual_lr
                           if args.visual_lr is not None else args.lr)
            else:
                base_lr = args.lr
            if step < args.warmup:
                lr = _warmup_lr(base_lr, args.warmup, step)
            else:
                e = step - args.warmup
                es = steps - args.warmup
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            scale = param_group.get('lr_scale', 1.0)
            # optimizer.set_lr(0.5 * (1 + np.cos(np.pi * e / es))) 
            param_group['learning_rate'] = lr * scale
            for param in param_group['params']:
                param.optimize_attr['learning_rate'] = scale * lr
            # print("group:{}, lr:{}".format(param_group['group'], param_group['learning_rate']))
        return lr

    return _lr_adjuster


def warmup_step_lr(optimizer, args, decay_t=500, decay_rate=0.8):
    def _lr_adjuster(step):
        for param_group in optimizer._param_groups:
            if param_group['group'] == 'text':
                base_lr = args.text_lr
            elif param_group['group'] == 'visual':
                base_lr = args.visual_lr
            else:
                base_lr = args.lr
            if step < args.warmup:
                lr = _warmup_lr(base_lr, args.warmup, step)
            else:
                e = step - args.warmup
                lr = base_lr * decay_rate**(e // decay_t)
            scale = param_group.get('lr_scale', 1.0)
            param_group['learning_rate'] = scale * lr
        return lr

    return _lr_adjuster

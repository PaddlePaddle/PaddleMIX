import sys
from itertools import repeat
import collections.abc
import logging
import math
import numpy as np
import scipy
import paddle
import paddle.distributed as dist


def resize_clip_pos_embed(state_dict,
                          model,
                          interpolation: str='bicubic',
                          seq_dim=1):
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return
    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[
            extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))
    logging.info('Resizing position embedding grid-size from %s to %s',
                 old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape((1, old_grid_size[0], old_grid_size[1],
                                       -1)).transpose(perm=[0, 3, 1, 2])
    pos_emb_img = paddle.nn.functional.interpolate(
        x=pos_emb_img, size=grid_size, mode=interpolation, align_corners=True)
    pos_emb_img = pos_emb_img.transpose(perm=[0, 2, 3, 1]).reshape(
        (1, grid_size[0] * grid_size[1], -1))[0]
    if pos_emb_tok is not None:
        new_pos_embed = paddle.concat(x=[pos_emb_tok, pos_emb_img], axis=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed


def resize_visual_pos_embed(state_dict,
                            model,
                            interpolation: str='bicubic',
                            seq_dim=1):
    old_pos_embed = state_dict.get('positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return
    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[
            extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))
    logging.info('Resizing position embedding grid-size from %s to %s',
                 old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape((1, old_grid_size[0], old_grid_size[1],
                                       -1)).transpose(perm=[0, 3, 1, 2])
    pos_emb_img = paddle.nn.functional.interpolate(
        x=pos_emb_img, size=grid_size, mode=interpolation, align_corners=True)
    pos_emb_img = pos_emb_img.transpose(perm=[0, 2, 3, 1]).reshape(
        (1, grid_size[0] * grid_size[1], -1))[0]
    if pos_emb_tok is not None:
        new_pos_embed = paddle.concat(x=[pos_emb_tok, pos_emb_img], axis=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['positional_embedding'] = new_pos_embed


def resize_evaclip_pos_embed(state_dict,
                             model,
                             interpolation: str='bicubic',
                             seq_dim=1):
    all_keys = list(state_dict.keys())
    if 'visual.pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['visual.pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.visual.patch_embed.num_patches
        num_extra_tokens = model.visual.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens)**
                        0.5)
        new_size = int(num_patches**0.5)
        if orig_size != new_size:
            print('Position interpolate from %dx%d to %dx%d' %
                  (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                (-1, orig_size, orig_size,
                 embedding_size)).transpose(perm=[0, 3, 1, 2])
            pos_tokens = paddle.nn.functional.interpolate(
                x=pos_tokens,
                size=(new_size, new_size),
                mode='bicubic',
                align_corners=False)
            pos_tokens = pos_tokens.transpose(perm=[0, 2, 3, 1]).flatten(
                start_axis=1, stop_axis=2)
            new_pos_embed = paddle.concat(x=(extra_tokens, pos_tokens), axis=1)
            state_dict['visual.pos_embed'] = new_pos_embed
            patch_embed_proj = state_dict['visual.patch_embed.proj.weight']
            patch_size = model.visual.patch_embed.patch_size
            state_dict[
                'visual.patch_embed.proj.weight'] = paddle.nn.functional.interpolate(
                    x=patch_embed_proj.astype(dtype='float32'),
                    size=patch_size,
                    mode='bicubic',
                    align_corners=False)


def resize_eva_pos_embed(state_dict,
                         model,
                         interpolation: str='bicubic',
                         seq_dim=1):
    all_keys = list(state_dict.keys())
    if 'pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.visual.patch_embed.num_patches
        num_extra_tokens = model.visual.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens)**
                        0.5)
        new_size = int(num_patches**0.5)
        if orig_size != new_size:
            print('Position interpolate from %dx%d to %dx%d' %
                  (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                (-1, orig_size, orig_size,
                 embedding_size)).transpose(perm=[0, 3, 1, 2])
            pos_tokens = paddle.nn.functional.interpolate(
                x=pos_tokens,
                size=(new_size, new_size),
                mode='bicubic',
                align_corners=False)
            pos_tokens = pos_tokens.transpose(perm=[0, 2, 3, 1]).flatten(
                start_axis=1, stop_axis=2)
            new_pos_embed = paddle.concat(x=(extra_tokens, pos_tokens), axis=1)
            state_dict['pos_embed'] = new_pos_embed
            patch_embed_proj = state_dict['patch_embed.proj.weight']
            patch_size = model.visual.patch_embed.patch_size
            state_dict[
                'patch_embed.proj.weight'] = paddle.nn.functional.interpolate(
                    x=patch_embed_proj.astype(dtype='float32'),
                    size=patch_size,
                    mode='bicubic',
                    align_corners=False)


def resize_rel_pos_embed(state_dict,
                         model,
                         interpolation: str='bicubic',
                         seq_dim=1):
    all_keys = list(state_dict.keys())
    for key in all_keys:
        if 'relative_position_index' in key:
            state_dict.pop(key)
        if 'relative_position_bias_table' in key:
            rel_pos_bias = state_dict[key]
            src_num_pos, num_attn_heads = rel_pos_bias.shape
            dst_num_pos, _ = model.visual.state_dict()[key].shape
            dst_patch_shape = model.visual.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (
                dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens)**0.5)
            dst_size = int((dst_num_pos - num_extra_tokens)**0.5)
            if src_size != dst_size:
                print('Position interpolate for %s from %dx%d to %dx%d' %
                      (key, src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r**n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-06:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q
                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q**(i + 1)
                r_ids = [(-_) for _ in reversed(dis)]
                x = r_ids + [0] + dis
                y = r_ids + [0] + dis
                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)
                print('Original positions = %s' % str(x))
                print('Target positions = %s' % str(dx))
                all_rel_pos_bias = []
                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, (i)].reshape(
                        (src_size, src_size)).astype(dtype='float32').numpy()
                    #use scipy for numpy input
                    f = scipy.interpolate.interp2d(x, y, z, kind='cubic')
                    if isinstance(rel_pos_bias.place, paddle.dtype):
                        dtype = rel_pos_bias.place
                    elif isinstance(rel_pos_bias.place,
                                    str) and rel_pos_bias.place not in [
                                        'cpu', 'cuda', 'ipu', 'xpu'
                                    ]:
                        dtype = rel_pos_bias.place
                    elif isinstance(rel_pos_bias.place, paddle.Tensor):
                        dtype = rel_pos_bias.place.dtype
                    else:
                        dtype = paddle.float32
                    all_rel_pos_bias.append(
                        paddle.to_tensor(
                            x=f(dx, dy), dtype='float32').reshape((-1, 1))
                        .cast(dtype))
                rel_pos_bias = paddle.concat(x=all_rel_pos_bias, axis=-1)
                new_rel_pos_bias = paddle.concat(
                    x=(rel_pos_bias, extra_tokens), axis=0)
                state_dict[key] = new_rel_pos_bias
    if 'pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.visual.patch_embed.num_patches
        num_extra_tokens = model.visual.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens)**
                        0.5)
        new_size = int(num_patches**0.5)
        if orig_size != new_size:
            print('Position interpolate from %dx%d to %dx%d' %
                  (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                (-1, orig_size, orig_size,
                 embedding_size)).transpose(perm=[0, 3, 1, 2])
            pos_tokens = paddle.nn.functional.interpolate(
                x=pos_tokens,
                size=(new_size, new_size),
                mode='bicubic',
                align_corners=False)
            pos_tokens = pos_tokens.transpose(perm=[0, 2, 3, 1]).flatten(
                start_axis=1, stop_axis=2)
            new_pos_embed = paddle.concat(x=(extra_tokens, pos_tokens), axis=1)
            state_dict['pos_embed'] = new_pos_embed
            patch_embed_proj = state_dict['patch_embed.proj.weight']
            patch_size = model.visual.patch_embed.patch_size
            state_dict[
                'patch_embed.proj.weight'] = paddle.nn.functional.interpolate(
                    x=patch_embed_proj.astype(dtype='float32'),
                    size=patch_size,
                    mode='bicubic',
                    align_corners=False)


def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (paddle.nn.BatchNorm2d,
                                        paddle.nn.SyncBatchNorm)):
        # res = torchvision.ops.misc.FrozenBatchNorm2d(module.num_features)
        res.is_test = True
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.epsilon = module.epsilon
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join(
                [name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match,
                                             full_child_name)
            if new_child is not child:
                res.add_sublayer(child_name, new_child)
    return res


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def clip_grad_norm_(
        parameters, max_norm, norm_type,
        error_if_nonfinite: bool = False):
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
        return paddle.to_tensor([0.])
    if norm_type == float("inf"):
        norms = [g.detach().abs().max() for g in grads]
        total_norm = norms[0] if len(norms) == 1 else paddle.max(paddle.stack(norms))
    else:
        total_norm = paddle.norm(paddle.stack([paddle.norm(g.detach(), norm_type) for g in grads]), norm_type)
    if error_if_nonfinite and paddle.logical_or(total_norm.isnan(), total_norm.isinf()):
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
    total_norm_clip = paddle.norm(paddle.stack([paddle.norm(g.detach(), norm_type) for g in grads]), norm_type)
    return total_norm_clip


def clip_grad_norm(
        model, max_norm, norm_type=2.0,
        error_if_nonfinite: bool = False):
    parameters = model.parameters()
    return clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite)

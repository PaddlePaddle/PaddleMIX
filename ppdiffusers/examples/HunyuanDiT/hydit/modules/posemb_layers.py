import paddle
import numpy as np
from typing import Union

def TypePromote(x, y):
    TYPE_PROMOTE_DICT ={
        'INT16FP16':'float16',
        'INT16FP32':'float32',
        'INT16FP64':'float64',

        'INT32FP16':'float32',
        'INT32FP32':'float32',
        'INT32FP64':'float64',

        'INT64FP16':'float64',
        'INT64FP32':'float64',
        'INT64FP64':'float64',
    }
    if x.dtype.name + y.dtype.name in TYPE_PROMOTE_DICT:
        promote_type = TYPE_PROMOTE_DICT[x.dtype.name + y.dtype.name]
    elif y.dtype.name + x.dtype.name in TYPE_PROMOTE_DICT:
        promote_type = TYPE_PROMOTE_DICT[y.dtype.name + x.dtype.name]
    else:
        return x,y
    return x.astype(promote_type),y.astype(promote_type)

def _to_tuple(x):
    if isinstance(x, int):
        return x, x
    else:
        return x


def get_fill_resize_and_crop(src, tgt):    # src 来源的分辨率   tgt base 分辨率
    th, tw = _to_tuple(tgt)
    h, w = _to_tuple(src)

    tr = th / tw        # base 分辨率
    r = h / w           # 目标分辨率

    # resize
    if r > tr:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))    # 根据base分辨率，将目标分辨率resize下来

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def get_meshgrid(start, *args):
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start)
        start = (0, 0)
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start)
        stop = _to_tuple(args[0])
        num = (stop[0] - start[0], stop[1] - start[1])
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start)       # 左上角   eg: 12,0
        stop = _to_tuple(args[0])      # 右下角   eg: 20,32
        num = _to_tuple(args[1])       # 目标大小  eg: 32,124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    grid_h = np.linspace(start[0], stop[0], num[0], endpoint=False, dtype=np.float32) # 12-20 中间差值32份   0-32 中间差值124份
    grid_w = np.linspace(start[1], stop[1], num[1], endpoint=False, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)   # [2, W, H]
    return grid

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, start, *args, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = get_meshgrid(start, *args)   # [2, H, w]
    # grid_h = np.arange(grid_size, dtype=np.float32)
    # grid_w = np.arange(grid_size, dtype=np.float32)
    # grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    # grid = np.stack(grid, axis=0)   # [2, W, H]

    grid = grid.reshape([2, 1, *grid.shape[1:]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)    # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (W,H)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)   # (M, D/2)
    emb_cos = np.cos(out)   # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                   Rotary Positional Embedding Functions                       #
#################################################################################
# https://github.com/facebookresearch/llama/blob/main/llama/model.py#L443

def get_2d_rotary_pos_embed(embed_dim, start, *args, use_real=True):
    """
    This is a 2d version of precompute_freqs_cis, which is a RoPE for image tokens with 2d structure.

    Parameters
    ----------
    embed_dim: int
        embedding dimension size
    start: int or tuple of int
        If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop, step is 1;
        If len(args) == 2, start is start, args[0] is stop, args[1] is num.
    use_real: bool
        If True, return real part and imaginary part separately. Otherwise, return complex numbers.

    Returns
    -------
    pos_embed: torch.Tensor
        [HW, D/2]
    """
    grid = get_meshgrid(start, *args)   # [2, H, w]
    grid = grid.reshape([2, 1, *grid.shape[1:]])   # 返回一个采样矩阵  分辨率与目标分辨率一致
    pos_embed = get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=use_real)
    return pos_embed


def get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=False):
    assert embed_dim % 4 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_rotary_pos_embed(embed_dim // 2, grid[0].reshape(-1), use_real=use_real)  # (H*W, D/4)
    emb_w = get_1d_rotary_pos_embed(embed_dim // 2, grid[1].reshape(-1), use_real=use_real)  # (H*W, D/4)

    if use_real:
        cos = paddle.concat([emb_h[0], emb_w[0]], axis=1)    # (H*W, D/2)
        sin = paddle.concat([emb_h[1], emb_w[1]], axis=1)    # (H*W, D/2)
        return cos, sin
    else:
        emb = paddle.concat([emb_h, emb_w], axis=1)    # (H*W, D/2)
        return emb


def get_1d_rotary_pos_embed(dim: int, pos: Union[np.ndarray, int], theta: float = 10000.0, use_real=False):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        pos (np.ndarray, int): Position indices for the frequency tensor. [S] or scalar
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool, optional): If True, return real part and imaginary part separately.
                                   Otherwise, return complex numbers.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials. [S, D/2]

    """
    if isinstance(pos, int):
        pos = np.arange(pos)
    freqs = 1.0 / (theta ** (paddle.arange(start=0, end=dim, step=2)[: (dim // 2)].astype(dtype='float32') / dim))  # [D/2]
    t = paddle.to_tensor(data=pos)  # type: ignore  # [S]
    input_0, vec2_0 = TypePromote(t, freqs)
    freqs = paddle.outer(input_0, vec2_0).astype(dtype='float32')  # type: ignore   # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, axis=1)  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, axis=1)  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = paddle.complex(paddle.ones_like(x=freqs) * paddle.cos(
            freqs), paddle.ones_like(x=freqs) * paddle.sin(freqs))
        return freqs_cis



def calc_sizes(rope_img, patch_size, th, tw):
    """ 计算 RoPE 的尺寸. """
    if rope_img == 'extend':
        # 拓展模式
        sub_args = [(th, tw)]
    elif rope_img.startswith('base'):
        # 基于一个尺寸, 其他尺寸插值获得.
        base_size = int(rope_img[4:]) // 8 // patch_size            # 基于512作为base，其他根据512差值得到
        start, stop = get_fill_resize_and_crop((th, tw), base_size)   # 需要在32x32里面 crop的左上角和右下角
        sub_args = [start, stop, (th, tw)]
    else:
        raise ValueError(f"Unknown rope_img: {rope_img}")
    return sub_args


def init_image_posemb(rope_img,
                      resolutions,
                      patch_size,
                      hidden_size,
                      num_heads,
                      log_fn,
                      rope_real=True,
                      ):
    freqs_cis_img = {}
    for reso in resolutions:
        th, tw = reso.height // 8 // patch_size, reso.width // 8 // patch_size
        sub_args = calc_sizes(rope_img, patch_size, th, tw)      #  [左上角, 右下角, 目标高宽]   需要在32x32里面 crop的左上角和右下角
        freqs_cis_img[str(reso)] = get_2d_rotary_pos_embed(hidden_size // num_heads, *sub_args, use_real=rope_real)
        log_fn(f"    Using image RoPE ({rope_img}) ({'real' if rope_real else 'complex'}): {sub_args} | ({reso}) "
               f"{freqs_cis_img[str(reso)][0].shape if rope_real else freqs_cis_img[str(reso)].shape}")
    return freqs_cis_img

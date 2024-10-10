import triton
import triton.language as tl


@triton.jit
def adaptive_layer_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    scale_ptr,
    shift_ptr,
    M,
    N,
    seq_size,
    epsilon,
    BLOCK_SIZE: tl.constexpr,
    weight_attr: tl.constexpr,
    bias_attr: tl.constexpr,
):
    row = tl.program_id(axis=0)
    x_ptr += row * N
    y_ptr += row * N
    # Compute mean
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _sum_square = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for col_off in range(0, N, BLOCK_SIZE):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        eles = tl.load(x_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        _sum += eles
        _sum_square += eles * eles
    mean = tl.sum(_sum, axis=0) / N
    var = tl.sum(_sum_square, axis=0) / N - mean * mean
    rstd = 1 / tl.sqrt(var + epsilon)
    # Compute output
    scale_ptr += (row // seq_size) * N
    shift_ptr += (row // seq_size) * N
    for col_off in range(0, N, BLOCK_SIZE):
        cols = col_off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        eles = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (eles - mean) * rstd
        if weight_attr:
            weights = tl.load(weight_ptr + cols, mask=mask, other=0.0)
            x_hat = x_hat * weights
        if bias_attr:
            bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
            x_hat = x_hat + bias
        scales = tl.load(scale_ptr + cols, mask=mask, other=0.0)
        shifts = tl.load(shift_ptr + cols, mask=mask, other=0.0)
        y = x_hat * (1 + scales) + shifts
        tl.store(y_ptr + cols, y, mask=mask)

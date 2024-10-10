import triton
import triton.language as tl


@triton.jit
def split_concat_kernel(
    out0,
    out1,
    out2,
    qkv,
    eqkv,
    batch,
    seq_qkv,
    seq_eqkv,
    output_hidden,
    BLOCK_SIZE: tl.constexpr,
):
    out_id = tl.program_id(axis=0)
    batch = tl.program_id(axis=1)
    out_row = tl.program_id(axis=2)
    if out_row < seq_qkv:
        read_ptr = out_id * output_hidden + out_row * 3 * output_hidden + batch * seq_qkv * output_hidden * 3 + qkv
    else:
        read_ptr = (
            out_id * output_hidden
            + (out_row - seq_qkv) * 3 * output_hidden
            + batch * seq_eqkv * output_hidden * 3
            + eqkv
        )

    read_offsets = tl.arange(0, BLOCK_SIZE)
    mask = read_offsets < output_hidden
    read_data = tl.load(read_ptr + read_offsets, mask=mask)

    real_output = out0
    if out_id == 1:
        real_output = out1
    elif out_id == 2:
        real_output = out2

    write_ptr = batch * (seq_qkv + seq_eqkv) * output_hidden + out_row * output_hidden + real_output + read_offsets

    tl.store(write_ptr, read_data, mask=mask)

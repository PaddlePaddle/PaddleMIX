import triton
import triton.language as tl


@triton.jit
def ln_partial_rotary_emb_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    q_norm_weight_ptr,
    q_norm_bias_ptr,
    k_norm_weight_ptr,
    k_norm_bias_ptr,
    outq_ptr,
    outk_ptr,
    text_seq_length,
    batch,
    num_heads,
    seq_len,
    n_elements,
    norm_eps,
    HEAD_DIM: tl.constexpr,
):
    # 计算当前线程处理的元素范围
    b_pid = tl.program_id(axis=0)
    h_pid = tl.program_id(axis=1)
    s_pid = tl.program_id(axis=2)

    block_start = b_pid * num_heads * seq_len * HEAD_DIM + h_pid * seq_len * HEAD_DIM + s_pid * HEAD_DIM
    read_offsets = block_start + tl.arange(0, HEAD_DIM)
    mask = read_offsets < n_elements
    q = tl.load(q_ptr + read_offsets, mask=mask)
    k = tl.load(k_ptr + read_offsets, mask=mask)

    # qk layernorm
    offs = tl.arange(0, HEAD_DIM)
    masks = offs < HEAD_DIM
    q_mean = tl.sum(q) / HEAD_DIM
    q_var = tl.sum(q * q) / HEAD_DIM - q_mean * q_mean
    q_rstd = 1 / tl.sqrt(q_var + norm_eps)
    q_resi_hat = (q - q_mean) * q_rstd
    q_weights = tl.load(q_norm_weight_ptr + offs, mask=masks)
    q_resi_hat = q_resi_hat * q_weights
    q_bias = tl.load(q_norm_bias_ptr + offs, mask=masks)
    q_resi_hat = q_resi_hat + q_bias

    k_mean = tl.sum(k, axis=0) / HEAD_DIM
    k_var = tl.sum(k * k, axis=0) / HEAD_DIM - k_mean * k_mean
    k_rstd = 1 / tl.sqrt(k_var + norm_eps)
    k_resi_hat = (k - k_mean) * k_rstd
    k_weights = tl.load(k_norm_weight_ptr + offs, mask=masks)
    k_resi_hat = k_resi_hat * k_weights
    k_bias = tl.load(k_norm_bias_ptr + offs, mask=masks)
    k_resi_hat = k_resi_hat + k_bias

    # qk rotary_emb
    if s_pid > text_seq_length - 1:
        q1, q2 = tl.split(tl.reshape(q_resi_hat, (32, 2)))
        qc = tl.interleave(-q2, q1)

        k1, k2 = tl.split(tl.reshape(k_resi_hat, (32, 2)))
        kc = tl.interleave(-k2, k1)

        block_cs_start = (s_pid - text_seq_length) * HEAD_DIM
        read_cs_offsets = block_cs_start + tl.arange(0, HEAD_DIM)
        cs_mask = read_cs_offsets < ((seq_len - text_seq_length) * HEAD_DIM)
        cos = tl.load(cos_ptr + read_cs_offsets, mask=cs_mask)
        sin = tl.load(sin_ptr + read_cs_offsets, mask=cs_mask)

        oq = tl.cast(tl.cast(q_resi_hat, tl.float32) * cos + tl.cast(qc, tl.float32) * sin, tl.float16)
        ok = tl.cast(tl.cast(k_resi_hat, tl.float32) * cos + tl.cast(kc, tl.float32) * sin, tl.float16)

        tl.store(outq_ptr + read_offsets, oq, mask=mask)
        tl.store(outk_ptr + read_offsets, ok, mask=mask)
    else:
        tl.store(outq_ptr + read_offsets, q_resi_hat, mask=mask)
        tl.store(outk_ptr + read_offsets, k_resi_hat, mask=mask)

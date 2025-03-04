import paddle
import triton
import triton.language as tl
from paddle import _C_ops
from paddle.base.framework import OpProtoHolder
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode
from typing import Any, List, Literal, Optional, Tuple, Union

from .triton_utils import get_dtype_str, paddle_use_triton, rendering_common_template

########################### sage attention ###############################
@paddle_use_triton(
    key=["1"]
)
def sageattn_quant_per_block_int8_kernel(Input, Output, Scale, L,
                                        stride_iz, stride_ih, stride_in,
                                        stride_oz, stride_oh, stride_on,
                                        stride_sz, stride_sh,
                                        sm_scale,
                                        h_attn: tl.constexpr,                 # grid num, through compiling
                                        bsz: tl.constexpr,                    # grid num, through compiling
                                        C: tl.constexpr,
                                        BLK: tl.constexpr
                                    ):
    off_blk = tl.program_id(axis=0)
    off_h = tl.program_id(axis=1)
    off_b = tl.program_id(axis=2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk

    x_data = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x_data = x_data.to(tl.float32)
    x_data *= sm_scale
    scale = tl.max(tl.abs(x_data)) / 127.
    x_int8 = x_data / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)
    
# per-block quant triton API
def sageattn_quant_per_block_int8(x, 
                                km=None, 
                                BLK=128,
                                sm_scale=1.0, 
                                tensor_layout="HND"):
    """
    [params]
        x: paddle.Tensor, dtype in fp16 or bf16, this is usually q or k input tensor.
        km: paddle.Tensor, the mean tensor of k tensor. Must be provided when the `x` is k tensor.
        BLK: int, the BLK for computing q & k tensor. Default 128 for q, 64 for k, which is an optimized value.
        sm_scale: float, the scale factor for dynamic quant.
        tensor_layout: string. Only in ['HND', 'NHD'], 'HND' -> [bsz, num_heads, seq_len, head_dim],
                                                        'HND' -> [bsz, seq_len, num_heads, head_dim]
    [Examples]
        batch_size = 2
        num_heads = 24
        seq_len = 1376
        head_dim = 64
        
        sm_scale = 1.0 / (head_dim_og ** 0.5)
        
        # note: this layout is 'NHD'
        tensor_layout = 'NHD'
        q = paddle.randn(shape=(batch_size, seq_len, num_heads, head_dim), dtype="float16")
        k = paddle.randn(shape=(batch_size, seq_len, num_heads, head_dim), dtype="float16")
        v = paddle.randn(shape=(batch_size, seq_len, num_heads, head_dim), dtype="float16")
        
        km = paddle.mean(k, axis=seq_dim, keepdim=True)
        
        q_int8, q_scale = sageattn_quant_per_block_int8(
            q, km=None, BLK=BLKQ, sm_scale=sm_scale, tensor_layout=tensor_layout)
        k_int8, k_scale = sageattn_quant_per_block_int8(
            k, km=km, BLK=BLKK, sm_scale=sm_scale, tensor_layout=tensor_layout)
    """
    if km is not None:
        x = x - km
        
    if tensor_layout == "HND":
        b, h_attn, seq_len, head_dim = x.shape

        # there is no `stride` attribute in static mode, so we need to compute it manually
        stride_iz, stride_ih, stride_in = head_dim * seq_len * h_attn, head_dim * seq_len, head_dim * 1
        stride_oz, stride_oh, stride_on = head_dim * seq_len * h_attn, head_dim * seq_len, head_dim * 1
    elif tensor_layout == "NHD":
        b, seq_len, h_attn, head_dim = x.shape
        
        stride_iz, stride_ih, stride_in = head_dim * seq_len * h_attn, head_dim * 1, head_dim * h_attn
        stride_oz, stride_oh, stride_on = head_dim * seq_len * h_attn, head_dim * 1, head_dim * h_attn
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    
    if sm_scale is None:
        sm_scale = head_dim**-0.5

    L = seq_len
    C = head_dim
    sm_scale = sm_scale * 1.44269504 if km is None else 1.0

    stride_sz = h_attn * ((seq_len + BLK - 1) // BLK)
    stride_sh =  (seq_len + BLK - 1) // BLK
    
    prepare_attr_for_triton_kernel = """
    auto output_tensor = paddle::empty(x.shape(), paddle::DataType::INT8, x.place());
    
    auto input_tensor = x;
    auto input_shape = x.shape();
    
    // define params
    int b, h_attn, seq_len, head_dim;
    int stride_iz, stride_ih, stride_in;
    int stride_oz, stride_oh, stride_on;
    
    // allocate
    if (tensor_layout == std::string("HND")) {
        // tensor layout unpack
        b = input_shape[0];
        h_attn = input_shape[1];
        seq_len = input_shape[2];
        head_dim = input_shape[3];
        
        // stride unpack
        auto tensor_strides = input_tensor.strides();
        stride_iz = tensor_strides[0];
        stride_ih = tensor_strides[1];
        stride_in = tensor_strides[2];
        
        auto tensor_o_strides = output_tensor.strides();
        stride_oz = tensor_o_strides[0];
        stride_oh = tensor_o_strides[1];
        stride_on = tensor_o_strides[2];
        
    } else if (tensor_layout == std::string("NHD")) {
        // tensor layout unpack
        b = input_shape[0];
        h_attn = input_shape[2];    // reverse
        seq_len = input_shape[1];
        head_dim = input_shape[3];
        
        // stride unpack
        auto tensor_strides = input_tensor.strides();
        stride_iz = tensor_strides[0];
        stride_ih = tensor_strides[2];    // reverse
        stride_in = tensor_strides[1];
        
        auto tensor_o_strides = output_tensor.strides();
        stride_oz = tensor_o_strides[0];
        stride_oh = tensor_o_strides[2];  // reverse
        stride_on = tensor_o_strides[1];
    }
    else {
        throw std::runtime_error("Unsupported tensor layout");
    }

    auto scale_tensor = paddle::empty({b, h_attn, (seq_len + BLK - 1) / BLK}, 
                                        paddle::DataType::FLOAT32, x.place());
    int L = seq_len;
    int stride_sz = scale_tensor.strides()[0];
    int stride_sh = scale_tensor.strides()[1];
    int bsz = b;
"""

    op_name = "triton_sageattn_quant_per_block"
    op_name += get_dtype_str(x.dtype)
    op_name += f"_BLK{BLK}_seq{seq_len}_h{h_attn}_dim{head_dim}"
    
    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        Output = paddle.empty(x.shape, dtype=paddle.int8)
        Scale = paddle.empty((b, h_attn, (seq_len + BLK - 1) // BLK), dtype='float32')
        # output_tensor & scale_tensor has beed defined in above areas
        prepare_ptr_for_triton_kernel = """
    // prepare tensor
    CUdeviceptr input_ptrs[3] = {
        get_tensor_ptr(x),
        get_tensor_ptr(output_tensor),
        get_tensor_ptr(scale_tensor)
    };
"""
        return_tensor_names = "output_tensor, scale_tensor"
        
        template_used = rendering_common_template(
            sageattn_quant_per_block_int8, 
            prepare_attr_for_triton_kernel=prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel=prepare_ptr_for_triton_kernel,
            return_tensor_names=return_tensor_names
        )
        grid = ("(L + BLK - 1) / BLK", "h_attn", "bsz")
        sageattn_quant_per_block_int8_kernel[(op_name, template_used, grid)](
            Input=x, 
            Output=Output, 
            Scale=Scale, 
            L=L, 
            stride_iz=stride_iz, 
            stride_ih=stride_ih, 
            stride_in=stride_in,
            stride_oz=stride_oz, 
            stride_oh=stride_oh, 
            stride_on=stride_on,
            stride_sz=stride_sz, 
            stride_sh=stride_sh,
            sm_scale=sm_scale,
            h_attn=h_attn,      # grid num, for compiling
            bsz=b,              # grid num, for compiling
            C=C, 
            BLK=BLK
        )
        
        
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            op_name, x, km, BLK,
            sm_scale, tensor_layout
        )
        return outs[0], outs[1]
    else:
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "x": x,
            "km@OPTIONAL": km,
        }
        out_int8 = helper.create_variable_for_type_inference(dtype=paddle.int8)
        out_scale = helper.create_variable_for_type_inference(dtype=paddle.float32)
        
        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs={
                "BLK": BLK,
                "sm_scale": sm_scale,
                "tensor_layout": tensor_layout,
            },
            outputs={"output_tensor": out_int8, "scale_tensor": out_scale}
        )
        return out_int8, out_scale


def per_block_int8(q, k, km=None, BLKQ=128, BLKK=64, sm_scale=None, 
                   tensor_layout="HND"):
    q_int8, q_scale = sageattn_quant_per_block_int8(
        q, km=None, BLK=BLKQ, sm_scale=sm_scale, tensor_layout=tensor_layout)
    k_int8, k_scale = sageattn_quant_per_block_int8(
        k, km=km, BLK=BLKK, sm_scale=sm_scale, tensor_layout=tensor_layout)
    return q_int8, q_scale, k_int8, k_scale


@paddle_use_triton(
    key=["1"]
)
def sageattn_quant_query_per_thread_int8_kernel(Input, Output, Scale, L,
                                                stride_iz, stride_ih, stride_in,
                                                stride_oz, stride_oh, stride_on,
                                                stride_sz, stride_sh,
                                                h_qo: tl.constexpr, bsz: tl.constexpr,
                                                C: tl.constexpr, BLK: tl.constexpr, 
                                                WARP: tl.constexpr, BKG: tl.constexpr):
    off_blk = tl.program_id(0) // 8
    off_tld = tl.program_id(0) % 8
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 8 + off_tld

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    scale = tl.max(tl.abs(x)) / 127. + 0.0000001
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)


def sageattn_quant_query_per_thread_int8(x, 
                                        BLK=128,
                                        WARP=32,
                                        sm_scale=1.0, 
                                        tensor_layout="HND"):
    """
    [params]
        x: paddle.Tensor, dtype in fp16 or bf16, this is usually q or k input tensor.
        BLK: int, the BLK for computing q tensor. Default 128 for q, 64 for k, which is an optimized value.
        WARP: int, the WARP for computing q. Default 32.
        sm_scale: float, the scale factor for dynamic quant.
        tensor_layout: string. Only in ['HND', 'NHD'], 'HND' -> [bsz, num_heads, seq_len, head_dim],
                                                        'HND' -> [bsz, seq_len, num_heads, head_dim]
    """ 
    if tensor_layout == "HND":
        b, h_qo, seq_len, head_dim = x.shape

        # there is no `stride` attribute in static mode, so we need to compute it manually
        stride_bz_q, stride_h_q, stride_seq_q    = head_dim * seq_len * h_qo, head_dim * seq_len, head_dim * 1
        stride_bz_qo, stride_h_qo, stride_seq_qo = head_dim * seq_len * h_qo, head_dim * seq_len, head_dim * 1
    elif tensor_layout == "NHD":
        b, seq_len, h_qo, head_dim = x.shape
        
        stride_bz_q, stride_h_q, stride_seq_q    = head_dim * seq_len * h_qo, head_dim * 1, head_dim * h_qo
        stride_bz_qo, stride_h_qo, stride_seq_qo = head_dim * seq_len * h_qo, head_dim * 1, head_dim * h_qo
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    
    if sm_scale is None:
        sm_scale = head_dim**-0.5

    L = seq_len
    C = head_dim

    stride_sz = h_qo * ((seq_len + BLK - 1) // BLK) * (BLK // WARP) * 8
    stride_sh =  ((seq_len + BLK - 1) // BLK) * (BLK // WARP) * 8
    
    prepare_attr_for_triton_kernel = """
    auto output_tensor = paddle::empty(x.shape(), paddle::DataType::INT8, x.place());
    
    auto input_tensor = x;
    auto input_shape = x.shape();
    
    // define params
    int b, h_qo, seq_len, head_dim;
    int stride_iz, stride_ih, stride_in;
    int stride_oz, stride_oh, stride_on;
    
    // allocate
    if (tensor_layout == std::string("HND")) {
        // tensor layout unpack
        b = input_shape[0];
        h_qo = input_shape[1];
        seq_len = input_shape[2];
        head_dim = input_shape[3];
        
        // stride unpack
        auto tensor_strides = input_tensor.strides();
        stride_iz = tensor_strides[0];
        stride_ih = tensor_strides[1];
        stride_in = tensor_strides[2];
        
        auto tensor_o_strides = output_tensor.strides();
        stride_oz = tensor_o_strides[0];
        stride_oh = tensor_o_strides[1];
        stride_on = tensor_o_strides[2];
        
    } else if (tensor_layout == std::string("NHD")) {
        // tensor layout unpack
        b = input_shape[0];
        h_qo = input_shape[2];    // reverse
        seq_len = input_shape[1];
        head_dim = input_shape[3];
        
        // stride unpack
        auto tensor_strides = input_tensor.strides();
        stride_iz = tensor_strides[0];
        stride_ih = tensor_strides[2];    // reverse
        stride_in = tensor_strides[1];
        
        auto tensor_o_strides = output_tensor.strides();
        stride_oz = tensor_o_strides[0];
        stride_oh = tensor_o_strides[2];  // reverse
        stride_on = tensor_o_strides[1];
    }
    else {
        throw std::runtime_error("Unsupported tensor layout");
    }

    auto scale_tensor = paddle::empty({b, h_qo, (seq_len + BLK - 1) / BLK * (BLK / WARP) * 8}, 
                                        paddle::DataType::FLOAT32, x.place());
    int L = seq_len;
    int stride_sz = scale_tensor.strides()[0];
    int stride_sh = scale_tensor.strides()[1];
    int bsz = b;
"""

    op_name = "triton_sageattn_quant_query_per_thread"
    op_name += get_dtype_str(x.dtype)
    op_name += f"_BLK{BLK}_seq{seq_len}_h{h_qo}_dim{head_dim}"
    
    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        Output = paddle.empty(x.shape, dtype=paddle.int8)
        Scale = paddle.empty(shape=(b, h_qo, ((seq_len + BLK - 1) // BLK) * (BLK // WARP) * 8), dtype='float32')
        # output_tensor & scale_tensor has beed defined in above areas
        prepare_ptr_for_triton_kernel = """
    // prepare tensor
    CUdeviceptr input_ptrs[3] = {
        get_tensor_ptr(x),
        get_tensor_ptr(output_tensor),
        get_tensor_ptr(scale_tensor)
    };
"""
        return_tensor_names = "output_tensor, scale_tensor"
        
        template_used = rendering_common_template(
            sageattn_quant_query_per_thread_int8, 
            prepare_attr_for_triton_kernel=prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel=prepare_ptr_for_triton_kernel,
            return_tensor_names=return_tensor_names
        )
        grid = ("((L + BKG - 1) / BKG) * (BKG / WARP) * 8", "h_qo", "bsz")
        sageattn_quant_query_per_thread_int8_kernel[(op_name, template_used, grid)](
            Input=x, 
            Output=Output, 
            Scale=Scale, 
            L=L, 
            stride_iz=stride_bz_q, 
            stride_ih=stride_h_q, 
            stride_in=stride_seq_q,
            stride_oz=stride_bz_qo, 
            stride_oh=stride_h_qo, 
            stride_on=stride_seq_qo,
            stride_sz=stride_sz, 
            stride_sh=stride_sh,
            h_qo=h_qo,          # grid num, for compiling
            bsz=b,              # grid num, for compiling
            C=C, 
            BLK=WARP,
            WARP=WARP,          # grid num, for compiling
            BKG=BLK            # grid num, for compiling
        )
        
        
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            op_name, x, BLK, WARP,
            sm_scale, tensor_layout
        )
        return outs[0], outs[1]
    else:
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "x": x
        }
        out_int8 = helper.create_variable_for_type_inference(dtype=paddle.int8)
        out_scale = helper.create_variable_for_type_inference(dtype=paddle.float32)
        
        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs={
                "BLK": BLK,
                "WARP": WARP,
                "sm_scale": sm_scale,
                "tensor_layout": tensor_layout,
            },
            outputs={"output_tensor": out_int8, "scale_tensor": out_scale}
        )
        return out_int8, out_scale


@paddle_use_triton(
    key=["1"]
)
def sageattn_quant_key_per_thread_int8_kernel(Input, Output, Scale, L,
                                            stride_iz, stride_ih, stride_in,
                                            stride_oz, stride_oh, stride_on,
                                            stride_sz, stride_sh,
                                            h_kv: tl.constexpr, bsz: tl.constexpr,
                                            C: tl.constexpr, BLK: tl.constexpr, 
                                            WARP: tl.constexpr, BKG: tl.constexpr):      
    off_blk = tl.program_id(0) // 4
    off_tld = tl.program_id(0) % 4
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n0 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2
    offs_n1 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2 + 1
    offs_k = tl.arange(0, C)

    input_ptrs0 = Input + off_b * stride_iz + off_h * stride_ih + offs_n0[:, None] * stride_in + offs_k[None, :]
    input_ptrs1 = Input + off_b * stride_iz + off_h * stride_ih + offs_n1[:, None] * stride_in + offs_k[None, :]
    output_ptrs0 = Output + off_b * stride_oz + off_h * stride_oh + offs_n0[:, None] * stride_on + offs_k[None, :]
    output_ptrs1 = Output + off_b * stride_oz + off_h * stride_oh + offs_n1[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 4 + off_tld

    x0 = tl.load(input_ptrs0, mask=offs_n0[:, None] < L)
    x1 = tl.load(input_ptrs1, mask=offs_n1[:, None] < L)
    x0 = x0.to(tl.float32)
    x1 = x1.to(tl.float32)
    scale = max(tl.max(tl.abs(x0)), tl.max(tl.abs(x1))) / 127. + 0.0000001
    x0_int8 = x0 / scale
    x1_int8 = x1 / scale
    x0_int8 += 0.5 * tl.where(x0_int8 >= 0, 1, -1)
    x1_int8 += 0.5 * tl.where(x1_int8 >= 0, 1, -1)
    x0_int8 = x0_int8.to(tl.int8)
    x1_int8 = x1_int8.to(tl.int8)
    tl.store(output_ptrs0, x0_int8, mask=offs_n0[:, None] < L)
    tl.store(output_ptrs1, x1_int8, mask=offs_n1[:, None] < L)
    tl.store(scale_ptrs, scale)


def sageattn_quant_key_per_thread_int8(x, 
                                    km,
                                    BLK=64,
                                    WARP=64,
                                    sm_scale=1.0,
                                    tensor_layout="HND"):
    """
    [params]
        x: paddle.Tensor, dtype in fp16 or bf16, this is usually q or k input tensor.
        km: paddle.Tensor, the mean in seq_dim of tensor K.
        BLK: int, the BLK for computing q tensor. Default 128 for q, 64 for k, which is an optimized value.
        WARP: int, the WARP for computing q. Default 64.
        sm_scale: float, the scale factor for dynamic quant.
        tensor_layout: string. Only in ['HND', 'NHD'], 'HND' -> [bsz, num_heads, seq_len, head_dim],
                                                        'HND' -> [bsz, seq_len, num_heads, head_dim]
    """ 
    if tensor_layout == "HND":
        b, h_kv, seq_len, head_dim = x.shape

        # there is no `stride` attribute in static mode, so we need to compute it manually
        stride_bz_k, stride_h_k, stride_seq_k    = head_dim * seq_len * h_kv, head_dim * seq_len, head_dim * 1
        stride_bz_ko, stride_h_ko, stride_seq_ko = head_dim * seq_len * h_kv, head_dim * seq_len, head_dim * 1
    elif tensor_layout == "NHD":
        b, seq_len, h_kv, head_dim = x.shape
        
        stride_bz_k, stride_h_k, stride_seq_k    = head_dim * seq_len * h_kv, head_dim * 1, head_dim * h_kv
        stride_bz_ko, stride_h_ko, stride_seq_ko = head_dim * seq_len * h_kv, head_dim * 1, head_dim * h_kv
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    
    if sm_scale is None:
        sm_scale = head_dim**-0.5

    x = x - km

    L = seq_len
    C = head_dim

    stride_sz = h_kv * ((seq_len + BLK - 1) // BLK) * (BLK // WARP) * 4
    stride_sh =  ((seq_len + BLK - 1) // BLK) * (BLK // WARP) * 4
    
    prepare_attr_for_triton_kernel = """
    auto output_tensor = paddle::empty(x.shape(), paddle::DataType::INT8, x.place());
    
    auto input_tensor = x;
    auto input_shape = x.shape();
    
    // define params
    int b, h_kv, seq_len, head_dim;
    int stride_iz, stride_ih, stride_in;
    int stride_oz, stride_oh, stride_on;
    
    // allocate
    if (tensor_layout == std::string("HND")) {
        // tensor layout unpack
        b = input_shape[0];
        h_kv = input_shape[1];
        seq_len = input_shape[2];
        head_dim = input_shape[3];
        
        // stride unpack
        auto tensor_strides = input_tensor.strides();
        stride_iz = tensor_strides[0];
        stride_ih = tensor_strides[1];
        stride_in = tensor_strides[2];
        
        auto tensor_o_strides = output_tensor.strides();
        stride_oz = tensor_o_strides[0];
        stride_oh = tensor_o_strides[1];
        stride_on = tensor_o_strides[2];
        
    } else if (tensor_layout == std::string("NHD")) {
        // tensor layout unpack
        b = input_shape[0];
        h_kv = input_shape[2];    // reverse
        seq_len = input_shape[1];
        head_dim = input_shape[3];
        
        // stride unpack
        auto tensor_strides = input_tensor.strides();
        stride_iz = tensor_strides[0];
        stride_ih = tensor_strides[2];    // reverse
        stride_in = tensor_strides[1];
        
        auto tensor_o_strides = output_tensor.strides();
        stride_oz = tensor_o_strides[0];
        stride_oh = tensor_o_strides[2];  // reverse
        stride_on = tensor_o_strides[1];
    }
    else {
        throw std::runtime_error("Unsupported tensor layout");
    }

    auto scale_tensor = paddle::empty({b, h_kv, (seq_len + BLK - 1) / BLK * (BLK / WARP) * 4}, 
                                        paddle::DataType::FLOAT32, x.place());
    int L = seq_len;
    int stride_sz = scale_tensor.strides()[0];
    int stride_sh = scale_tensor.strides()[1];
    int bsz = b;
"""

    op_name = "triton_sageattn_quant_key_per_thread"
    op_name += get_dtype_str(x.dtype)
    op_name += f"_BLK{BLK}_seq{seq_len}_h{h_kv}_dim{head_dim}"
    
    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        Output = paddle.empty(x.shape, dtype=paddle.int8)
        Scale = paddle.empty(shape=(b, h_kv, ((seq_len + BLK - 1) // BLK) * (BLK // WARP) * 4), dtype='float32')
        # output_tensor & scale_tensor has beed defined in above areas
        prepare_ptr_for_triton_kernel = """
    // prepare tensor
    CUdeviceptr input_ptrs[3] = {
        get_tensor_ptr(x),
        get_tensor_ptr(output_tensor),
        get_tensor_ptr(scale_tensor)
    };
"""
        return_tensor_names = "output_tensor, scale_tensor"
        
        template_used = rendering_common_template(
            sageattn_quant_key_per_thread_int8, 
            prepare_attr_for_triton_kernel=prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel=prepare_ptr_for_triton_kernel,
            return_tensor_names=return_tensor_names
        )
        grid = ("((L + BKG - 1) / BKG) * (BKG / WARP) * 8", "h_kv", "bsz")
        sageattn_quant_key_per_thread_int8_kernel[(op_name, template_used, grid)](
            Input=x, 
            Output=Output, 
            Scale=Scale, 
            L=L, 
            stride_iz=stride_bz_k, 
            stride_ih=stride_h_k, 
            stride_in=stride_seq_k,
            stride_oz=stride_bz_ko, 
            stride_oh=stride_h_ko, 
            stride_on=stride_seq_ko,
            stride_sz=stride_sz, 
            stride_sh=stride_sh,
            h_kv=h_kv,          # grid num, for compiling
            bsz=b,              # grid num, for compiling
            C=C, 
            BLK=WARP,
            WARP=WARP,          # grid num, for compiling
            BKG=BLK            # grid num, for compiling
        )
        
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            op_name, x, km, BLK, WARP,
            sm_scale, tensor_layout
        )
        return outs[0], outs[1]
    else:
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "x": x,
            "km": km
        }
        out_int8 = helper.create_variable_for_type_inference(dtype=paddle.int8)
        out_scale = helper.create_variable_for_type_inference(dtype=paddle.float32)
        
        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs={
                "BLK": BLK,
                "WARP": WARP,
                "sm_scale": sm_scale,
                "tensor_layout": tensor_layout,
            },
            outputs={"output_tensor": out_int8, "scale_tensor": out_scale}
        )
        return out_int8, out_scale


def per_thread_int8(q, k, km=None, BLKQ=128, BLKK=64, WARPQ=32, WARPK=64, sm_scale=None, 
                   tensor_layout="HND"):
    q_int8, q_scale = sageattn_quant_query_per_thread_int8(
        q, BLK=BLKQ, WARP=WARPQ, sm_scale=sm_scale, tensor_layout=tensor_layout)
    k_int8, k_scale = sageattn_quant_key_per_thread_int8(
        k, km=km, BLK=BLKK, WARP=WARPK, sm_scale=sm_scale, tensor_layout=tensor_layout)
    return q_int8, q_scale, k_int8, k_scale


@paddle_use_triton(
    key=["1"]
)
def sageattn_attn_fwd_causal_false_kernel(
            Q, K, V, Q_scale, K_scale, Out, Lse, 
            stride_qz, stride_qh, stride_qn,
            stride_kz, stride_kh, stride_kn,  
            stride_vz, stride_vh, stride_vn,  
            stride_oz, stride_oh, stride_on,  
            qo_len, kv_len, BSZ,
            h_qo: tl.constexpr, 
            num_kv_groups: tl.constexpr,
            HEAD_DIM: tl.constexpr,  
            BLOCK_M: tl.constexpr,  
            BLOCK_N: tl.constexpr,  
            RETURN_LSE: tl.constexpr):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    q_scale_offset = (off_z * h_qo + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (h_qo // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)  
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] 
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    q = tl.load(Q_ptrs, mask = offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    
    # fused zone 
    lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)   
        k = tl.load(K_ptrs, mask = k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale 
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        
        v = tl.load(V_ptrs, mask = offs_n[:, None] < (kv_len - start_n))
        p = p.to(tl.float16)
        
        acc += tl.dot(p, v, out_dtype=tl.float16)   
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    # zone end

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * h_qo + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask = (offs_m < qo_len))
        

def sageattn_forward_causal_false(q, k, v, 
                                  q_scale, k_scale, 
                                  output_dtype="float16",
                                  tensor_layout="HND", 
                                  return_lse=False):
    """
    [params]
        q: paddle.Tensor, dtype in int8, q tensor after quant.
        k: paddle.Tensor, dtype in int8, k tensor after quant.
        v: paddle.Tensor, dtype in fp16 or bf16, v tensor.
        q_scale: paddle.Tensor, dtype in fp16 or bf16, this is the output tensor for scale factor, from quant kernel.
        k_scale: paddle.Tensor, dtype in fp16 or bf16, this is the output tensor for scale factor, from quant kernel.
        output_dtype: string. Only in ['float16', 'bfloat16']. The datatype of q, k, v tensor.
        tensor_layout: string. Only in ['HND', 'NHD'], 'HND' -> [bsz, num_heads, seq_len, head_dim],
                        'HND' -> [bsz, seq_len, num_heads, head_dim]
        return_lse: bool. Return lse correction or not. Useful in parallel computing. Default False.
    [Examples]
        batch_size = 2
        num_heads = 24
        seq_len = 1376
        head_dim = 64
        
        sm_scale = 1.0 / (head_dim_og ** 0.5)
        
        # note: this layout is 'NHD'
        tensor_layout = 'NHD'
        q = paddle.randn(shape=(batch_size, seq_len, num_heads, head_dim), dtype="float16")
        k = paddle.randn(shape=(batch_size, seq_len, num_heads, head_dim), dtype="float16")
        v = paddle.randn(shape=(batch_size, seq_len, num_heads, head_dim), dtype="float16")
        
        km = paddle.mean(k, axis=seq_dim, keepdim=True)
        
        q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout)
        o, lse = sageattn_forward_causal_false(q_int8, k_int8, v, q_scale, k_scale, 
                                                output_dtype="float16", tensor_layout=tensor_layout)
    """
    assert output_dtype in ["float16", "bfloat16"]
    
    Out = paddle.empty(q.shape, dtype=output_dtype)
    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_qz, stride_qh, stride_qn = h_qo * qo_len * head_dim, qo_len * head_dim, head_dim
        stride_kz, stride_kh, stride_kn = h_kv * kv_len * head_dim, kv_len * head_dim, head_dim
        stride_vz, stride_vh, stride_vn = h_kv * kv_len * head_dim, kv_len * head_dim, head_dim
        stride_oz, stride_oh, stride_on = h_qo * qo_len * head_dim, qo_len * head_dim, head_dim
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        
        stride_qz, stride_qh, stride_qn = qo_len * h_qo * head_dim, head_dim, h_qo * head_dim
        stride_kz, stride_kh, stride_kn = kv_len * h_kv * head_dim, head_dim, h_kv * head_dim
        stride_vz, stride_vh, stride_vn = kv_len * h_kv * head_dim, head_dim, h_kv * head_dim
        stride_oz, stride_oh, stride_on = qo_len * h_qo * head_dim, head_dim, h_qo * head_dim
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    
    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv
    BSZ = b
    
    prepare_attr_for_triton_kernel = """
    paddle::DataType output_t;
    if (output_dtype == std::string("float16")) {
        output_t = paddle::DataType::FLOAT16;
    } else {
        output_t = paddle::DataType::BFLOAT16;
    }
    
    auto out_tensor = paddle::empty(q.shape(), output_t, q.place());
    auto q_strides = q.strides();
    auto k_strides = k.strides();
    auto v_strides = v.strides();
    auto o_strides = out_tensor.strides();
    
    int b, h_qo, qo_len, head_dim;
    int kv_len, h_kv;
    
    int stride_qz, stride_qh, stride_qn;
    int stride_kz, stride_kh, stride_kn;
    int stride_vz, stride_vh, stride_vn;
    int stride_oz, stride_oh, stride_on;
    
    if (tensor_layout == "HND") {
        b = q.shape()[0];
        h_qo = q.shape()[1];
        qo_len = q.shape()[2];
        head_dim = q.shape()[3];
        
        h_kv = k.shape()[1];
        kv_len = k.shape()[2];
        
        stride_qz = q_strides[0];
        stride_qh = q_strides[1];
        stride_qn = q_strides[2];
        
        stride_kz = k_strides[0];
        stride_kh = k_strides[1];
        stride_kn = k_strides[2];
        
        stride_vz = v_strides[0];
        stride_vh = v_strides[1];
        stride_vn = v_strides[2];
        
        stride_oz = o_strides[0];
        stride_oh = o_strides[1];
        stride_on = o_strides[2];
    } else if (tensor_layout == "NHD") {
        b = q.shape()[0];
        qo_len = q.shape()[1];   // reverse
        h_qo = q.shape()[2];
        head_dim = q.shape()[3];
        
        kv_len = k.shape()[1];   // reverse
        h_kv = k.shape()[2];
        
        stride_qz = q_strides[0];
        stride_qh = q_strides[2];       // reverse
        stride_qn = q_strides[1];
        
        stride_kz = k_strides[0];
        stride_kh = k_strides[2];       // reverse
        stride_kn = k_strides[1];
        
        stride_vz = v_strides[0];
        stride_vh = v_strides[2];       // reverse
        stride_vn = v_strides[1];
        
        stride_oz = o_strides[0];
        stride_oh = o_strides[2];       // reverse
        stride_on = o_strides[1];
    } else {
        throw std::runtime_error("Unsupported tensor layout");
    }
    
    int BSZ = b;
"""

    op_name = "triton_sageattn_attn_fwd_causal_false"
    op_name += get_dtype_str(q.dtype)
    op_name += f"_BSZ{BSZ}_seq{qo_len}_h{h_qo}_dim{HEAD_DIM_K}"
    
    sageattn_attn_fwd_causal_false_config = []
    if head_dim == 64:
        sageattn_attn_fwd_causal_false_config.append({
            "num_warps": 4,
            "num_stages": 3
        })
    else:
        sageattn_attn_fwd_causal_false_config.append({
            "num_warps": 8,
            "num_stages": 4
        })
    
    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        if return_lse:
            Lse = paddle.empty((b, h_qo, qo_len), dtype=paddle.float32)
        else:
            Lse = paddle.empty((0, 0, 0), dtype=paddle.float32)
        prepare_ptr_for_triton_kernel = """
        paddle::Tensor lse_tensor;
        
        if (return_lse) {
            lse_tensor = paddle::empty({b, h_qo, qo_len}, q.dtype(), q.place());
        } else {
            lse_tensor = paddle::empty({1,1,1}, paddle::DataType::FLOAT32, paddle::CPUPlace());
        }
        
        CUdeviceptr input_ptrs[7] = {
            get_tensor_ptr(q),
            get_tensor_ptr(k),
            get_tensor_ptr(v),
            get_tensor_ptr(q_scale),
            get_tensor_ptr(k_scale),
            get_tensor_ptr(out_tensor),
            get_tensor_ptr(lse_tensor)
        };
    """
        return_tensor_names = "out_tensor, lse_tensor"
        template_used = rendering_common_template(
            sageattn_forward_causal_false, 
            prepare_attr_for_triton_kernel=prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel=prepare_ptr_for_triton_kernel,
            return_tensor_names=return_tensor_names
        )
        grid = ("(qo_len + BLOCK_M - 1) / BLOCK_M", "h_qo", "BSZ")
        sageattn_attn_fwd_causal_false_kernel[(op_name, template_used, grid, sageattn_attn_fwd_causal_false_config)](
            Q=q, 
            K=k, 
            V=v, 
            Q_scale=q_scale, 
            K_scale=k_scale, 
            Out=Out, 
            Lse=Lse, 
            stride_qz=stride_qz, 
            stride_qh=stride_qh, 
            stride_qn=stride_qn,
            stride_kz=stride_kz, 
            stride_kh=stride_kh, 
            stride_kn=stride_kn,
            stride_vz=stride_vz, 
            stride_vh=stride_vh, 
            stride_vn=stride_vn,
            stride_oz=stride_oz, 
            stride_oh=stride_oh, 
            stride_on=stride_on,
            qo_len=qo_len,
            kv_len=kv_len, 
            BSZ=BSZ,
            h_qo=h_qo, 
            num_kv_groups=num_kv_groups,
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_M=128, 
            BLOCK_N=64, 
            RETURN_LSE=1 if return_lse else 0
        )
        
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            op_name,
            q,k,v, q_scale, k_scale, 
            output_dtype,
            tensor_layout, 
            return_lse
        )

        return outs[0], outs[1]
    else:
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "q": q,
            "k": k,
            "v": v,
            "q_scale": q_scale,
            "k_scale": k_scale,
        }
        out_tensor = helper.create_variable_for_type_inference(dtype=Out.dtype)
        out_lse = helper.create_variable_for_type_inference(dtype=Lse.dtype)
        
        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs={
                "output_type": output_dtype,
                "tensor_layout": tensor_layout,
                "return_lse": 1 if return_lse else 0
            },
            outputs={
                "out_tensor": out_tensor,
                "lse_tensor": out_lse
            }
        )
        
        return out_tensor, out_lse
    

@paddle_use_triton(
    key=["1"]
)
def sageattn_attn_fwd_causal_true_kernel(
            Q, K, V, Q_scale, K_scale, Out, Lse, 
            stride_qz, stride_qh, stride_qn,
            stride_kz, stride_kh, stride_kn,  
            stride_vz, stride_vh, stride_vn,  
            stride_oz, stride_oh, stride_on,  
            qo_len, kv_len, BSZ,
            h_qo: tl.constexpr, 
            num_kv_groups: tl.constexpr,
            HEAD_DIM: tl.constexpr,  
            BLOCK_M: tl.constexpr,  
            BLOCK_N: tl.constexpr,  
            RETURN_LSE: tl.constexpr,):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    q_scale_offset = (off_z * h_qo + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (h_qo // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, BLOCK_N)  
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None] 
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    q = tl.load(Q_ptrs, mask = offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    
    # restore the K_scale_ptr, K_ptrs, V_ptrs
    original_K_scale_ptr = K_scale_ptr
    original_K_ptrs = K_ptrs
    original_V_ptrs = V_ptrs
    
    # fused zone
    # STAGE == 1
    lo, hi = 0, start_m * BLOCK_M
    
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)   
        k = tl.load(K_ptrs, mask = k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale 

        # stage == 1 branch
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        # stage == 1 branch end
        
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        
        v = tl.load(V_ptrs, mask = offs_n[:, None] < (kv_len - start_n))
        p = p.to(tl.float16)
        
        acc += tl.dot(p, v, out_dtype=tl.float16)   
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
    
    # stage == 2
    # restore
    K_scale_ptr = original_K_scale_ptr
    V_ptrs = original_V_ptrs
    K_ptrs = original_K_ptrs
    
    # begin stage == 2
    lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    lo = tl.multiple_of(lo, BLOCK_M)
    K_scale_ptr += lo // BLOCK_N
    K_ptrs += stride_kn * lo
    V_ptrs += stride_vn * lo

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)   
        k = tl.load(K_ptrs, mask = k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale 

        # stage == 2 branch
        mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = qk + tl.where(mask, 0, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        # stage == 2 branch end
        
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        
        v = tl.load(V_ptrs, mask = offs_n[:, None] < (kv_len - start_n))
        p = p.to(tl.float16)
        
        acc += tl.dot(p, v, out_dtype=tl.float16)   
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vn
        
    # fused zone end
    
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * h_qo + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask = (offs_m < qo_len))
        

def sageattn_forward_causal_true(q, k, v, 
                                  q_scale, k_scale, 
                                  output_dtype="float16",
                                  tensor_layout="HND", 
                                  return_lse=False):
    """
    [params]
        q: paddle.Tensor, dtype in int8, q tensor after quant.
        k: paddle.Tensor, dtype in int8, k tensor after quant.
        v: paddle.Tensor, dtype in fp16 or bf16, v tensor.
        q_scale: paddle.Tensor, dtype in fp16 or bf16, this is the output tensor for scale factor, from quant kernel.
        k_scale: paddle.Tensor, dtype in fp16 or bf16, this is the output tensor for scale factor, from quant kernel.
        output_dtype: string. Only in ['float16', 'bfloat16']. The datatype of q, k, v tensor.
        tensor_layout: string. Only in ['HND', 'NHD'], 'HND' -> [bsz, num_heads, seq_len, head_dim],
                        'HND' -> [bsz, seq_len, num_heads, head_dim]
        return_lse: bool. Return lse correction or not. Useful in parallel computing. Default False.
    [Examples]
        batch_size = 2
        num_heads = 24
        seq_len = 1376
        head_dim = 64
        
        sm_scale = 1.0 / (head_dim_og ** 0.5)
        
        # note: this layout is 'NHD'
        tensor_layout = 'NHD'
        q = paddle.randn(shape=(batch_size, seq_len, num_heads, head_dim), dtype="float16")
        k = paddle.randn(shape=(batch_size, seq_len, num_heads, head_dim), dtype="float16")
        v = paddle.randn(shape=(batch_size, seq_len, num_heads, head_dim), dtype="float16")
        
        km = paddle.mean(k, axis=seq_dim, keepdim=True)
        
        q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout)
        o, lse = sageattn_forward_causal_true(q_int8, k_int8, v, q_scale, k_scale, 
                                                output_dtype="float16", tensor_layout=tensor_layout)
    """
    assert output_dtype in ["float16", "bfloat16"]
    
    Out = paddle.empty(q.shape, dtype=output_dtype)
    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_qz, stride_qh, stride_qn = h_qo * qo_len * head_dim, qo_len * head_dim, head_dim
        stride_kz, stride_kh, stride_kn = h_kv * kv_len * head_dim, kv_len * head_dim, head_dim
        stride_vz, stride_vh, stride_vn = h_kv * kv_len * head_dim, kv_len * head_dim, head_dim
        stride_oz, stride_oh, stride_on = h_qo * qo_len * head_dim, qo_len * head_dim, head_dim
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        
        stride_qz, stride_qh, stride_qn = qo_len * h_qo * head_dim, head_dim, h_qo * head_dim
        stride_kz, stride_kh, stride_kn = kv_len * h_kv * head_dim, head_dim, h_kv * head_dim
        stride_vz, stride_vh, stride_vn = kv_len * h_kv * head_dim, head_dim, h_kv * head_dim
        stride_oz, stride_oh, stride_on = qo_len * h_qo * head_dim, head_dim, h_qo * head_dim
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    
    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv
    BSZ = b
    
    prepare_attr_for_triton_kernel = """
    paddle::DataType output_t;
    if (output_dtype == std::string("float16")) {
        output_t = paddle::DataType::FLOAT16;
    } else {
        output_t = paddle::DataType::BFLOAT16;
    }
    
    auto out_tensor = paddle::empty(q.shape(), output_t, q.place());
    auto q_strides = q.strides();
    auto k_strides = k.strides();
    auto v_strides = v.strides();
    auto o_strides = out_tensor.strides();
    
    int b, h_qo, qo_len, head_dim;
    int kv_len, h_kv;
    
    int stride_qz, stride_qh, stride_qn;
    int stride_kz, stride_kh, stride_kn;
    int stride_vz, stride_vh, stride_vn;
    int stride_oz, stride_oh, stride_on;
    
    if (tensor_layout == "HND") {
        b = q.shape()[0];
        h_qo = q.shape()[1];
        qo_len = q.shape()[2];
        head_dim = q.shape()[3];
        
        h_kv = k.shape()[1];
        kv_len = k.shape()[2];
        
        stride_qz = q_strides[0];
        stride_qh = q_strides[1];
        stride_qn = q_strides[2];
        
        stride_kz = k_strides[0];
        stride_kh = k_strides[1];
        stride_kn = k_strides[2];
        
        stride_vz = v_strides[0];
        stride_vh = v_strides[1];
        stride_vn = v_strides[2];
        
        stride_oz = o_strides[0];
        stride_oh = o_strides[1];
        stride_on = o_strides[2];
    } else if (tensor_layout == "NHD") {
        b = q.shape()[0];
        qo_len = q.shape()[1];   // reverse
        h_qo = q.shape()[2];
        head_dim = q.shape()[3];
        
        kv_len = k.shape()[1];   // reverse
        h_kv = k.shape()[2];
        
        stride_qz = q_strides[0];
        stride_qh = q_strides[2];       // reverse
        stride_qn = q_strides[1];
        
        stride_kz = k_strides[0];
        stride_kh = k_strides[2];       // reverse
        stride_kn = k_strides[1];
        
        stride_vz = v_strides[0];
        stride_vh = v_strides[2];       // reverse
        stride_vn = v_strides[1];
        
        stride_oz = o_strides[0];
        stride_oh = o_strides[2];       // reverse
        stride_on = o_strides[1];
    } else {
        throw std::runtime_error("Unsupported tensor layout");
    }
    
    int HEAD_DIM_K = head_dim;
    int num_kv_groups = h_qo / h_kv;
    int BSZ = b;
"""

    op_name = "triton_sageattn_attn_fwd_causal_true"
    op_name += get_dtype_str(q.dtype)
    op_name += f"_BSZ{BSZ}_seq{qo_len}_h{h_qo}_dim{HEAD_DIM_K}"
    
    sageattn_attn_fwd_causal_true_config = []
    if head_dim == 64:
        sageattn_attn_fwd_causal_true_config.append({
            "num_warps": 4,
            "num_stages": 4
        })
    else:
        sageattn_attn_fwd_causal_true_config.append({
            "num_warps": 8,
            "num_stages": 4
        })
    
    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        if return_lse:
            Lse = paddle.empty((b, h_qo, qo_len), dtype=paddle.float32)
        else:
            Lse = paddle.empty((0, 0, 0), dtype=paddle.float32)
        prepare_ptr_for_triton_kernel = """
        paddle::Tensor lse_tensor;
        
        if (return_lse) {
            lse_tensor = paddle::empty({b, h_qo, qo_len}, q.dtype(), q.place());
        } else {
            lse_tensor = paddle::empty({1,1,1}, paddle::DataType::FLOAT32, paddle::CPUPlace());
        }

        CUdeviceptr input_ptrs[7] = {
            get_tensor_ptr(q),
            get_tensor_ptr(k),
            get_tensor_ptr(v),
            get_tensor_ptr(q_scale),
            get_tensor_ptr(k_scale),
            get_tensor_ptr(out_tensor),
            get_tensor_ptr(lse_tensor)
        };
    """
        return_tensor_names = "out_tensor, lse_tensor"
        template_used = rendering_common_template(
            sageattn_forward_causal_true, 
            prepare_attr_for_triton_kernel=prepare_attr_for_triton_kernel,
            prepare_ptr_for_triton_kernel=prepare_ptr_for_triton_kernel,
            return_tensor_names=return_tensor_names
        )
        grid = ("(qo_len + BLOCK_M - 1) / BLOCK_M", "h_qo", "BSZ")
        sageattn_attn_fwd_causal_true_kernel[(op_name, template_used, grid, sageattn_attn_fwd_causal_true_config)](
            Q=q, 
            K=k, 
            V=v, 
            Q_scale=q_scale, 
            K_scale=k_scale, 
            Out=Out, 
            Lse=Lse, 
            stride_qz=stride_qz, 
            stride_qh=stride_qh, 
            stride_qn=stride_qn,
            stride_kz=stride_kz, 
            stride_kh=stride_kh, 
            stride_kn=stride_kn,
            stride_vz=stride_vz, 
            stride_vh=stride_vh, 
            stride_vn=stride_vn,
            stride_oz=stride_oz, 
            stride_oh=stride_oh, 
            stride_on=stride_on,
            qo_len=qo_len,
            kv_len=kv_len, 
            BSZ=BSZ,
            h_qo=h_qo, 
            num_kv_groups=num_kv_groups,
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_M=128, 
            BLOCK_N=64, 
            RETURN_LSE=1 if return_lse else 0
        )
        
    if in_dynamic_or_pir_mode():
        outs = _C_ops._run_custom_op(
            op_name,
            q,k,v, q_scale, k_scale, 
            output_dtype,
            tensor_layout, 
            return_lse
        )

        return outs[0], outs[1]
    else:
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "q": q,
            "k": k,
            "v": v,
            "q_scale": q_scale,
            "k_scale": k_scale,
        }
        out_tensor = helper.create_variable_for_type_inference(dtype=Out.dtype)
        out_lse = helper.create_variable_for_type_inference(dtype=Lse.dtype)
        
        helper.append_op(
            type=op_name,
            inputs=inputs,
            attrs={
                "output_type": output_dtype,
                "tensor_layout": tensor_layout,
                "return_lse": 1 if return_lse else 0
            },
            outputs={
                "out_tensor": out_tensor,
                "lse_tensor": out_lse
            }
        )
        
        return out_tensor, out_lse


# ============== sage attention triton API =================
def sageattn_qk_int8_pv_fp16_triton(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    quant_gran: str="per_block",
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    return_lse: bool = False,
) -> paddle.Tensor:
    """
    Examples:
        batch_size = 2
        num_heads = 24
        seq_len = 1376
        head_dim = 64
        q = paddle.randn(shape=(batch_size, seq_len, num_heads, head_dim), dtype="float16")
        k = paddle.randn(shape=(batch_size, seq_len, num_heads, head_dim), dtype="float16")
        v = paddle.randn(shape=(batch_size, seq_len, num_heads, head_dim), dtype="float16")
        sm_scale = 1 / (head_dim ** 0.5)
        
        o = paddlemix.triton_ops.sageattn_qk_int8_pv_fp16_triton(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=sm_scale, smooth_k=True, return_lse=False)
    """
    dtype = q.dtype
    assert dtype in [paddle.float16, paddle.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert str(q.place) == str(k.place) == str(v.place), f"All tensors must be on the same device. Got q: {q.place}, k: {k.place}, v: {v.place}"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    head_dim_og = q.shape[-1]
    # if not 64 or 128, then fill to 64 or 128
    if head_dim_og < 64:
        q = paddle.nn.functional.pad(q, pad=[0, 64-head_dim_og])
        k = paddle.nn.functional.pad(k, pad=[0, 64-head_dim_og])
        v = paddle.nn.functional.pad(v, pad=[0, 64-head_dim_og])
    elif head_dim_og > 64 and head_dim_og < 128:
        q = paddle.nn.functional.pad(q, pad=[0, 128-head_dim_og])
        k = paddle.nn.functional.pad(k, pad=[0, 128-head_dim_og])
        v = paddle.nn.functional.pad(v, pad=[0, 128-head_dim_og])
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")
    
    seq_dim = 1 if tensor_layout == "NHD" else 2
    
    if smooth_k:
        km = paddle.mean(k, axis=seq_dim, keepdim=True)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = paddle.matmul(paddle.transpose(q, [0, 2, 1, 3]), paddle.squeeze(paddle.transpose(q, [0, 2, 3, 1]), axis=-1)).astype(paddle.float32)
            else:
                lse_correction = paddle.matmul(q, paddle.squeeze(paddle.transpose(km, [0, 1, 3, 2]), axis=-1)).astype(paddle.float32)
    else:
        km = None
        
    if dtype == paddle.bfloat16 or dtype == paddle.float32:
        v = paddle.cast(v, dtype=paddle.float16)
        
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)
    
    if quant_gran == "per_block":
        q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout)

    if is_causal:
        o, lse = sageattn_forward_causal_true(q_int8, k_int8, v, q_scale, k_scale, output_dtype="float16", tensor_layout=tensor_layout, return_lse=return_lse)
    else:
        o, lse = sageattn_forward_causal_false(q_int8, k_int8, v, q_scale, k_scale, output_dtype="float16", tensor_layout=tensor_layout, return_lse=return_lse)
    
    o = o[..., :head_dim_og]
    
    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o
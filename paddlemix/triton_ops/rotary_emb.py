


import paddle
import triton
import triton.language as tl
from paddle import _C_ops
from paddle.base.framework import OpProtoHolder
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode

from .triton_utils import get_dtype_str, paddle_use_triton, rendering_common_template


@paddle_use_triton(
    key=["1"],
)
def apply_rotary_emb_kernel(
    q_ptr,  
    k_ptr,
    cos_ptr,
    sin_ptr,
    outq_ptr,
    outk_ptr, 
    batch,
    num_heads,
    seq_len,
    head_dim,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 计算当前线程处理的元素范围
    b_pid = tl.program_id(axis=0)  # grid内哪个Block
    h_pid = tl.program_id(axis=1)
    s_pid = tl.program_id(axis=2)
    
    block_start = b_pid * num_heads * seq_len * head_dim + h_pid * seq_len * head_dim + s_pid * head_dim 
    read_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = read_offsets < n_elements
    even_mask = tl.arange(0, BLOCK_SIZE) % 2 == 0
    q0 = tl.load(q_ptr + read_offsets, mask=mask & even_mask) #0，2，4，6，8
    q1 = tl.load(q_ptr + read_offsets + 1, mask=mask & even_mask) #1，3，5，7，9
    
    k0 = tl.load(k_ptr + read_offsets, mask=mask & even_mask) #0，2，4，6，8
    k1 = tl.load(k_ptr + read_offsets + 1, mask=mask & even_mask) #1，3，5，7，9
    
    
    # 加载 cos 和 sin 
    block_cs_start = s_pid * head_dim
    read_cs_offsets = block_cs_start + tl.arange(0, BLOCK_SIZE)
    cs_mask = read_cs_offsets < (seq_len * head_dim)
    cos0 = tl.load(cos_ptr + read_cs_offsets, mask=cs_mask & even_mask)#0，2，4，6，8
    cos1 = tl.load(cos_ptr + read_cs_offsets + 1, mask=cs_mask & even_mask)#1，3，5，7，9
    sin0 = tl.load(sin_ptr + read_cs_offsets, mask=cs_mask & even_mask)#0，2，4，6，8
    sin1 = tl.load(sin_ptr + read_cs_offsets + 1, mask=cs_mask & even_mask)#1，3，5，7，9


    oq0 = tl.cast(tl.cast(q0, tl.float32) * cos0 - tl.cast(q1, tl.float32) * sin0,tl.float16)
    oq1 = tl.cast(tl.cast(q1, tl.float32) * cos1 + tl.cast(q0, tl.float32) * sin1,tl.float16)
    
    ok0 = tl.cast(tl.cast(k0, tl.float32) * cos0 - tl.cast(k1, tl.float32) * sin0,tl.float16)
    ok1 = tl.cast(tl.cast(k1, tl.float32) * cos1 + tl.cast(k0, tl.float32) * sin1,tl.float16)
    
    # 将结果存储到全局内存
    tl.store(outq_ptr + read_offsets, oq0, mask=mask & even_mask)
    tl.store(outq_ptr + read_offsets + 1, oq1, mask=mask & even_mask)
    tl.store(outk_ptr + read_offsets, ok0, mask=mask & even_mask)
    tl.store(outk_ptr + read_offsets + 1, ok1, mask=mask & even_mask)
    
    

def apply_rotary_emb_triton(
    q,
    k,
    cos,
    sin,
):
    batch = q.shape[0]
    num_heads = q.shape[1]
    seq_len = q.shape[2]
    head_dim = q.shape[3]
    n_elements = batch * num_heads * seq_len * head_dim
    
    prepare_attr_for_triton_kernel = """
    // 这个名字必须保证和kernel形式参数一致！
    int batch = q.dims()[0];
    int num_heads = q.dims()[1];
    int seq_len =  q.dims()[2];
    int head_dim =  q.dims()[3];
    int n_elements = batch * num_heads * seq_len * head_dim;
    """
    
    
    assert head_dim == 64, "wdfdfref"
    BLOCK_SIZE = head_dim
    op_name = "apply_rotary_emb_triton"
    op_name += get_dtype_str(q.dtype)
    op_name += f"_{BLOCK_SIZE}"
    # 创建输出张量
    
    # apply_rotary_emb_kernel_config = [
    #     {"num_warps": 2},
    #     {"num_warps": 4},
    #     {"num_warps": 8},
    #     {"num_warps": 16},
    #     {"num_warps": 32},
    # ]
    if op_name not in OpProtoHolder.instance().op_proto_map.keys():
        outq = paddle.empty_like(q)
        outk = paddle.empty_like(k)
        
        prepare_ptr_for_triton_kernel = """
        // 这个名字必须保证和kernel形式参数一致！
        auto q_ptr = get_tensor_ptr(q);
        auto k_ptr = get_tensor_ptr(k);
        auto cos_ptr = get_tensor_ptr(cos);
        auto sin_ptr = get_tensor_ptr(sin);

        auto out0_tensor = paddle::empty(q.shape(), q.dtype(), q.place());
        auto out1_tensor = paddle::empty(k.shape(), k.dtype(), k.place());
        auto outq_ptr = get_tensor_ptr(out0_tensor);
        auto outk_ptr = get_tensor_ptr(out1_tensor);
        """
        return_tensor_names = "out0_tensor, out1_tensor"
        
        template_used = rendering_common_template(
            apply_rotary_emb_triton, prepare_attr_for_triton_kernel, prepare_ptr_for_triton_kernel, return_tensor_names
        )

        
        grid = ("batch","num_heads","seq_len")
        apply_rotary_emb_kernel[(op_name,template_used, grid)](
            q_ptr = q,  
            k_ptr = k,
            cos_ptr = cos,
            sin_ptr = sin,
            outq_ptr = outq,
            outk_ptr = outk, 
            batch = batch,
            num_heads = num_heads,
            seq_len = seq_len,
            head_dim = head_dim,
            n_elements = n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    if in_dynamic_or_pir_mode():
        #print(f"== we are in dynamic mode, op_name: {op_name}")
        outs = _C_ops._run_custom_op(
            op_name,
            q,
            k,
            cos,
            sin,
        )
        return outs[0],outs[1]
    else:
        #print(f"== we are in dynamic to static mode, op_name: {op_name}")
        helper = LayerHelper(op_name, **locals())
        inputs = {
            "q": q,
            "k": k,
            "cos": cos,
            "sin": sin,
        }
        outq = helper.create_variable_for_type_inference(dtype=q.dtype)
        outk = helper.create_variable_for_type_inference(dtype=q.dtype)

        helper.append_op(
            type=op_name,
            inputs=inputs,
            outputs={"out0_tensor": outq,"out1_tensor": outk},
        )
        return outq,outk



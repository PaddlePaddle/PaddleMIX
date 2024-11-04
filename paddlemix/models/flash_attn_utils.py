from ppdiffusers.utils import logging

logger = logging.get_logger(__name__)

# test if flash attention is available
def is_flash_attn_available():
    try:
        import paddle
        if "npu" in paddle.get_device(): # NOTE: flash attn has not been tested yet
            return False
        q = paddle.rand((1, 4, 2, 8)).astype('bfloat16') 
        output = paddle.nn.functional.flash_attention.flash_attention(q, q, q, 0.9, False, False)
        return True
    except:
        return False

HAS_FLASH_ATTN = is_flash_attn_available()


def has_flash_attn_func():
    if HAS_FLASH_ATTN:
        try:
            from paddle.nn.functional.flash_attention import flash_attention as flash_attn_func
            from paddle.nn.functional.flash_attention import flash_attn_unpadded as flash_attn_varlen_func
            return flash_attn_func, flash_attn_varlen_func
        except:
            return None, None
    else:
        return None, None

def create_attention_module(config, module_type, layer_idx=None):
    if has_flash_attn_func()[0] is not None:
        if module_type == "qwen2vl":
            from paddlemix.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLFlashAttention2
            return Qwen2VLFlashAttention2(config, layer_idx)
        elif module_type == "vision":
            from paddlemix.models.qwen2_vl.modeling_qwen2_vl import VisionFlashAttention2
            return VisionFlashAttention2(config.embed_dim, num_heads=config.num_heads)
    else:
        logger.warning(f'Warning: Flash Attention2 is not available for {module_type}, fallback to normal attention.')
        
    if module_type == "qwen2vl":
        from paddlemix.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention
        return Qwen2VLAttention(config, layer_idx)
    elif module_type == "vision":
        from paddlemix.models.qwen2_vl.modeling_qwen2_vl import VisionAttention
        return VisionAttention(config.embed_dim, num_heads=config.num_heads)
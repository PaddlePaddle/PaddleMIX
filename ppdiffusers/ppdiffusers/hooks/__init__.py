from ..utils import is_paddle_available

if is_paddle_available():
    from .hooks import HookRegistry, ModelHook
    from .layerwise_casting import apply_layerwise_casting, apply_layerwise_casting_hook
    from .pyramid_attention_broadcast import PyramidAttentionBroadcastConfig, apply_pyramid_attention_broadcast

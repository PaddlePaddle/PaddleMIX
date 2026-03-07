import numpy as np
import random
import os
import logging.config

import paddle
import paddle.nn as nn
from typing import Callable, Dict, Any, Optional

def apply_func_to_submodules(
    module: paddle.nn.Layer,
    class_type: type,
    function: Callable,
    parent_name: str = "",
    return_d: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """
    Recursively iterate through direct submodules of a Paddle `Layer` and apply `function`
    when a submodule is instance of `class_type`.

    Args:
        module (paddle.nn.Layer): the root module to traverse.
        class_type (type): target class type to match.
        function (callable): function to call for matched submodules. Called as function(submodule, **kwargs).
        parent_name (str): name of parent module for building full name.
        return_d (dict|None): optional dict to collect function return values keyed by full_name.
        **kwargs: extra keyword args forwarded to `function`. Keys `'name'`, `'full_name'`, and
                  `'parent_module'` will be added/overwritten for each call.
    Returns:
        return_d if provided, else None.
    """
    # Prefer to use the internal _sub_layers dict to get immediate children (name -> layer)
    # Fallback: try to use named_sublayers() but note it yields nested sublayers recursively.
    sub_items = None
    if hasattr(module, "_sub_layers"):
        # _sub_layers is an OrderedDict mapping local_name -> Layer
        sub_items = list(module._sub_layers.items())
    else:
        # Fallback (may include nested children): try to get immediate pairs from named_sublayers
        try:
            # named_sublayers yields (name, layer) but may be recursive; we keep as fallback
            sub_items = list(module.named_sublayers())
        except Exception:
            sub_items = []

    for name, submodule in sub_items:
        full_name = f"{parent_name}.{name}" if parent_name else name

        # copy kwargs to avoid mutating caller's dict across recursion
        local_kwargs = dict(kwargs)
        # pass contextual info
        local_kwargs['name'] = name
        local_kwargs['full_name'] = full_name
        local_kwargs['parent_module'] = module

        if isinstance(submodule, class_type):
            if return_d is not None:
                return_d[full_name] = function(submodule, **local_kwargs)
            else:
                function(submodule, **local_kwargs)

        # Recurse into the submodule's children
        apply_func_to_submodules(submodule, class_type, function, full_name, return_d, **local_kwargs)

    if return_d is not None:
        return return_d
        
class StraightThrough(nn.Layer):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def seed_everything(seed=42):
    """
    固定 PaddlePaddle 的随机数种子，以确保实验可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 固定 paddle 的随机种子
    paddle.seed(seed)

    # Paddle 目前不像 torch 那样有 cudnn.deterministic / benchmark
    # 但我们可以设置确定性计算选项，减少浮动
    paddle.framework.random._manual_program_seed(seed)
    # 如果使用动态图，这个函数也能确保算子初始化一致
    if paddle.get_device().startswith("gpu"):
        print(f"[Info] Using GPU with fixed seed {seed}")
    else:
        print(f"[Info] Using {paddle.get_device()} with fixed seed {seed}")

def setup_logging(log_file):
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'DEBUG',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'standard',
                'filename': log_file,
                'mode': 'a',
            }
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': True
            }
        }
    }
    logging.config.dictConfig(logging_config)

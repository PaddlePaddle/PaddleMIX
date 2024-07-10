import random

import numpy as np
import paddle


def set_seeds(seed_list, device=None):
    if isinstance(seed_list, (tuple, list)):
        seed = sum(seed_list)
    else:
        seed = seed_list
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    return paddle.Generator().manual_seed(seed)


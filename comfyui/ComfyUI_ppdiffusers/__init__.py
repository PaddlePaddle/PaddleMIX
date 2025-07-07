# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    import paddle
except ImportError:
    print("Paddle is not installed. Please install it to use this node.")
    __all__ = []
else:
    from .basic_nodes import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_BASIC
    from .basic_nodes import (
        NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_BASIC,
    )
    from .flux_pipe_nodes import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_FLUX
    from .flux_pipe_nodes import (
        NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_FLUX,
    )
    from .sd_pipe_nodes import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_SD
    from .sd_pipe_nodes import (
        NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_SD,
    )
    from .sdxl_pipe_nodes import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_SDXL
    from .sdxl_pipe_nodes import (
        NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_SDXL,
    )
    from .wan_pipe_nodes import NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_WAN
    from .wan_pipe_nodes import (
        NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_WAN,
    )

    NODE_CLASS_MAPPINGS = {
        **NODE_CLASS_MAPPINGS_BASIC,
        **NODE_CLASS_MAPPINGS_SD,
        **NODE_CLASS_MAPPINGS_SDXL,
        **NODE_CLASS_MAPPINGS_FLUX,
        **NODE_CLASS_MAPPINGS_WAN,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        **NODE_DISPLAY_NAME_MAPPINGS_BASIC,
        **NODE_DISPLAY_NAME_MAPPINGS_SD,
        **NODE_DISPLAY_NAME_MAPPINGS_SDXL,
        **NODE_DISPLAY_NAME_MAPPINGS_FLUX,
        **NODE_DISPLAY_NAME_MAPPINGS_WAN,
    }
    __all__ = [
        "NODE_CLASS_MAPPINGS",
        "NODE_DISPLAY_NAME_MAPPINGS",
    ]

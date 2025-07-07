# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from ..utils.logging import get_logger

logger = get_logger(__name__)


class CacheMixin:
    """
    A class for enable/disabling caching techniques on diffusion models.

    Supported caching techniques:
        - [Pyramid Attention Broadcast](https://huggingface.co/papers/2408.12588)
        - SortTaylor optimization
        - TeaBlockCache + Taylor optimization
    """

    _cache_config = None

    @property
    def is_cache_enabled(self) -> bool:
        return self._cache_config is not None

    def enable_cache(self, config) -> None:
        """
        Enable caching techniques on the model.

        Args:
            config (`Union[PyramidAttentionBroadcastConfig, SortBlockConfig, TeaBlockCacheTaylorConfig]`):
                The configuration for applying the caching technique. Currently supported caching techniques are:
                    - [`~hooks.PyramidAttentionBroadcastConfig`]
                    - [`~hooks.SortBlockConfig`]
                    - [`~hooks.TeaBlockCacheTaylorConfig`]

        Example:

        ```python
        >>> import paddle
        >>> from ppdiffusers import CogVideoXPipeline, PyramidAttentionBroadcastConfig

        >>> pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", paddle_dtype=paddle.bfloat16)

        >>> config = PyramidAttentionBroadcastConfig(
        ...     spatial_attention_block_skip_range=2,
        ...     spatial_attention_timestep_skip_range=(100, 800),
        ...     current_timestep_callback=lambda: pipe.current_timestep,
        ... )
        >>> pipe.transformer.enable_cache(config)

        >>> # Or for SortBlock optimization:
        >>> from ppdiffusers import FluxPipeline, SortBlockConfig
        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
        >>> config = SortBlockConfig(
        ...     timestep_start=900,
        ...     timestep_end=100,
        ...     beta=0.3,
        ...     current_timestep_callback=lambda: pipe._current_timestep,
        ... )
        >>> pipe.transformer.enable_cache(config)

        >>> # Or for TeaBlockCache + Taylor optimization:
        >>> from ppdiffusers import FluxPipeline, TeaBlockCacheTaylorConfig
        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16)
        >>> config = TeaBlockCacheTaylorConfig(
        ...     step_start=50,
        ...     step_end=950,
        ...     block_cache_start=1,
        ...     single_block_cache_start=1,
        ...     taylor_max_order=1,
        ...     taylor_first_enhance=1,
        ...     current_timestep_callback=lambda: pipe._current_timestep,
        ... )
        >>> pipe.transformer.enable_cache(config)
        ```
        """
        from ..hooks import (
            PyramidAttentionBroadcastConfig,
            SortBlockConfig,
            TeaBlockCacheTaylorConfig,
            apply_pyramid_attention_broadcast,
            apply_sort_block,
            apply_teablockcache_taylor,
        )

        if isinstance(config, PyramidAttentionBroadcastConfig):
            apply_pyramid_attention_broadcast(self, config)
        elif isinstance(config, SortBlockConfig):
            apply_sort_block(self, config)
        elif isinstance(config, TeaBlockCacheTaylorConfig):
            apply_teablockcache_taylor(self, config)
        else:
            raise ValueError(f"Cache config {type(config)} is not supported.")
        self._cache_config = config

    def disable_cache(self) -> None:
        from ..hooks import (
            HookRegistry,
            PyramidAttentionBroadcastConfig,
            SortBlockConfig,
            TeaBlockCacheTaylorConfig,
        )

        if self._cache_config is None:
            logger.warning("Caching techniques have not been enabled, so there's nothing to disable.")
            return
        if isinstance(self._cache_config, PyramidAttentionBroadcastConfig):
            registry = HookRegistry.check_if_exists_or_initialize(self)
            registry.remove_hook("pyramid_attention_broadcast", recurse=True)
        elif isinstance(self._cache_config, SortBlockConfig):
            registry = HookRegistry.check_if_exists_or_initialize(self)
            registry.remove_hook("sort_block", recurse=True)
        elif isinstance(self._cache_config, TeaBlockCacheTaylorConfig):
            registry = HookRegistry.check_if_exists_or_initialize(self)
            registry.remove_hook("teablockcache_taylor", recurse=True)
        else:
            raise ValueError(f"Cache config {type(self._cache_config)} is not supported.")
        self._cache_config = None

    def _reset_stateful_cache(self, recurse: bool = True) -> None:
        from ..hooks import HookRegistry

        HookRegistry.check_if_exists_or_initialize(self).reset_stateful_hooks(recurse=recurse)

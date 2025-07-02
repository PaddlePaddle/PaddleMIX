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

from contextlib import contextmanager

from paddle.distributed import fleet

try:
    import transformer_engine.paddle as te
    from transformer_engine.common.recipe import DelayedScaling, Format

    _IS_TRANSFORMER_ENGINE_INSTALLED = True

except ModuleNotFoundError:
    _IS_TRANSFORMER_ENGINE_INSTALLED = False


class TransformerEngineHelper:
    @staticmethod
    def is_installed():
        return _IS_TRANSFORMER_ENGINE_INSTALLED

    @staticmethod
    def get_te():
        assert (
            TransformerEngineHelper.is_installed()
        ), "TransformerEngine is not installed. Please install it first or disable it."
        return te

    @staticmethod
    def get_te_recompute_func():
        assert (
            TransformerEngineHelper.is_installed()
        ), "TransformerEngine is not installed. Please install it first or disable it."
        return te.recompute

    @staticmethod
    def get_fp8_group():
        assert (
            TransformerEngineHelper.is_installed()
        ), "TransformerEngine is not installed. Please install it first or disable it."

        try:
            hcg = fleet.get_hybrid_communicate_group()
        except:
            return None
        use_pp = hcg.get_pipe_parallel_world_size() > 1
        if not use_pp:
            return None

        dp_group = hcg.get_data_parallel_group()
        tp_group = hcg.get_model_parallel_group()
        if dp_group.nranks <= 1:
            return tp_group
        if tp_group.nranks <= 1:
            return dp_group

        return dp_group

    @staticmethod
    @contextmanager
    def fp8_autocast(enabled=False, fp8_group=None, fp8_amax_history_len=1024, fp8_amax_compute_algo="max"):
        if TransformerEngineHelper.is_installed():
            fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
            fp8_recipe = DelayedScaling(
                fp8_format=fp8_format, amax_history_len=fp8_amax_history_len, amax_compute_algo=fp8_amax_compute_algo
            )
            with te.fp8_autocast(enabled=enabled, fp8_group=fp8_group, fp8_recipe=fp8_recipe):
                yield
        else:  # null context
            yield

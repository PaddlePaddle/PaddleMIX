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

import importlib.util
import os


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def is_gradio_available():
    return _is_package_available("gradio")


def use_modelscope() -> bool:
    return os.environ.get("USE_MODELSCOPE_HUB", "0").lower() in ["true", "1"]


def use_openmind() -> bool:
    return os.environ.get("USE_OPENMIND_HUB", "0").lower() in ["true", "1"]


def is_pyav_available():
    return _is_package_available("av")


def is_pillow_available():
    return _is_package_available("PIL")


def is_matplotlib_available():
    return _is_package_available("matplotlib")


# version

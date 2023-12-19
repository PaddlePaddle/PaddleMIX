# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import os

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE, hf_cache_home


def str2bool(v):
    if isinstance(v, bool):
        return v
    if not isinstance(v, str):
        v = str(v)
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Not supported value: {}".format(v))

# make sure we have abs path
ppnlp_cache_home = os.path.abspath(os.path.expanduser(
    os.getenv("PPNLP_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "paddlenlp"))
))
ppdiffusers_default_cache_path = os.path.abspath(os.path.join(ppnlp_cache_home, "ppdiffusers"))
diffusers_default_cache_path = os.path.abspath(HUGGINGFACE_HUB_CACHE)

CONFIG_NAME = "config.json"
TORCH_WEIGHTS_NAME = "diffusion_pytorch_model.bin"
TORCH_SAFETENSORS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
FLAX_WEIGHTS_NAME = "diffusion_flax_model.msgpack"
ONNX_WEIGHTS_NAME = "model.onnx"
ONNX_EXTERNAL_WEIGHTS_NAME = "weights.pb"

HUGGINGFACE_CO_RESOLVE_ENDPOINT = "https://huggingface.co"
PPDIFFUSERS_CACHE = ppdiffusers_default_cache_path
DIFFUSERS_CACHE = diffusers_default_cache_path
DIFFUSERS_DYNAMIC_MODULE_NAME = "diffusers_modules"
PPDIFFUSERS_DYNAMIC_MODULE_NAME = "ppdiffusers_modules"
# make sure we have abs path
HF_MODULES_CACHE = os.path.abspath(os.getenv("HF_MODULES_CACHE", os.path.join(hf_cache_home, "modules")))
PPDIFFUSERS_MODULES_CACHE = os.path.abspath(os.getenv("PPDIFFUSERS_MODULES_CACHE", os.path.join(ppnlp_cache_home, "modules")))

PADDLE_WEIGHTS_NAME = "model_state.pdparams"
FASTDEPLOY_WEIGHTS_NAME = "inference.pdiparams"
FASTDEPLOY_MODEL_NAME = "inference.pdmodel"
WEIGHTS_NAME = PADDLE_WEIGHTS_NAME

TEST_DOWNLOAD_SERVER = "https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/tests"
DOWNLOAD_SERVER = "https://bj.bcebos.com/paddlenlp/models/community"
PPNLP_BOS_RESOLVE_ENDPOINT = os.getenv("PPNLP_ENDPOINT", "https://bj.bcebos.com/paddlenlp")
DEPRECATED_REVISION_ARGS = ["fp16", "non-ema"]
TEXT_ENCODER_ATTN_MODULE = ".self_attn"
LOW_CPU_MEM_USAGE_DEFAULT = str2bool(os.getenv("LOW_CPU_MEM_USAGE_DEFAULT", False))

NEG_INF = -1e4

get_map_location_default = lambda *args, **kwargs: os.getenv("MAP_LOCATION_DEFAULT", "cpu")
FROM_HF_HUB = str2bool(os.getenv("FROM_HF_HUB", False))
FROM_DIFFUSERS = str2bool(os.getenv("FROM_DIFFUSERS", False))
TO_DIFFUSERS = str2bool(os.getenv("TO_DIFFUSERS", False))

# FOR tests
if bool(os.getenv("PATCH_ALLCLOSE", False)):
    import paddle
    import numpy
    from pprint import pprint
    numpy.set_printoptions(precision=4)

    paddle_raw_all_close = paddle.allclose
    
    def allclose_pd(x, y, rtol=1e-05, atol=1e-08, equal_nan=False, name=None):
        pprint(x.numpy())
        pprint(y.numpy())
        return paddle_raw_all_close(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan, name=name)
    paddle.allclose = allclose_pd

    numpy_raw_all_close = numpy.allclose
    def allclose_np(a, b, rtol=1e-05, atol=1e-08, equal_nan=False, ):
        pprint(a)
        pprint(b)
        return numpy_raw_all_close(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    numpy.allclose = allclose_np
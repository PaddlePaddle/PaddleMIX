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

import importlib

VOCODERS = {}


def register_vocoder(cls):
    VOCODERS[cls.__name__.lower()] = cls
    VOCODERS[cls.__name__] = cls
    return cls


def get_vocoder_cls(hparams):
    if hparams["vocoder"] in VOCODERS:
        return VOCODERS[hparams["vocoder"]]
    else:
        vocoder_cls = hparams["vocoder"]
        pkg = ".".join(vocoder_cls.split(".")[:-1])
        cls_name = vocoder_cls.split(".")[-1]
        vocoder_cls = getattr(importlib.import_module(pkg), cls_name)
        return vocoder_cls

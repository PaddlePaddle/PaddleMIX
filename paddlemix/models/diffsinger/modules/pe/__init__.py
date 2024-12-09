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

from utils import hparams

from .pm import ParselmouthPE
from .pw import HarvestPE
from .rmvpe import RMVPE


def initialize_pe():
    pe = hparams["pe"]
    pe_ckpt = hparams["pe_ckpt"]
    if pe == "parselmouth":
        return ParselmouthPE()
    elif pe == "rmvpe":
        return RMVPE(pe_ckpt)
    elif pe == "harvest":
        return HarvestPE()
    else:
        raise ValueError(f" [x] Unknown f0 extractor: {pe}")

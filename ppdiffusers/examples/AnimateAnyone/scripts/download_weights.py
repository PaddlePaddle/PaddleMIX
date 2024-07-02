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

import os
from pathlib import Path

from ppdiffusers.utils import bos_aistudio_hf_download


def load_weight():
    print("Preparing AnimateAnyone pretrained weights...")
    local_dir = "./pretrained_weights"
    pretrained_model_name_or_path = "tsaiyue/AnimateAnyone_PD"
    os.makedirs(local_dir, exist_ok=True)
    for file_name in [
        "config.json",
        "denoising_unet.pdparams",
        "motion_module_stage2.pdparams",
        "pose_guider.pdparams",
        "reference_unet.pdparams",
        "control_v11p_sd15_openpose.pdparams",
        "animatediff_mm_sd_v15_v2.pdparams",
        "denoising_unet_initial4stage1.pdparams",
    ]:
        path = Path(file_name)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue

        bos_aistudio_hf_download(pretrained_model_name_or_path, weights_name=file_name, cache_dir=local_dir)


if __name__ == "__main__":
    load_weight()

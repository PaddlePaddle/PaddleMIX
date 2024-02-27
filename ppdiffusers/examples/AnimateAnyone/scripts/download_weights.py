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

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download


def prepare_base_model():
    print("Preparing base stable-diffusion-v1-5 weights...")
    local_dir = "./pretrained_weights/stable-diffusion-v1-5"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in ["unet/config.json", "unet/diffusion_pytorch_model.bin"]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue
        hf_hub_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )


def prepare_anyone():
    print("Preparing AnimateAnyone weights...")
    local_dir = "./pretrained_weights"
    os.makedirs(local_dir, exist_ok=True)
    for hub_file in [
        "denoising_unet.pth",
        "motion_module.pth",
        "pose_guider.pth",
        "reference_unet.pth",
    ]:
        path = Path(hub_file)
        saved_path = local_dir / path
        if os.path.exists(saved_path):
            continue

        hf_hub_download(
            repo_id="patrolli/AnimateAnyone",
            subfolder=PurePosixPath(path.parent),
            filename=PurePosixPath(path.name),
            local_dir=local_dir,
        )


if __name__ == "__main__":
    prepare_base_model()
    prepare_anyone()

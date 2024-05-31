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

import json
import os
from pathlib import Path

import paddle
import safetensors
import wandb


def create_folder_if_necessary(path):
    path = "/".join(path.split("/")[:-1])
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_save(ckpt, path):
    try:
        os.remove(f"{path}.bak")
    except OSError:
        pass
    try:
        os.rename(path, f"{path}.bak")
    except OSError:
        pass
    if path.endswith(".pdparams"):
        paddle.save(obj=ckpt, path=path)
    elif path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ckpt, f, indent=4)
    elif path.endswith(".safetensors"):
        raise NotImplementedError("Automatic saving to safetensors is not supported yet")
    else:
        raise ValueError(f"File extension not supported: {path}")


def load_or_fail(path, wandb_run_id=None):
    accepted_extensions = [".pt", ".ckpt", ".json", ".safetensors", ".pdparams"]
    try:
        assert any(
            [path.endswith(ext) for ext in accepted_extensions]
        ), f"Automatic loading not supported for this extension: {path}"
        if not os.path.exists(path):
            checkpoint = None
        elif path.endswith(".pt") or path.endswith(".ckpt"):
            checkpoint = paddle.load(path=path)
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
        elif path.endswith(".safetensors"):
            checkpoint = {}
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    checkpoint[key] = f.get_tensor(key)
        elif path.endswith(".pdparams"):
            checkpoint = paddle.load(path=path)
        return checkpoint
    except Exception as e:
        if wandb_run_id is not None:
            wandb.alert(
                title=f"Corrupt checkpoint for run {wandb_run_id}",
                text=f"Training {wandb_run_id} tried to load checkpoint {path} and failed",
            )
        raise e

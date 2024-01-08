# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import hashlib
import urllib
import warnings
import tqdm
import paddle
from pathlib import Path
import json
import re
from copy import deepcopy
import logging
from .model import CLAP

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"clap_amodel_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs

_RN50 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    yfcc15m="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-455df137.pt",
    cc12m="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pt",
)

_RN50_quickgelu = dict(
    openai="https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    yfcc15m="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-455df137.pt",
    cc12m="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pt",
)

_RN101 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    yfcc15m="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn101-quickgelu-yfcc15m-3e04b30e.pt",
)

_RN101_quickgelu = dict(
    openai="https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    yfcc15m="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn101-quickgelu-yfcc15m-3e04b30e.pt",
)

_RN50x4 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
)

_RN50x16 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
)

_RN50x64 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
)

_VITB32 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    laion400m_e31="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt",
    laion400m_e32="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt",
    laion400m_avg="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt",
)

_VITB32_quickgelu = dict(
    openai="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    laion400m_e31="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt",
    laion400m_e32="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt",
    laion400m_avg="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt",
)

_VITB16 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
)

_VITL14 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
)

_PRETRAINED = {
    "RN50": _RN50,
    "RN50-quickgelu": _RN50_quickgelu,
    "RN101": _RN101,
    "RN101-quickgelu": _RN101_quickgelu,
    "RN50x4": _RN50x4,
    "RN50x16": _RN50x16,
    "ViT-B-32": _VITB32,
    "ViT-B-32-quickgelu": _VITB32_quickgelu,
    "ViT-B-16": _VITB16,
    "ViT-L-14": _VITL14,
}

def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        if os.path.basename(cf)[0] == ".":
            continue  # Ignore hidden files

        with open(cf, "r") as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ("embed_dim", "audio_cfg", "text_cfg")): # only HTSAT and PANN
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {
        k: v
        for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))
    }


_rescan_model_configs()  # initial populate of model config registry

def get_pretrained_url(model: str, tag: str):
    if model not in _PRETRAINED:
        return ""
    model_pretrained = _PRETRAINED[model]
    if tag not in model_pretrained:
        return ""
    return model_pretrained[tag]

def download_pretrained(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    if "openaipublic" in url:
        expected_sha256 = url.split("/")[-2]
    else:
        expected_sha256 = ""

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if expected_sha256:
            if (
                hashlib.sha256(open(download_target, "rb").read()).hexdigest()
                == expected_sha256
            ):
                return download_target
            else:
                warnings.warn(
                    f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
                )
        else:
            return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if (
        expected_sha256
        and hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        != expected_sha256
    ):
        raise RuntimeError(
            f"Model has been downloaded but the SHA256 checksum does not not match"
        )

    return download_target

def load_state_dict(checkpoint_path: str, skip_params=True):
    checkpoint = paddle.load(checkpoint_path)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if skip_params:
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict

def create_clap_model(
    amodel_name: str,
    tmodel_name: str,
    pretrained: str = "",
    precision: str = "fp32",
    device: str = "cpu",
    jit: bool = False,
    force_quick_gelu: bool = False,
    openai_model_cache_dir: str = os.path.expanduser("~/.cache/clip"),
    skip_params=True,
    pretrained_audio: str = "",
    pretrained_text: str = "",
    enable_fusion: bool = False,
    fusion_type: str = "None"
):
    amodel_name = amodel_name.replace(
        "/", "-"
    )  # for callers using old naming with / in ViT names
    pretrained_orig = pretrained
    pretrained = pretrained.lower()

    if amodel_name in _MODEL_CONFIGS:
        logging.info(f"Loading {amodel_name} model config.")
        model_cfg = deepcopy(_MODEL_CONFIGS[amodel_name])
    else:
        logging.error(
            f"Model config for {amodel_name} not found; available models {list(_MODEL_CONFIGS.keys())}."
        )
        raise RuntimeError(f"Model config for {amodel_name} not found.")

    if force_quick_gelu:
        # override for use of QuickGELU on non-OpenAI transformer models
        model_cfg["quick_gelu"] = True

    model_cfg["text_cfg"]["model_type"] = tmodel_name
    model_cfg["enable_fusion"] = enable_fusion
    model_cfg["fusion_type"] = fusion_type
    model = CLAP(**model_cfg)

    if pretrained:
        checkpoint_path = ""
        url = get_pretrained_url(amodel_name, pretrained)
        if url:
            checkpoint_path = download_pretrained(url, root=openai_model_cache_dir)
        elif os.path.exists(pretrained_orig):
            checkpoint_path = pretrained_orig
        if checkpoint_path:
            logging.info(
                f"Loading pretrained {amodel_name}-{tmodel_name} weights ({pretrained})."
            )
            ckpt = load_state_dict(checkpoint_path, skip_params=True)
            model.load_dict(ckpt)
            param_names = [n for n, p in model.named_parameters()]
            # for n in param_names:
            #     print(n, "\t", "Loaded" if n in ckpt else "Unloaded")
        else:
            logging.warning(
                f"Pretrained weights ({pretrained}) not found for model {amodel_name}."
            )
            raise RuntimeError(
                f"Pretrained weights ({pretrained}) not found for model {amodel_name}."
            )

    if pretrained_audio:
        if amodel_name.startswith("PANN"):
            if "Cnn14_mAP" in pretrained_audio:  # official checkpoint
                audio_ckpt = paddle.load(pretrained_audio)
                audio_ckpt = audio_ckpt["model"]
                keys = list(audio_ckpt.keys())
                for key in keys:
                    if (
                        "spectrogram_extractor" not in key
                        and "logmel_extractor" not in key
                    ):
                        v = audio_ckpt.pop(key)
                        audio_ckpt["audio_branch." + key] = v
            elif os.path.basename(pretrained_audio).startswith(
                "PANN"
            ):  # checkpoint trained via HTSAT codebase
                audio_ckpt = paddle.load(pretrained_audio)
                audio_ckpt = audio_ckpt["state_dict"]
                keys = list(audio_ckpt.keys())
                for key in keys:
                    if key.startswith("sed_model"):
                        v = audio_ckpt.pop(key)
                        audio_ckpt["audio_branch." + key[10:]] = v
            elif os.path.basename(pretrained_audio).startswith(
                "finetuned"
            ):  # checkpoint trained via linear probe codebase
                audio_ckpt = paddle.load(pretrained_audio)
            else:
                raise ValueError("Unknown audio checkpoint")
        elif amodel_name.startswith("HTSAT"):
            if "HTSAT_AudioSet_Saved" in pretrained_audio:  # official checkpoint
                audio_ckpt = paddle.load(pretrained_audio)
                audio_ckpt = audio_ckpt["state_dict"]
                keys = list(audio_ckpt.keys())
                for key in keys:
                    if key.startswith("sed_model") and (
                        "spectrogram_extractor" not in key
                        and "logmel_extractor" not in key
                    ):
                        v = audio_ckpt.pop(key)
                        audio_ckpt["audio_branch." + key[10:]] = v
            elif os.path.basename(pretrained_audio).startswith(
                "HTSAT"
            ):  # checkpoint trained via HTSAT codebase
                audio_ckpt = paddle.load(pretrained_audio)
                audio_ckpt = audio_ckpt["state_dict"]
                keys = list(audio_ckpt.keys())
                for key in keys:
                    if key.startswith("sed_model"):
                        v = audio_ckpt.pop(key)
                        audio_ckpt["audio_branch." + key[10:]] = v
            elif os.path.basename(pretrained_audio).startswith(
                "finetuned"
            ):  # checkpoint trained via linear probe codebase
                audio_ckpt = paddle.load(pretrained_audio)
            else:
                raise ValueError("Unknown audio checkpoint")
        else:
            raise f"this audio encoder pretrained checkpoint is not support"

        model.load_dict(audio_ckpt, strict=False)
        logging.info(
            f"Loading pretrained {amodel_name} weights ({pretrained_audio})."
        )
        param_names = [n for n, p in model.named_parameters()]
        for n in param_names:
            print(n, "\t", "Loaded" if n in audio_ckpt else "Unloaded")

    return model, model_cfg

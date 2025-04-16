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

import glob
import os
from typing import Dict, List, Union

import paddle
import safetensors.paddle

from ppdiffusers import DiffusionPipeline
from ppdiffusers.utils import (
    DIFFUSERS_CACHE,
    FROM_AISTUDIO,
    FROM_HF_HUB,
    PPDIFFUSERS_CACHE,
)


class CheckpointMergerPipeline(DiffusionPipeline):
    """
    A class that supports merging diffusion models based on the discussion here:
    https://github.com/huggingface/diffusers/issues/877

    Example usage:-

    pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", custom_pipeline="checkpoint_merger.py")

    merged_pipe = pipe.merge(["CompVis/stable-diffusion-v1-4","prompthero/openjourney"], interp = 'inv_sigmoid', alpha = 0.8, force = True)

    merged_pipe.to('cuda')

    prompt = "An astronaut riding a unicycle on Mars"

    results = merged_pipe(prompt)

    ## For more details, see the docstring for the merge method.

    """

    def __init__(self):
        self.register_to_config()
        super().__init__()

    def _convert_dict(self, ori_dict):
        del_flag = False
        for key, value in ori_dict.items():
            if isinstance(value, list):
                for item in value:
                    if item in ["ppdiffusers", "ppdiffusers.transformers"]:
                        ori_dict[key] = item
                    elif item == "diffusers" or item == "diffusers_paddle":
                        ori_dict[key] = "ppdiffusers"
                    elif item == "transformers" or item == "paddlenlp.transformers":
                        ori_dict[key] = "ppdiffusers.transformers"
            if key == "requires_safety_checker":
                del_flag = True
        if del_flag:
            del ori_dict["requires_safety_checker"]
        return ori_dict

    def _compare_model_configs(self, dict0, dict1):
        print(dict0)
        print(dict1)
        if dict0 == dict1:
            return True
        else:
            config0, meta_keys0 = self._remove_meta_keys(dict0)
            config1, meta_keys1 = self._remove_meta_keys(dict1)
            if config0 == config1:
                print(f"Warning !: Mismatch in keys {meta_keys0} and {meta_keys1}.")
                return True
        return False

    def _remove_meta_keys(self, config_dict: Dict):
        meta_keys = []
        temp_dict = config_dict.copy()
        for key in config_dict.keys():
            if key.startswith("_"):
                temp_dict.pop(key)
                meta_keys.append(key)
        return (temp_dict, meta_keys)

    @paddle.no_grad()
    def merge(
        self,
        pretrained_model_name_or_path_list: List[Union[str, os.PathLike]],
        **kwargs,
    ):
        """
        Returns a new pipeline object of the class 'DiffusionPipeline' with the merged checkpoints(weights) of the models passed
        in the argument 'pretrained_model_name_or_path_list' as a list.

        Parameters:
        -----------
            pretrained_model_name_or_path_list : A list of valid pretrained model names in the HuggingFace hub or paths to locally stored models in the HuggingFace format.

            **kwargs:
                Supports all the default DiffusionPipeline.get_config_dict kwargs viz..

                cache_dir, resume_download, force_download, proxies, local_files_only, token, revision, paddle_dtype, device_map.

                alpha - The interpolation parameter. Ranges from 0 to 1.  It affects the ratio in which the checkpoints are merged. A 0.8 alpha
                    would mean that the first model checkpoints would affect the final result far less than an alpha of 0.2

                interp - The interpolation method to use for the merging. Supports "sigmoid", "inv_sigmoid", "add_diff" and None.
                    Passing None uses the default interpolation which is weighted sum interpolation. For merging three checkpoints, only "add_diff" is supported.

                force - Whether to ignore mismatch in model_config.json for the current models. Defaults to False.

                variant - which variant of a pretrained model to load, e.g. "fp16" (None)

        """
        # Default kwargs from DiffusionPipeline
        from_hf_hub = kwargs.pop("from_hf_hub", FROM_HF_HUB)
        from_aistudio = kwargs.pop("from_aistudio", FROM_AISTUDIO)
        cache_dir = kwargs.pop("cache_dir", None)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        token = kwargs.pop("token", None)
        variant = kwargs.pop("variant", None)
        revision = kwargs.pop("revision", None)
        paddle_dtype = kwargs.pop("paddle_dtype", None)
        device_map = kwargs.pop("device_map", None)

        alpha = kwargs.pop("alpha", 0.5)
        interp = kwargs.pop("interp", None)

        print("Received list", pretrained_model_name_or_path_list)
        print(f"Combining with alpha={alpha}, interpolation mode={interp}")

        checkpoint_count = len(pretrained_model_name_or_path_list)
        # Ignore result from model_index_json comparison of the two checkpoints
        force = kwargs.pop("force", False)

        # If less than 2 checkpoints, nothing to merge. If more than 3, not supported for now.
        if checkpoint_count > 3 or checkpoint_count < 2:
            raise ValueError(
                "Received incorrect number of checkpoints to merge. Ensure that either 2 or 3 checkpoints are being"
                " passed."
            )

        print("Received the right number of checkpoints")
        # chkpt0, chkpt1 = pretrained_model_name_or_path_list[0:2]
        # chkpt2 = pretrained_model_name_or_path_list[2] if checkpoint_count == 3 else None

        # Validate that the checkpoints can be merged
        # Step 1: Load the model config and compare the checkpoints. We'll compare the model_index.json first while ignoring the keys starting with '_'
        config_dicts = []
        for pretrained_model_name_or_path in pretrained_model_name_or_path_list:
            config_dict = DiffusionPipeline.load_config(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
            )
            config_dict = self._convert_dict(config_dict)
            config_dicts.append(config_dict)

        comparison_result = True
        for idx in range(1, len(config_dicts)):
            comparison_result &= self._compare_model_configs(config_dicts[idx - 1], config_dicts[idx])
            if not force and comparison_result is False:
                raise ValueError("Incompatible checkpoints. Please check model_index.json for the models.")
        print("Compatible model_index.json files found")
        # Step 2: Basic Validation has succeeded. Let's download the models and save them into our local files.
        cached_folders = []
        for pretrained_model_name_or_path, config_dict in zip(pretrained_model_name_or_path_list, config_dicts):
            if os.path.isdir(pretrained_model_name_or_path):
                cached_folder = pretrained_model_name_or_path
            else:
                DiffusionPipeline.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    resume_download=True,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    revision=revision,
                    safety_checker=None,
                    use_safetensors=True,
                )
                if from_aistudio:
                    cached_folder = None  # TODO, check aistudio cache
                elif from_hf_hub:
                    cached_folder = os.path.join(DIFFUSERS_CACHE, pretrained_model_name_or_path)
                else:
                    cached_folder = os.path.join(PPDIFFUSERS_CACHE, pretrained_model_name_or_path)

            print("Cached Folder", cached_folder)
            cached_folders.append(cached_folder)

        # Step 3:-
        # Load the first checkpoint as a diffusion pipeline and modify its module state_dict in place
        final_pipe = DiffusionPipeline.from_pretrained(
            cached_folders[0],
            paddle_dtype=paddle_dtype,
            device_map=device_map,
            variant=variant,
            safety_checker=None,
        )
        final_pipe.to(self.device)

        checkpoint_path_2 = None
        if len(cached_folders) > 2:
            checkpoint_path_2 = os.path.join(cached_folders[2])

        if interp == "sigmoid":
            theta_func = CheckpointMergerPipeline.sigmoid
        elif interp == "inv_sigmoid":
            theta_func = CheckpointMergerPipeline.inv_sigmoid
        elif interp == "add_diff":
            theta_func = CheckpointMergerPipeline.add_difference
        else:
            theta_func = CheckpointMergerPipeline.weighted_sum

        # Find each module's state dict.
        for attr in final_pipe.config.keys():
            if not attr.startswith("_"):
                checkpoint_path_1 = os.path.join(cached_folders[1], attr)
                if os.path.exists(checkpoint_path_1):
                    files = [
                        *glob.glob(os.path.join(checkpoint_path_1, "*.safetensors")),
                        *glob.glob(os.path.join(checkpoint_path_1, "*.pdparams")),
                    ]
                    checkpoint_path_1 = files[0] if len(files) > 0 else None
                if len(cached_folders) < 3:
                    checkpoint_path_2 = None
                else:
                    checkpoint_path_2 = os.path.join(cached_folders[2], attr)
                    if os.path.exists(checkpoint_path_2):
                        files = [
                            *glob.glob(os.path.join(checkpoint_path_2, "*.safetensors")),
                            *glob.glob(os.path.join(checkpoint_path_2, "*.pdparams")),
                        ]
                        checkpoint_path_2 = files[0] if len(files) > 0 else None
                # For an attr if both checkpoint_path_1 and 2 are None, ignore.
                # If atleast one is present, deal with it according to interp method, of course only if the state_dict keys match.
                if checkpoint_path_1 is None and checkpoint_path_2 is None:
                    print(f"Skipping {attr}: not present in 2nd or 3d model")
                    continue
                try:
                    module = getattr(final_pipe, attr)
                    if isinstance(module, bool):  # ignore requires_safety_checker boolean
                        continue
                    theta_0 = getattr(module, "state_dict")
                    theta_0 = theta_0()

                    update_theta_0 = getattr(module, "set_state_dict")
                    theta_1 = (
                        safetensors.paddle.load_file(checkpoint_path_1)
                        if (checkpoint_path_1.endswith(".safetensors"))
                        else paddle.load(checkpoint_path_1)
                    )
                    theta_2 = None
                    if checkpoint_path_2:
                        theta_2 = (
                            safetensors.paddle.load_file(checkpoint_path_2)
                            if (checkpoint_path_2.endswith(".safetensors"))
                            else paddle.load(checkpoint_path_2)
                        )

                    if not theta_0.keys() == theta_1.keys():
                        print(f"Skipping {attr}: key mismatch")
                        continue
                    if theta_2 and not theta_1.keys() == theta_2.keys():
                        print(f"Skipping {attr}:y mismatch")
                except Exception as e:
                    print(f"Skipping {attr} do to an unexpected error: {str(e)}")
                    continue
                print(f"MERGING {attr}")

                for key in theta_0.keys():
                    if theta_2:
                        theta_0[key] = theta_func(theta_0[key], theta_1[key], theta_2[key], alpha)
                    else:
                        theta_0[key] = theta_func(theta_0[key], theta_1[key], None, alpha)

                del theta_1
                del theta_2
                update_theta_0(theta_0)

                del theta_0
        return final_pipe

    @staticmethod
    def weighted_sum(theta0, theta1, theta2, alpha):
        return ((1 - alpha) * theta0) + (alpha * theta1)

    # Smoothstep (https://en.wikipedia.org/wiki/Smoothstep)
    @staticmethod
    def sigmoid(theta0, theta1, theta2, alpha):
        alpha = alpha * alpha * (3 - (2 * alpha))
        return theta0 + ((theta1 - theta0) * alpha)

    # Inverse Smoothstep (https://en.wikipedia.org/wiki/Smoothstep)
    @staticmethod
    def inv_sigmoid(theta0, theta1, theta2, alpha):
        import math

        alpha = 0.5 - math.sin(math.asin(1.0 - 2.0 * alpha) / 3.0)
        return theta0 + ((theta1 - theta0) * alpha)

    @staticmethod
    def add_difference(theta0, theta1, theta2, alpha):
        return theta0 + (theta1 - theta2) * (1.0 - alpha)

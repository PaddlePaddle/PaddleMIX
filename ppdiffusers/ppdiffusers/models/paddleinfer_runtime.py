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
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from ..configuration_utils import FrozenDict
from ..utils import (
    DIFFUSERS_CACHE,
    FROM_HF_HUB,
    HF_HUB_OFFLINE,
    PADDLE_INFER_MODEL_NAME,
    PADDLE_INFER_WEIGHTS_NAME,
    PPDIFFUSERS_CACHE,
    _add_variant,
    _get_model_file,
    is_paddle_available,
    logging,
)
from ..version import VERSION as __version__

__all__ = ["PaddleInferRuntimeModel"]

if is_paddle_available():
    import paddle
    import paddle.inference as paddle_infer

logger = logging.get_logger(__name__)


class PaddleInferRuntimeModel:
    def __init__(self, model=None, config=None, **kwargs):
        logger.info("ppdiffusers.PaddleInferRuntimeModel")
        self.model = model
        self.config = config
        self.model_save_dir = kwargs.get("model_save_dir", None)
        self.latest_model_name = kwargs.get("latest_model_name", None)
        self.latest_params_name = kwargs.get("latest_params_name", None)
        if self.latest_model_name is None:
            self.latest_model_name = PADDLE_INFER_MODEL_NAME
        if self.latest_params_name is None:
            self.latest_params_name = PADDLE_INFER_WEIGHTS_NAME

    def __call__(self, **kwargs):
        kwargs.pop("output_shape", None)
        kwargs.pop("infer_op", None)
        inputs = {}
        # for k, v in kwargs.items():
        #     if k == "timestep":
        #         if v.ndim == 0:
        #             # fix 0D tensor error
        #             v = v.reshape((1,))
        #         # fix dtype error
        #         v = v.astype("float32")
        #     inputs[k] = v
        # input_names = self.model.get_input_names()
        # for i, name in enumerate(input_names):
        #     input_tensor = self.model.get_input_handle(name)
        #     if name not in inputs:
        #         raise ValueError(f"Input {name} is not in the model.")
        #     if isinstance(inputs[name], int):
        #         inputs[name] = paddle.to_tensor(inputs[name])
        #         if inputs[name].ndim == 0:  # fix 0D tensor error
        #             inputs[name] = inputs[name].reshape((1,))
        #             logger.warning(f"Input {name} is 0D tensor, reshape to (1,)")
        #     # if not isinstance(input_tensor, paddle.Tensor):
        #     #     input_tensor = paddle.to_tensor(input_tensor)
        #     #     logger.warning(f"Input {name} is not paddle tensor, convert to paddle tensor")
        #     #     if input_tensor.ndim == 0:  # fix 0D tensor error
        #     #         input_tensor = input_tensor.reshape((1,))
        #     input_tensor.reshape(inputs[name].shape)
        #     input_tensor.copy_from_cpu(inputs[name].numpy())
        # # do the inference
        # self.model.run()
        # results = []
        # # get out data from output tensor
        # output_names = self.model.get_output_names()
        # for i, name in enumerate(output_names):
        #     output_tensor = self.model.get_output_handle(name)
        #     output_data = output_tensor.copy_to_cpu()
        #     results.append(paddle.to_tensor(output_data))
        # return results
        for k, v in kwargs.items():
            if isinstance(v, int):
                v = paddle.to_tensor(v)
            if k == "timestep" or k == "num_frames":
                if v.ndim == 0:
                    # fix 0D tensor error
                    v = v.reshape((1,))
                # fix dtype error
                v = v.astype("float32")
            if isinstance(v, np.ndarray):
                v = paddle.to_tensor(v)
            inputs[k] = v
        input_list = []
        input_names = self.model.get_input_names()
        for i, name in enumerate(input_names):
            if name not in inputs:
                raise ValueError(f"Input {name} is not in the model.")
            input_list.append(inputs[name])
        # do the inference (zero copy)
        self.model.run(input_list)
        results = []
        # get out data from output tensor
        output_names = self.model.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = self.model.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(paddle.to_tensor(output_data))
        return results

    @staticmethod
    def load_model(
        model_path: Union[str, Path],
        params_path: Union[str, Path] = None,
        use_optim_cache: bool = False,
        infer_config: Optional["paddle_infer.Config"] = None,
    ):
        """
        Loads an FastDeploy Inference Model with fastdeploy.RuntimeOption
        Arguments:
            model_path (`str` or `Path`):
                Model path from which to load
            params_path (`str` or `Path`):
                Params path from which to load
            use_optim_cache (`bool`, *optional*, defaults to `False`):
                Whether to automatically load the optimized parameters from cache.(If the cache does not exist, it will be automatically generated)
            runtime_options (fd.RuntimeOption, *optional*):
                The RuntimeOption of fastdeploy to initialize the fastdeploy runtime. Default setting
                the device to cpu and the backend to paddle inference
        """
        if infer_config is None:
            infer_config = paddle_infer.Config()

        if use_optim_cache:
            # 首次运行，自动生成优化模型
            params_dir = os.path.dirname(params_path)
            optim_cache_dir = os.path.join(params_dir, "_optim_cache")
            if not os.path.exists(optim_cache_dir):
                os.makedirs(optim_cache_dir)
                infer_config.switch_ir_optim(True)
                infer_config.set_optim_cache_dir(optim_cache_dir)
                infer_config.enable_save_optim_model(True)
            else:
                # 第二次运行，加载缓存的optim模型
                infer_config.switch_ir_optim(False)
                optimized_params_path = os.path.join(optim_cache_dir, "_optimized.pdiparams")
                optimized_model_path = os.path.join(optim_cache_dir, "_optimized.pdmodel")
                model_path = optimized_model_path
                params_path = optimized_params_path

        infer_config.set_prog_file(model_path)
        infer_config.set_params_file(params_path)
        return paddle_infer.create_predictor(infer_config)

    def _save_pretrained(
        self,
        save_directory: Union[str, Path],
        model_file_name: Optional[str] = None,
        params_file_name: Optional[str] = None,
        **kwargs
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~FastDeployRuntimeModel.from_pretrained`] class method. It will always save the
        latest_model_name.
        Arguments:
            save_directory (`str` or `Path`):
                Directory where to save the model file.
            model_file_name(`str`, *optional*):
                Overwrites the default model file name from `"inference.pdmodel"` to `model_file_name`. This allows you to save the
                model with a different name.
            params_file_name(`str`, *optional*):
                Overwrites the default model file name from `"inference.pdiparams"` to `params_file_name`. This allows you to save the
                model with a different name.
        """
        model_file_name = model_file_name if model_file_name is not None else PADDLE_INFER_MODEL_NAME
        params_file_name = params_file_name if params_file_name is not None else PADDLE_INFER_WEIGHTS_NAME

        src_model_path = self.model_save_dir.joinpath(self.latest_model_name)
        dst_model_path = Path(save_directory).joinpath(model_file_name)

        try:
            shutil.copyfile(src_model_path, dst_model_path)
        except shutil.SameFileError:
            pass

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        **kwargs,
    ):
        """
        Save a model to a directory, so that it can be re-loaded using the [`~FastDeployRuntimeModel.from_pretrained`] class
        method.:
        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # saving model weights/files
        self._save_pretrained(save_directory, **kwargs)

    @classmethod
    def _from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        model_file_name: Optional[str] = None,
        params_file_name: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[str] = None,
        subfolder: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        infer_config: Optional["paddle_infer.Config"] = None,
        use_optim_cache: bool = False,
        from_hf_hub: Optional[bool] = False,
        proxies: Optional[Dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        user_agent: Union[Dict, str, None] = None,
        is_onnx_model: bool = False,
        **kwargs,
    ):
        """
        Load a model from a directory or the HF Hub.
        Arguments:
            pretrained_model_name_or_path (`str` or `Path`):
                Directory from which to load
            model_file_name (`str`):
                Overwrites the default model file name from `"inference.pdmodel"` to `file_name`. This allows you to load
                different model files from the same repository or directory.
            params_file_name (`str`):
                Overwrites the default params file name from `"inference.pdiparams"` to `file_name`. This allows you to load
                different model files from the same repository or directory.
            use_auth_token (`str` or `bool`):
                Is needed to load models from a private or gated repository
            revision (`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            cache_dir (`Union[str, Path]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            runtime_options (`fastdeploy.RuntimeOption`, *optional*):
                The RuntimeOption of fastdeploy.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization
        """

        model_file_name = model_file_name if model_file_name is not None else PADDLE_INFER_MODEL_NAME
        params_file_name = params_file_name if params_file_name is not None else PADDLE_INFER_WEIGHTS_NAME
        config = None

        # load model from local directory
        if os.path.isdir(pretrained_model_name_or_path):
            model_path = os.path.join(pretrained_model_name_or_path, model_file_name)
            params_path = os.path.join(pretrained_model_name_or_path, params_file_name)

            model = PaddleInferRuntimeModel.load_model(
                model_path,
                params_path,
                infer_config=infer_config,
                use_optim_cache=use_optim_cache,
            )
            # 加载模型配置文件
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    config = FrozenDict(config)
            kwargs["model_save_dir"] = Path(pretrained_model_name_or_path)

        # load model from hub or paddle bos
        else:
            model_cache_path = _get_model_file(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                weights_name=model_file_name,
                subfolder=subfolder,
                cache_dir=cache_dir,
                force_download=force_download,
                revision=revision,
                from_hf_hub=from_hf_hub,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
            )

            params_cache_path = _get_model_file(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                weights_name=params_file_name,
                subfolder=subfolder,
                cache_dir=cache_dir,
                force_download=force_download,
                revision=revision,
                from_hf_hub=from_hf_hub,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
            )
            kwargs["latest_params_name"] = Path(params_cache_path).name
            kwargs["model_save_dir"] = Path(model_cache_path).parent
            kwargs["latest_model_name"] = Path(model_cache_path).name

            model = PaddleInferRuntimeModel.load_model(
                model_cache_path,
                params_cache_path,
                infer_config=infer_config,
                use_optim_cache=use_optim_cache,
            )
            # 加载模型配置文件
            config_path = os.path.join(kwargs["model_save_dir"], "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    config = FrozenDict(config)

        return cls(model=model, config=config, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        model_file_name: Optional[str] = None,
        params_file_name: Optional[str] = None,
        infer_configs: Optional["paddle_infer.Config"] = None,
        use_optim_cache: bool = False,
        **kwargs,
    ):
        from_hf_hub = kwargs.pop("from_hf_hub", FROM_HF_HUB)
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        variant = kwargs.pop("variant", None)

        user_agent = {
            "ppdiffusers": __version__,
            "file_type": "model",
            "framework": "paddleinfer",
        }

        return cls._from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model_file_name=_add_variant(model_file_name, variant),
            params_file_name=_add_variant(params_file_name, variant),
            use_auth_token=use_auth_token,
            revision=revision,
            subfolder=subfolder,
            force_download=force_download,
            cache_dir=cache_dir,
            infer_config=infer_configs,
            use_optim_cache=use_optim_cache,
            from_hf_hub=from_hf_hub,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            user_agent=user_agent,
            **kwargs,
        )

    @property
    def dtype(self) -> Union[str, paddle.dtype]:
        return "float32"

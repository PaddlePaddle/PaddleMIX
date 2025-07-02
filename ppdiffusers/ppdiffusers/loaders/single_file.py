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
import copy
import os
from io import BytesIO
from pathlib import Path

import paddle
import requests
from huggingface_hub import hf_hub_download
from huggingface_hub.file_download import _request_wrapper, hf_raise_for_status

from ..utils import (
    BACKENDS_MAPPING,
    DIFFUSERS_CACHE,
    HF_HUB_OFFLINE,
    PPDIFFUSERS_CACHE,
    deprecate,
    is_omegaconf_available,
    logging,
    ppdiffusers_url_download,
    smart_load,
)

try:
    from paddlenlp.transformers.model_utils import no_init_weights
except ImportError:
    from ..utils.paddle_utils import no_init_weights

from ..models.modeling_utils import ContextManagers, faster_set_state_dict


def http_file_name(
    url: str,
    *,
    proxies=None,
    headers=None,
    timeout=10.0,
    max_retries=0,
):
    """
    Get a remote file name.
    """
    headers = copy.deepcopy(headers) or {}
    r = _request_wrapper(
        method="GET",
        url=url,
        stream=True,
        proxies=proxies,
        headers=headers,
        timeout=timeout,
        # max_retries=max_retries,
    )
    hf_raise_for_status(r)
    displayed_name = url.split("/")[-1]
    content_disposition = r.headers.get("Content-Disposition")
    if content_disposition is not None and "filename=" in content_disposition:
        # Means file is on CDN
        displayed_name = content_disposition.split("filename=")[-1]
    return displayed_name


logger = logging.get_logger(__name__)


class FromSingleFileMixin:
    """
    Load model weights saved in the `.ckpt` format into a [`DiffusionPipeline`].
    """

    @classmethod
    def from_ckpt(cls, *args, **kwargs):
        deprecation_message = "The function `from_ckpt` is deprecated in favor of `from_single_file` and will be removed in ppdiffusers v.0.21. Please make sure to use `StableDiffusionPipeline.from_single_file(...)` instead."
        deprecate("from_ckpt", "0.21.0", deprecation_message, standard_warn=False)
        return cls.from_single_file(*args, **kwargs)

    @classmethod
    def from_single_file(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Instantiate a [`DiffusionPipeline`] from pretrained pipeline weights saved in the `.ckpt` or `.safetensors`
        format. The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A link to the `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                    - A path to a *file* containing all pipeline weights.
            paddle_dtype (`str` or `paddle.dtype`, *optional*):
                Override the default `paddle.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            extract_ema (`bool`, *optional*, defaults to `False`):
                Whether to extract the EMA weights or not. Pass `True` to extract the EMA weights which usually yield
                higher quality images for inference. Non-EMA weights are usually better for continuing finetuning.
            upcast_attention (`bool`, *optional*, defaults to `None`):
                Whether the attention computation should always be upcasted.
            image_size (`int`, *optional*, defaults to 512):
                The image size the model was trained on. Use 512 for all Stable Diffusion v1 models and the Stable
                Diffusion v2 base model. Use 768 for Stable Diffusion v2.
            prediction_type (`str`, *optional*):
                The prediction type the model was trained on. Use `'epsilon'` for all Stable Diffusion v1 models and
                the Stable Diffusion v2 base model. Use `'v_prediction'` for Stable Diffusion v2.
            num_in_channels (`int`, *optional*, defaults to `None`):
                The number of input channels. If `None`, it is automatically inferred.
            scheduler_type (`str`, *optional*, defaults to `"pndm"`):
                Type of scheduler to use. Should be one of `["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm",
                "ddim"]`.
            load_safety_checker (`bool`, *optional*, defaults to `True`):
                Whether to load the safety checker or not.
            text_encoder ([`~transformers.CLIPTextModel`], *optional*, defaults to `None`):
                An instance of `CLIPTextModel` to use, specifically the
                [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant. If this
                parameter is `None`, the function loads a new instance of `CLIPTextModel` by itself if needed.
            vae (`AutoencoderKL`, *optional*, defaults to `None`):
                Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations. If
                this parameter is `None`, the function will load a new instance of [CLIP] by itself, if needed.
            tokenizer ([`~transformers.CLIPTokenizer`], *optional*, defaults to `None`):
                An instance of `CLIPTokenizer` to use. If this parameter is `None`, the function loads a new instance
                of `CLIPTokenizer` by itself if needed.
            original_config_file (`str`):
                Path to `.yaml` config file corresponding to the original architecture. If `None`, will be
                automatically inferred by looking for a key that only exists in SD2.0 models.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (for example the pipeline components of the
                specific pipeline class). The overwritten components are directly passed to the pipelines `__init__`
                method. See example below for more information.

        Examples:

        ```py
        >>> from ppdiffusers import StableDiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = StableDiffusionPipeline.from_single_file(
        ...     "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors"
        ... )

        >>> # Download pipeline from local file
        >>> # file is downloaded under ./v1-5-pruned-emaonly.ckpt
        >>> pipeline = StableDiffusionPipeline.from_single_file("./v1-5-pruned-emaonly")

        >>> # Enable float16
        >>> pipeline = StableDiffusionPipeline.from_single_file(
        ...     "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt",
        ...     paddle_dtype=paddle.float16,
        ... )
        ```
        """
        # import here to avoid circular dependency
        from ..pipelines.flux.convert_from_ckpt import (  # 修改此处路径
            download_from_original_flux_ckpt,
        )
        from ..pipelines.stable_diffusion.convert_from_ckpt import (
            download_from_original_stable_diffusion_ckpt,
        )

        # from ..pipelines.flux

        from_hf_hub = any(p in pretrained_model_link_or_path for p in ["huggingface.co", "hf.co", "hf-mirror"])
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )
        original_config_file = kwargs.pop("original_config_file", None)
        config_files = kwargs.pop("config_files", None)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        extract_ema = kwargs.pop("extract_ema", False)
        image_size = kwargs.pop("image_size", None)
        scheduler_type = kwargs.pop("scheduler_type", "pndm")
        num_in_channels = kwargs.pop("num_in_channels", None)
        upcast_attention = kwargs.pop("upcast_attention", None)
        load_safety_checker = kwargs.pop("load_safety_checker", False)
        prediction_type = kwargs.pop("prediction_type", None)
        text_encoder = kwargs.pop("text_encoder", None)
        vae = kwargs.pop("vae", None)
        controlnet = kwargs.pop("controlnet", None)
        adapter = kwargs.pop("adapter", None)
        tokenizer = kwargs.pop("tokenizer", None)

        paddle_dtype = kwargs.pop("paddle_dtype", None)

        use_safetensors = kwargs.pop("use_safetensors", None)

        pipeline_name = cls.__name__
        file_extension = pretrained_model_link_or_path.rsplit(".", 1)[-1]
        from_safetensors = file_extension == "safetensors"

        if from_safetensors and use_safetensors is False:
            raise ValueError("Make sure to install `safetensors` with `pip install safetensors`.")

        # TODO: For now we only support stable diffusion
        stable_unclip = None
        model_type = None

        if pipeline_name in [
            "StableDiffusionControlNetPipeline",
            "StableDiffusionControlNetImg2ImgPipeline",
            "StableDiffusionControlNetInpaintPipeline",
        ]:
            from ..models.controlnet import ControlNetModel
            from ..pipelines.controlnet.multicontrolnet import MultiControlNetModel

            #  list/tuple or a single instance of ControlNetModel or MultiControlNetModel
            if not (
                isinstance(controlnet, (ControlNetModel, MultiControlNetModel))
                or isinstance(controlnet, (list, tuple))
                and isinstance(controlnet[0], ControlNetModel)
            ):
                raise ValueError("ControlNet needs to be passed if loading from ControlNet pipeline.")
        elif "StableDiffusion" in pipeline_name or "Flux" in pipeline_name:
            # Model type will be inferred from the checkpoint.
            pass
        elif pipeline_name == "StableUnCLIPPipeline":
            model_type = "FrozenOpenCLIPEmbedder"
            stable_unclip = "txt2img"
        elif pipeline_name == "StableUnCLIPImg2ImgPipeline":
            model_type = "FrozenOpenCLIPEmbedder"
            stable_unclip = "img2img"
        elif pipeline_name == "PaintByExamplePipeline":
            model_type = "PaintByExample"
        elif pipeline_name == "LDMTextToImagePipeline":
            model_type = "LDMTextToImage"
        else:
            raise ValueError(f"Unhandled pipeline class: {pipeline_name}")

        pretrained_model_link_or_path = str(pretrained_model_link_or_path)
        if os.path.isfile(pretrained_model_link_or_path):
            checkpoint_path = pretrained_model_link_or_path
        elif pretrained_model_link_or_path.startswith("http://") or pretrained_model_link_or_path.startswith(
            "https://"
        ):
            if from_hf_hub:
                # remove huggingface url
                has_valid_url_prefix = False
                valid_url_prefixes = [
                    "https://huggingface.co/",
                    "huggingface.co/",
                    "hf.co/",
                    "https://hf.co/",
                    "hf-mirror.com/",
                    "https://hf-mirror.com/",
                ]
                for prefix in valid_url_prefixes:
                    if pretrained_model_link_or_path.startswith(prefix):
                        pretrained_model_link_or_path = pretrained_model_link_or_path[len(prefix) :]
                        has_valid_url_prefix = True

                # Code based on diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained
                ckpt_path = Path(pretrained_model_link_or_path)
                if not ckpt_path.is_file():
                    if not has_valid_url_prefix:
                        raise ValueError(
                            f"The provided path is either not a file or a valid huggingface URL was not provided. Valid URLs begin with {', '.join(valid_url_prefixes)}"
                        )

                    # get repo_id and (potentially nested) file path of ckpt in repo
                    repo_id = "/".join(ckpt_path.parts[:2])
                    file_path = "/".join(ckpt_path.parts[2:])

                    if file_path.startswith("blob/"):
                        file_path = file_path[len("blob/") :]

                    if file_path.startswith("main/"):
                        file_path = file_path[len("main/") :]

                    checkpoint_path = hf_hub_download(
                        repo_id,
                        filename=file_path,
                        cache_dir=cache_dir,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=use_auth_token,
                        revision=revision,
                        force_download=force_download,
                    )
                else:
                    checkpoint_path = ckpt_path
            else:
                checkpoint_path = ppdiffusers_url_download(
                    pretrained_model_link_or_path,
                    cache_dir=cache_dir,
                    filename=http_file_name(pretrained_model_link_or_path).strip('"'),
                    force_download=force_download,
                    resume_download=resume_download,
                )
        else:
            checkpoint_path = pretrained_model_link_or_path

        if "Flux" in pipeline_name:
            pipe = download_from_original_flux_ckpt(
                checkpoint_path,
                pipeline_class=cls,
                model_type=model_type,
                stable_unclip=stable_unclip,
                controlnet=controlnet,
                adapter=adapter,
                from_safetensors=from_safetensors,
                extract_ema=extract_ema,
                image_size=image_size,
                scheduler_type=scheduler_type,
                num_in_channels=num_in_channels,
                upcast_attention=upcast_attention,
                load_safety_checker=load_safety_checker,
                prediction_type=prediction_type,
                paddle_dtype=paddle_dtype,
                text_encoder=text_encoder,
                vae=vae,
                tokenizer=tokenizer,
                original_config_file=original_config_file,
                config_files=config_files,
                local_files_only=local_files_only,
            )
        else:

            pipe = download_from_original_stable_diffusion_ckpt(
                checkpoint_path,
                pipeline_class=cls,
                model_type=model_type,
                stable_unclip=stable_unclip,
                controlnet=controlnet,
                adapter=adapter,
                from_safetensors=from_safetensors,
                extract_ema=extract_ema,
                image_size=image_size,
                scheduler_type=scheduler_type,
                num_in_channels=num_in_channels,
                upcast_attention=upcast_attention,
                load_safety_checker=load_safety_checker,
                prediction_type=prediction_type,
                paddle_dtype=paddle_dtype,
                text_encoder=text_encoder,
                vae=vae,
                tokenizer=tokenizer,
                original_config_file=original_config_file,
                config_files=config_files,
                local_files_only=local_files_only,
            )

        return pipe


class FromOriginalVAEMixin:
    """
    Load pretrained ControlNet weights saved in the `.ckpt` or `.safetensors` format into an [`AutoencoderKL`].
    """

    @classmethod
    def from_single_file(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Instantiate a [`AutoencoderKL`] from pretrained ControlNet weights saved in the original `.ckpt` or
        `.safetensors` format. The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A link to the `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                    - A path to a *file* containing all pipeline weights.
            paddle_dtype (`str` or `paddle.dtype`, *optional*):
                Override the default `paddle.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to True, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            image_size (`int`, *optional*, defaults to 512):
                The image size the model was trained on. Use 512 for all Stable Diffusion v1 models and the Stable
                Diffusion v2 base model. Use 768 for Stable Diffusion v2.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            upcast_attention (`bool`, *optional*, defaults to `None`):
                Whether the attention computation should always be upcasted.
            scaling_factor (`float`, *optional*, defaults to 0.18215):
                The component-wise standard deviation of the trained latent space computed using the first batch of the
                training set. This is used to scale the latent space to have unit variance when training the diffusion
                model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
                diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z
                = 1 / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution
                Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (for example the pipeline components of the
                specific pipeline class). The overwritten components are directly passed to the pipelines `__init__`
                method. See example below for more information.

        <Tip warning={true}>

            Make sure to pass both `image_size` and `scaling_factor` to `from_single_file()` if you're loading
            a VAE from SDXL or a Stable Diffusion v2 model or higher.

        </Tip>

        Examples:

        ```py
        from ppdiffusers import AutoencoderKL

        url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"  # can also be local file
        model = AutoencoderKL.from_single_file(url)
        ```
        """
        from_hf_hub = any(p in pretrained_model_link_or_path for p in ["huggingface.co", "hf.co", "hf-mirror"])
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )

        if not is_omegaconf_available():
            raise ValueError(BACKENDS_MAPPING["omegaconf"][1])

        from omegaconf import OmegaConf

        from ..models import AutoencoderKL

        # import here to avoid circular dependency
        from ..pipelines.stable_diffusion.convert_from_ckpt import (
            convert_diffusers_vae_unet_to_ppdiffusers,
            convert_ldm_vae_checkpoint,
            create_vae_diffusers_config,
        )

        config_file = kwargs.pop("config_file", None)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        image_size = kwargs.pop("image_size", None)
        scaling_factor = kwargs.pop("scaling_factor", None)
        kwargs.pop("upcast_attention", None)

        paddle_dtype = kwargs.pop("paddle_dtype", None)

        use_safetensors = kwargs.pop("use_safetensors", None)

        file_extension = pretrained_model_link_or_path.rsplit(".", 1)[-1]
        from_safetensors = file_extension == "safetensors"

        if from_safetensors and use_safetensors is False:
            raise ValueError("Make sure to install `safetensors` with `pip install safetensors`.")

        pretrained_model_link_or_path = str(pretrained_model_link_or_path)
        if os.path.isfile(pretrained_model_link_or_path):
            checkpoint_path = pretrained_model_link_or_path
        elif pretrained_model_link_or_path.startswith("http://") or pretrained_model_link_or_path.startswith(
            "https://"
        ):
            if from_hf_hub:
                # remove huggingface url
                has_valid_url_prefix = False
                valid_url_prefixes = [
                    "https://huggingface.co/",
                    "huggingface.co/",
                    "hf.co/",
                    "https://hf.co/",
                    "hf-mirror.com/",
                    "https://hf-mirror.com/",
                ]
                for prefix in valid_url_prefixes:
                    if pretrained_model_link_or_path.startswith(prefix):
                        pretrained_model_link_or_path = pretrained_model_link_or_path[len(prefix) :]
                        has_valid_url_prefix = True

                # Code based on diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained
                ckpt_path = Path(pretrained_model_link_or_path)
                if not ckpt_path.is_file():
                    if not has_valid_url_prefix:
                        raise ValueError(
                            f"The provided path is either not a file or a valid huggingface URL was not provided. Valid URLs begin with {', '.join(valid_url_prefixes)}"
                        )
                    # get repo_id and (potentially nested) file path of ckpt in repo
                    repo_id = "/".join(ckpt_path.parts[:2])
                    file_path = "/".join(ckpt_path.parts[2:])

                    if file_path.startswith("blob/"):
                        file_path = file_path[len("blob/") :]

                    if file_path.startswith("main/"):
                        file_path = file_path[len("main/") :]

                    checkpoint_path = hf_hub_download(
                        repo_id,
                        filename=file_path,
                        cache_dir=cache_dir,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=use_auth_token,
                        revision=revision,
                        force_download=force_download,
                    )
                else:
                    checkpoint_path = ckpt_path

            else:
                checkpoint_path = ppdiffusers_url_download(
                    pretrained_model_link_or_path,
                    cache_dir=cache_dir,
                    filename=http_file_name(pretrained_model_link_or_path).strip('"'),
                    force_download=force_download,
                    resume_download=resume_download,
                )
        else:
            checkpoint_path = pretrained_model_link_or_path
        checkpoint = smart_load(checkpoint_path, return_numpy=True)

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        if config_file is None:
            config_url = "https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/v1-inference.yaml"
            config_file = BytesIO(requests.get(config_url).content)

        original_config = OmegaConf.load(config_file)

        # default to sd-v1-5
        image_size = image_size or 512

        vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

        if scaling_factor is None:
            if (
                "model" in original_config
                and "params" in original_config.model
                and "scale_factor" in original_config.model.params
            ):
                vae_scaling_factor = original_config.model.params.scale_factor
            else:
                vae_scaling_factor = 0.18215  # default SD scaling factor

        vae_config["scaling_factor"] = vae_scaling_factor

        init_contexts = []
        init_contexts.append(paddle.dtype_guard(paddle.float32))
        init_contexts.append(no_init_weights(_enable=True))
        if hasattr(paddle, "LazyGuard"):
            init_contexts.append(paddle.LazyGuard())
        with ContextManagers(init_contexts):
            vae = AutoencoderKL(**vae_config)

        # we must transpose linear layer
        faster_set_state_dict(vae, convert_diffusers_vae_unet_to_ppdiffusers(vae, converted_vae_checkpoint))

        if paddle_dtype is not None:
            vae.to(paddle_dtype=paddle_dtype)

        return vae


class FromOriginalControlnetMixin:
    """
    Load pretrained ControlNet weights saved in the `.ckpt` or `.safetensors` format into a [`ControlNetModel`].
    """

    @classmethod
    def from_single_file(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Instantiate a [`ControlNetModel`] from pretrained ControlNet weights saved in the original `.ckpt` or
        `.safetensors` format. The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A link to the `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                    - A path to a *file* containing all pipeline weights.
            paddle_dtype (`str` or `paddle.dtype`, *optional*):
                Override the default `paddle.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to True, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            image_size (`int`, *optional*, defaults to 512):
                The image size the model was trained on. Use 512 for all Stable Diffusion v1 models and the Stable
                Diffusion v2 base model. Use 768 for Stable Diffusion v2.
            upcast_attention (`bool`, *optional*, defaults to `None`):
                Whether the attention computation should always be upcasted.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (for example the pipeline components of the
                specific pipeline class). The overwritten components are directly passed to the pipelines `__init__`
                method. See example below for more information.

        Examples:

        ```py
        from ppdiffusers import StableDiffusionControlNetPipeline, ControlNetModel

        url = "https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth"  # can also be a local path
        model = ControlNetModel.from_single_file(url)

        url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.safetensors"  # can also be a local path
        pipe = StableDiffusionControlNetPipeline.from_single_file(url, controlnet=controlnet)
        ```
        """
        # import here to avoid circular dependency
        from ..pipelines.stable_diffusion.convert_from_ckpt import (
            download_controlnet_from_original_ckpt,
        )

        from_hf_hub = any(p in pretrained_model_link_or_path for p in ["huggingface.co", "hf.co", "hf-mirror"])
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )

        config_file = kwargs.pop("config_file", None)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        num_in_channels = kwargs.pop("num_in_channels", None)
        use_linear_projection = kwargs.pop("use_linear_projection", None)
        revision = kwargs.pop("revision", None)
        extract_ema = kwargs.pop("extract_ema", False)
        image_size = kwargs.pop("image_size", None)
        upcast_attention = kwargs.pop("upcast_attention", None)

        paddle_dtype = kwargs.pop("paddle_dtype", None)

        use_safetensors = kwargs.pop("use_safetensors", None)

        file_extension = pretrained_model_link_or_path.rsplit(".", 1)[-1]
        from_safetensors = file_extension == "safetensors"

        if from_safetensors and use_safetensors is False:
            raise ValueError("Make sure to install `safetensors` with `pip install safetensors`.")

        pretrained_model_link_or_path = str(pretrained_model_link_or_path)
        if os.path.isfile(pretrained_model_link_or_path):
            checkpoint_path = pretrained_model_link_or_path
        elif pretrained_model_link_or_path.startswith("http://") or pretrained_model_link_or_path.startswith(
            "https://"
        ):
            if from_hf_hub:
                # remove huggingface url
                has_valid_url_prefix = False
                valid_url_prefixes = [
                    "https://huggingface.co/",
                    "huggingface.co/",
                    "hf.co/",
                    "https://hf.co/",
                    "hf-mirror.com/",
                    "https://hf-mirror.com/",
                ]
                for prefix in valid_url_prefixes:
                    if pretrained_model_link_or_path.startswith(prefix):
                        pretrained_model_link_or_path = pretrained_model_link_or_path[len(prefix) :]
                        has_valid_url_prefix = True

                # Code based on diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained
                ckpt_path = Path(pretrained_model_link_or_path)
                if not ckpt_path.is_file():
                    if not has_valid_url_prefix:
                        raise ValueError(
                            f"The provided path is either not a file or a valid huggingface URL was not provided. Valid URLs begin with {', '.join(valid_url_prefixes)}"
                        )
                    # get repo_id and (potentially nested) file path of ckpt in repo
                    repo_id = "/".join(ckpt_path.parts[:2])
                    file_path = "/".join(ckpt_path.parts[2:])

                    if file_path.startswith("blob/"):
                        file_path = file_path[len("blob/") :]

                    if file_path.startswith("main/"):
                        file_path = file_path[len("main/") :]

                    checkpoint_path = hf_hub_download(
                        repo_id,
                        filename=file_path,
                        cache_dir=cache_dir,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=use_auth_token,
                        revision=revision,
                        force_download=force_download,
                    )
                else:
                    checkpoint_path = ckpt_path

            else:
                checkpoint_path = ppdiffusers_url_download(
                    pretrained_model_link_or_path,
                    cache_dir=cache_dir,
                    filename=http_file_name(pretrained_model_link_or_path).strip('"'),
                    force_download=force_download,
                    resume_download=resume_download,
                )
        else:
            checkpoint_path = pretrained_model_link_or_path

        if config_file is None:
            config_url = "https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/cldm_v15.yaml"
            config_file = BytesIO(requests.get(config_url).content)

        image_size = image_size or 512

        controlnet = download_controlnet_from_original_ckpt(
            checkpoint_path,
            original_config_file=config_file,
            image_size=image_size,
            extract_ema=extract_ema,
            num_in_channels=num_in_channels,
            upcast_attention=upcast_attention,
            from_safetensors=from_safetensors,
            use_linear_projection=use_linear_projection,
        )

        if paddle_dtype is not None:
            controlnet.to(paddle_dtype=paddle_dtype)

        return controlnet


FromCkptMixin = FromSingleFileMixin

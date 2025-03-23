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

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import paddle
import PIL.Image
from paddle import nn

from ppdiffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from ppdiffusers.image_processor import PipelineImageInput
from ppdiffusers.loaders import IPAdapterMixin
from ppdiffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_5,
    IPAdapterAttnProcessor,
    IPAdapterAttnProcessor2_5,
)
from ppdiffusers.models.modeling_pytorch_paddle_utils import (
    convert_pytorch_state_dict_to_paddle,
)
from ppdiffusers.models.modeling_utils import ContextManagers, faster_set_state_dict
from ppdiffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from ppdiffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from ppdiffusers.utils import (
    DIFFUSERS_CACHE,
    FROM_AISTUDIO,
    FROM_DIFFUSERS,
    FROM_HF_HUB,
    HF_HUB_OFFLINE,
    PPDIFFUSERS_CACHE,
    _get_model_file,
    logging,
    smart_load,
)
from ppdiffusers.utils.import_utils import is_ppxformers_available

try:
    from paddlenlp.transformers.model_utils import no_init_weights
except ImportError:
    from ppdiffusers.utils.paddle_utils import no_init_weights

from resampler import Resampler
from safetensors import safe_open

logger = logging.get_logger(__name__)


def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1
        )
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)
    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


class StableDiffusionXLInstantIDPipeline(StableDiffusionXLControlNetPipeline, IPAdapterMixin):
    def load_ip_adapter_instantid(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, paddle.Tensor]],
        subfolder: str = "",
        weight_name: str = None,
        image_emb_dim: int = 512,
        num_tokens: int = 16,
        scale: float = 0.5,
        **kwargs
    ):
        # Load the main state dict first.
        from_hf_hub = kwargs.pop("from_hf_hub", FROM_HF_HUB)
        from_aistudio = kwargs.pop("from_aistudio", FROM_AISTUDIO)
        cache_dir = kwargs.pop("cache_dir", None)

        if cache_dir is None:
            if from_aistudio:
                cache_dir = None
            elif from_hf_hub:
                cache_dir = DIFFUSERS_CACHE
            else:
                cache_dir = PPDIFFUSERS_CACHE

        from_diffusers = kwargs.pop("from_diffusers", FROM_DIFFUSERS)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        if subfolder is None:
            subfolder = ""
        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch" if from_diffusers else "paddle",
        }
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
                from_hf_hub=from_hf_hub,
                from_aistudio=from_aistudio,
            )
            if weight_name.endswith(".safetensors"):
                state_dict = {"image_proj": {}, "ip_adapter": {}}
                with safe_open(model_file, framework="np") as f:
                    metadata = f.metadata()
                    if metadata is None:
                        metadata = {}
                    if metadata.get("format", "pt") not in ["pt", "pd", "np"]:
                        raise OSError(
                            f"The safetensors archive passed at {model_file} does not contain the valid metadata. Make sure "
                            "you save your model with the `save_pretrained` method."
                        )
                    data_format = metadata.get("format", "pt")
                    if data_format == "pt" and not from_diffusers:
                        logger.warning(
                            "Detect the weight is in diffusers format, but currently, `from_diffusers` is set to "
                            "`False`. To proceed, we will change the value of `from_diffusers` to `True`!"
                        )
                        from_diffusers = True
                    for key in f.keys():
                        if key.startswith("image_proj."):
                            state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                        elif key.startswith("ip_adapter."):
                            state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
            else:
                state_dict = smart_load(model_file, return_numpy=True, return_is_torch_weight=True)
                is_torch_weight = state_dict.pop("is_torch_weight", False)
                if not from_diffusers and is_torch_weight:
                    logger.warning(
                        "Detect the weight is in diffusers format, but currently, `from_diffusers` is set to `False`. "
                        "To proceed, we will change the value of `from_diffusers` to `True`!"
                    )
                    from_diffusers = True
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        keys = list(state_dict.keys())
        if sorted(keys) != ["image_proj", "ip_adapter"]:
            raise ValueError("Required keys are (`image_proj` and `ip_adapter`) missing from the state dict.")

        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=self.unet.config.cross_attention_dim,
            ff_mult=4,
        )

        init_contexts = []
        init_contexts.append(paddle.dtype_guard(self.dtype))
        init_contexts.append(no_init_weights(_enable=True))
        if hasattr(paddle, "LazyGuard"):
            init_contexts.append(paddle.LazyGuard())
        with ContextManagers(init_contexts):
            self.image_proj_model = image_proj_model

        if from_diffusers:
            convert_pytorch_state_dict_to_paddle(self.image_proj_model, state_dict["image_proj"])

        faster_set_state_dict(self.image_proj_model, state_dict["image_proj"])

        self.image_proj_model_in_features = image_emb_dim

        # Unet
        class AttnProcessor_Layer(nn.Layer):
            def __init__(self):
                super().__init__()

            __call__ = AttnProcessor.__call__

        class AttnProcessor2_5_Layer(nn.Layer):
            def __init__(self, attention_op=None):
                super().__init__()
                assert attention_op in [None, "math", "auto", "flash", "cutlass", "memory_efficient"]
                self.attention_op = attention_op

            __call__ = AttnProcessor2_5.__call__

        attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor2_5_Layer() if is_ppxformers_available() else AttnProcessor_Layer()
            else:
                attn_processor_class = (
                    IPAdapterAttnProcessor2_5 if is_ppxformers_available() else IPAdapterAttnProcessor
                )
                attn_procs[name] = attn_processor_class(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=scale,
                    num_tokens=num_tokens,
                ).to(dtype=self.dtype)
        self.unet.set_attn_processor(attn_procs)

        ip_layers = nn.LayerList(sublayers=self.unet.attn_processors.values())
        if from_diffusers:
            convert_pytorch_state_dict_to_paddle(ip_layers, state_dict["ip_adapter"])

        faster_set_state_dict(ip_layers, state_dict["ip_adapter"])
        self.unet.to(dtype=self.dtype)

    def set_ip_adapter_scale(self, scale):
        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        for attn_processor in unet.attn_processors.values():
            if isinstance(attn_processor, IPAdapterAttnProcessor):
                attn_processor.scale = scale

    def _encode_prompt_image_emb(self, prompt_image_emb, num_images_per_prompt, do_classifier_free_guidance):

        if isinstance(prompt_image_emb, paddle.Tensor):
            prompt_image_emb = prompt_image_emb.copy()
        else:
            prompt_image_emb = paddle.to_tensor(data=prompt_image_emb)

        prompt_image_emb = prompt_image_emb.cast(dtype=self.image_proj_model.latents.dtype)
        prompt_image_emb = prompt_image_emb.reshape([1, -1, self.image_proj_model_in_features])

        if do_classifier_free_guidance:
            prompt_image_emb = paddle.concat(x=[paddle.zeros_like(x=prompt_image_emb), prompt_image_emb], axis=0)
        else:
            prompt_image_emb = paddle.concat(x=[prompt_image_emb], axis=0)

        prompt_image_emb = self.image_proj_model(prompt_image_emb)
        bs_embed, seq_len, _ = prompt_image_emb.shape
        prompt_image_emb = prompt_image_emb.tile(repeat_times=([1, num_images_per_prompt, 1]))
        prompt_image_emb = prompt_image_emb.reshape([bs_embed * num_images_per_prompt, seq_len, -1])
        return prompt_image_emb

    @paddle.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        pooled_prompt_embeds: Optional[paddle.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[paddle.Tensor] = None,
        image_embeds: Optional[paddle.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        ip_adapter_scale=None,
        low_gpu_mem_usage: bool = True,
        **kwargs
    ):
        """
        Function invoked when calling the pipeline for generation.
        Only the parameters introduced by InstantID are discussed here.
        For explanations of the previous parameters in StableDiffusionXLControlNetPipeline, please refer to https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/ppdiffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py


        Args:
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            low_gpu_mem_usage (`bool`, *optional*, defaults to `True`):
                Whether to use low memory usage mode.
                if True, some modules will be released from GPU to CPU when computing, that will require less GPU memory.
        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned containing the output images.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        controlnet = self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 0. set ip_adapter_scale
        if ip_adapter_scale is not None:
            self.set_ip_adapter_scale(ip_adapter_scale)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            image,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )

        guess_mode = guess_mode or global_pool_conditions

        # 3.1 Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        if low_gpu_mem_usage:
            self.image_proj_model.to(paddle.get_device())

        # 3.2 Encode image prompt
        prompt_image_emb = self._encode_prompt_image_emb(
            image_embeds, num_images_per_prompt, self.do_classifier_free_guidance
        )

        if low_gpu_mem_usage:
            self.image_proj_model.to("cpu")
            paddle.device.cuda.empty_cache()

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []
            for image_ in image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                images.append(image_)
            image = images
            height, width = image[0].shape[-2:]
        else:
            assert False

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
        )

        # 6.1 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = paddle.to_tensor(data=self.guidance_scale - 1).tile(
                repeat_times=([batch_size * num_images_per_prompt])
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            )

        # 7. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                (1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e))
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 7.2 Prepare added time ids & embeddings
        if isinstance(image, list):
            original_size = original_size or image[0].shape[-2:]
        else:
            original_size = original_size or image.shape[-2:]

        target_size = target_size or (height, width)
        add_text_embeds = pooled_prompt_embeds

        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        original_size = tuple(original_size)
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = paddle.concat(x=[negative_prompt_embeds, prompt_embeds], axis=0)
            add_text_embeds = paddle.concat(x=[negative_pooled_prompt_embeds, add_text_embeds], axis=0)
            add_time_ids = paddle.concat(x=[negative_add_time_ids, add_time_ids], axis=0)

        prompt_embeds = prompt_embeds.cast(self.dtype)
        add_text_embeds = add_text_embeds.cast(self.dtype)
        add_time_ids = add_time_ids.tile(repeat_times=([batch_size * num_images_per_prompt, 1]))
        encoder_hidden_states = paddle.concat(x=[prompt_embeds, prompt_image_emb], axis=1)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        if low_gpu_mem_usage:
            self.unet.to(paddle.get_device())

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = paddle.concat(x=[latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                # controlnet(s) inference
                if guess_mode and self.do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(chunks=2)[1]
                    controlnet_added_cond_kwargs = {
                        "text_embeds": add_text_embeds.chunk(chunks=2)[1],
                        "time_ids": add_time_ids.chunk(chunks=2)[1],
                    }
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds
                    controlnet_added_cond_kwargs = added_cond_kwargs

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [(c * s) for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=prompt_image_emb,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    added_cond_kwargs=controlnet_added_cond_kwargs,
                    return_dict=False,
                )

                if guess_mode and self.do_classifier_free_guidance:
                    # Inferred ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [
                        paddle.concat(x=[paddle.zeros_like(x=d), d]) for d in down_block_res_samples
                    ]
                    mid_block_res_sample = paddle.concat(
                        x=[paddle.zeros_like(x=mid_block_res_sample), mid_block_res_sample]
                    )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(chunks=2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0].cast(
                    self.dtype
                )

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if low_gpu_mem_usage:
            self.unet.to("cpu")
            paddle.device.cuda.empty_cache()

        if not output_type == "latent":
            needs_upcasting = self.vae.dtype == paddle.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.cast(next(iter(self.vae.post_quant_conv.named_parameters()))[1].dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return StableDiffusionXLPipelineOutput(images=image)

import random
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import paddle
from einops import rearrange
from paddlenlp.transformers import CLIPTokenizer, CLIPTextModel
from ...loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from ...models.modelscope_autoencoder_img2vid import AutoencoderKL_imgtovideo, get_first_stage_encoding
from ...models.modelscope_gaussion_sdedit import GaussianDiffusion_SDEdit, noise_schedule
from ...models.modelscope_st_unet_video2video import Vid2VidSTUNet
from ...utils import logging, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline
from . import VideoToVideoModelscopePipelineOutput
import paddle.nn.functional as F
from paddle.vision import transforms

logger = logging.get_logger(__name__)
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import paddle
        >>> from ppdiffusers import VideoToVideoModelscopePipeline
        >>> from ppdiffusers.utils import export_to_video

        >>> pipe = VideoToVideoModelscopePipeline.from_pretrained(
        ...     "/home/aistudio/video_to_video")
        >>> video_path = 'test.mp4'
        >>> prompt = "A panda is surfing on the sea"
        >>> video_frames = pipe(prompt=prompt,video_path=video_path).frames
        >>> video_path = export_to_video(video_frames)
        >>> video_path
        ```
"""


def tensor2vid(video: paddle.Tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> List[np.ndarray]:
    # This code is copied from https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78
    # reshape to ncfhw
    mean = paddle.to_tensor(data=mean).reshape([1, -1, 1, 1, 1])
    std = paddle.to_tensor(data=std).reshape([1, -1, 1, 1, 1])
    # unnormalize back to [0,1]
    video = video.multiply(std).add(y=paddle.to_tensor(mean))
    video.clip_(min=0, max=1)
    # prepare the final outputs
    i, c, f, h, w = video.shape
    images = video.transpose(perm=[2, 3, 0, 4, 1]).reshape([f, h, i * w, c])
    images = images.unbind(axis=0)
    images = [(image.cpu().numpy() * 255).astype("uint8") for image in images]
    return images


class VideoToVideoModelscopePipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin):
    """
    Pipeline for video-to-video generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL_imgtovideo`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer (`CLIPTokenizer`):
            A [`~transformers.CLIPTokenizer`] to tokenize text.
        unet ([`Vid2VidSDUNet`]):
            A [`Vid2VidSDUNet`] to denoise the encoded video latents.

    """

    def __init__(
            self,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            vae: AutoencoderKL_imgtovideo,
            unet: Vid2VidSTUNet,
    ):
        super().__init__()
        self.register_modules(text_encoder=text_encoder, tokenizer=tokenizer, vae=vae, unet=unet)

        self.seed = self.vae.config.seed
        self.batch_size = self.vae.config.batch_size
        self.target_fps = self.vae.config.target_fps
        self.max_frames = self.vae.config.max_frames
        self.latent_hei = self.vae.config.latent_hei
        self.latent_wid = self.vae.config.latent_wid
        self.vit_resolution = self.vae.config.vit_resolution
        self.vit_mean = self.vae.config.vit_mean
        self.vit_std = self.vae.config.vit_std
        self.negative_prompt = self.vae.config.negative_prompt
        self.positive_prompt = self.vae.config.positive_prompt
        self.solver_mode = self.vae.config.solver_mode
        paddle.seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.vid_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
        
        # [diffusion]
        sigmas = noise_schedule(
            schedule='logsnr_cosine_interp',
            n=1000,
            zero_terminal_snr=True,
            scale_min=2.0,
            scale_max=4.0)
        diffusion = GaussianDiffusion_SDEdit(
            sigmas=sigmas, prediction_type='v')
        self.diffusion = diffusion

    def _encode_prompt(
            self,
            prompt,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[paddle.Tensor] = None,
            negative_prompt_embeds: Optional[paddle.Tensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pd",
            )

            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pd").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not paddle.equal_all(
                    text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask
            else:
                attention_mask = None
            prompt_embeds = self.text_encoder(
                text_input_ids,
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.cast(self.text_encoder.dtype)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.tile([1, num_videos_per_prompt, 1])
        prompt_embeds = prompt_embeds.reshape([bs_embed * num_videos_per_prompt, seq_len, -1])

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pd",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids,
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.cast(self.text_encoder.dtype)

            negative_prompt_embeds = negative_prompt_embeds.tile([1, num_videos_per_prompt, 1])
            negative_prompt_embeds = negative_prompt_embeds.reshape([batch_size * num_videos_per_prompt, seq_len, -1])

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = paddle.concat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def input_preprocess(self,
                         vid_path,
                         prompt,
                         num_images_per_prompt,
                         do_classifier_free_guidance,
                         ):
        if prompt is None:
            prompt = ''
        caption = prompt + self.positive_prompt
        y = self._encode_prompt(
            caption,
            num_images_per_prompt,
            do_classifier_free_guidance,
        )

        max_frames = self.max_frames

        capture = cv2.VideoCapture(vid_path)
        _fps = capture.get(cv2.CAP_PROP_FPS)
        sample_fps = _fps
        _total_frame_num = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        stride = round(_fps / sample_fps)
        start_frame = 0

        pointer = 0
        frame_list = []
        while len(frame_list) < max_frames:
            ret, frame = capture.read()
            pointer += 1
            if (not ret) or (frame is None):
                break
            if pointer < start_frame:
                continue
            if pointer >= _total_frame_num + 1:
                break
            if (pointer - start_frame) % stride == 0:
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame = frame[:, :, ::-1]
                frame_list.append(frame)
        capture.release()

        video_data=paddle.stack([self.vid_trans(u) for u in frame_list],axis=0)

        return {'video_data': video_data, 'y': y}

    @paddle.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            video_path: str = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_frames: int = 16,
            num_inference_steps: int = 50,
            guidance_scale: float = 9.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            eta: float = 0.0,
            generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
            latents: Optional[paddle.Tensor] = None,
            prompt_embeds: Optional[paddle.Tensor] = None,
            negative_prompt_embeds: Optional[paddle.Tensor] = None,
            output_type: Optional[str] = "np",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):

        num_images_per_prompt = 1
        do_classifier_free_guidance = False

        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        negative_y = self._encode_prompt(
            self.negative_prompt,
            num_images_per_prompt,
            do_classifier_free_guidance
        )

        # input_process
        input_data = self.input_preprocess(
            vid_path=video_path,
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance
        )
        video_data = input_data['video_data']
        y = input_data['y']
        video_data = F.interpolate(
            video_data, size=(720, 1280), mode='bilinear')
        video_data = video_data.unsqueeze(0)
        video_data = paddle.to_tensor(video_data,place=negative_y.place)

        batch_size, frames_num, _, _, _ = video_data.shape
        video_data = rearrange(video_data, 'b f c h w -> (b f) c h w')

        video_data_list = paddle.chunk(
            video_data, video_data.shape[0] // 1, axis=0)

        with paddle.no_grad():
            decode_data = []
            for vd_data in video_data_list:
                encoder_posterior = self.vae.encode(vd_data)
                tmp = get_first_stage_encoding(encoder_posterior.latent_dist).detach()
                decode_data.append(tmp)
            video_data_feature = paddle.concat(decode_data, axis=0)
            video_data_feature = rearrange(
                video_data_feature, '(b f) c h w -> b c f h w', b=batch_size)
        paddle.device.cuda.empty_cache()

        with paddle.amp.auto_cast(enable=True):
            total_noise_levels = 600
            t = paddle.randint(
                total_noise_levels - 1,
                total_noise_levels, (1,),
                dtype=paddle.int64)

            noise = paddle.randn(shape=video_data_feature.shape,dtype=video_data_feature.dtype)
            noised_lr = self.diffusion.diffuse(video_data_feature, t, noise)
            model_kwargs = [{'y': y}, {'y': negative_y}]

            gen_vid = self.diffusion.sample(
                noise=noised_lr,
                model=self.unet,
                model_kwargs=model_kwargs,
                guide_scale=7.5,
                guide_rescale=0.2,
                solver='dpmpp_2m_sde' if self.solver_mode == 'fast' else 'heun',
                steps=30 if self.solver_mode == 'fast' else 50,
                t_max=total_noise_levels - 1,
                t_min=0,
                discretization='trailing')

            paddle.device.cuda.empty_cache()

            scale_factor = 0.18215
            vid_tensor_feature = 1. / scale_factor * gen_vid

            vid_tensor_feature = rearrange(vid_tensor_feature,
                                           'b c f h w -> (b f) c h w')
            vid_tensor_feature_list = paddle.chunk(
                vid_tensor_feature, vid_tensor_feature.shape[0] // 2, axis=0)
            decode_data = []
            for vd_data in vid_tensor_feature_list:
                tmp = self.vae.decode(vd_data).sample
                decode_data.append(tmp)
            vid_tensor_gen = paddle.concat(decode_data, axis=0)

        gen_video = rearrange(
            vid_tensor_gen, '(b f) c h w -> b c f h w', b=self.batch_size)

        video = tensor2vid(gen_video)

        if not return_dict:
            return (video,)
        return VideoToVideoModelscopePipelineOutput(frames=video)

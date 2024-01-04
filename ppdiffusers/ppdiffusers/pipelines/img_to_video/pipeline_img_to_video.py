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

import random
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import paddle
from einops import rearrange
from paddlenlp.transformers import CLIPVisionModelWithProjection
from paddlenlp.transformers.image_transforms import convert_to_rgb, normalize, resize
from paddlenlp.transformers.image_utils import to_numpy_array
from PIL import Image

from ...loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from ...models.modelscope_autoencoder_img2vid import AutoencoderKL_imgtovideo
from ...models.modelscope_gaussian_diffusion import GaussianDiffusion, beta_schedule
from ...models.modelscope_st_unet import STUNetModel
from ...utils import logging, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline
from . import ImgToVideoSDPipelineOutput

logger = logging.get_logger(__name__)
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import paddle
        >>> from ppdiffusers import ImgToVideoSDPipeline
        >>> from ppdiffusers.utils import export_to_video,load_image

        >>> pipe = ImgToVideoSDPipeline.from_pretrained(
        ...     "Yangchanghui/img_to_video_paddle", paddle_dtype=paddle.float32
        ... )

        >>> img = load_image('test.jpg')
        >>> video_frames = pipe(img).frames
        >>> video_path = export_to_video(video_frames)
        >>> video_path
        ```
"""


def center_crop_wide(img, size):
    scale = min(img.size[0] / size[0], img.size[1] / size[1])
    img = img.resize((round(img.width // scale), round(img.height // scale)), resample=Image.BOX)
    x1 = (img.width - size[0]) // 2
    y1 = (img.height - size[1]) // 2
    img = img.crop((x1, y1, x1 + size[0], y1 + size[1]))
    return img


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


class ImgToVideoSDPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin):
    """
    Pipeline for img-to-video generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL_imgtovideo`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        img_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen img-encoder ([clip-vit-H-patch14](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`SFUNetModel`]):
            A [`SFUNetModel`] to denoise the encoded video latents.

    """

    def __init__(
        self,
        vae: AutoencoderKL_imgtovideo,
        img_encoder: CLIPVisionModelWithProjection,
        unet: STUNetModel,
    ):
        super().__init__()
        self.register_modules(vae=vae, img_encoder=img_encoder, unet=unet)
        self.seed = self.vae.config.seed
        self.batch_size = self.vae.config.batch_size
        self.target_fps = self.vae.config.target_fps
        self.max_frames = self.vae.config.max_frames
        self.latent_hei = self.vae.config.latent_hei
        self.latent_wid = self.vae.config.latent_wid
        self.resolution_crop = self.vae.config.resolution_crop
        self.vit_resolution = self.vae.config.vit_resolution
        self.vit_mean = self.vae.config.vit_mean
        self.vit_std = self.vae.config.vit_std
        self.beta_type = self.vae.config.beta_type
        self.num_timesteps = self.vae.config.num_timesteps
        self.init_beta = self.vae.config.init_beta
        self.last_beta = self.vae.config.last_beta
        self.mean_type = self.vae.config.mean_type
        self.var_type = self.vae.config.var_type
        self.loss_type = self.vae.config.loss_type
        self.noise_strength = self.vae.config.noise_strength
        self.input_dim = self.vae.config.input_dim
        self.ddim_timesteps = self.vae.config.ddim_timesteps
        self.guide_scale = self.vae.config.guide_scale
        self.scale_factor = self.vae.config.scale_factor
        self.decoder_bs = self.vae.config.decoder_bs

        paddle.seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.beta = beta_schedule(
            schedule=self.beta_type,
            n_timestep=self.num_timesteps,
            linear_start=self.init_beta,
            linear_end=self.last_beta,
        )
        self.diffusion = GaussianDiffusion(
            betas=self.beta,
            mean_type=self.mean_type,
            var_type=self.var_type,
            loss_type=self.loss_type,
            rescale_timesteps=False,
            noise_strength=self.noise_strength,
        )

        self.fps_tensor = paddle.to_tensor([self.target_fps] * self.batch_size, dtype=paddle.int64)
        self.zero_feature = paddle.zeros([1, 1, self.input_dim])

    @paddle.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):

        # img preprocess
        image = center_crop_wide(image, [self.resolution_crop, self.resolution_crop])
        image = to_numpy_array(image)
        image = resize(image, [self.vit_resolution, self.vit_resolution]).astype("float32")
        image = image / 255
        image = normalize(image, self.vit_mean, self.vit_std)
        inputs = paddle.to_tensor(image).transpose([2, 0, 1]).unsqueeze(0)
        # clip
        img_embedding = self.img_encoder(inputs).image_embeds.unsqueeze(1)

        noise = self.build_noise()

        model_kwargs = [{"y": img_embedding, "fps": self.fps_tensor}, {"y": self.zero_feature, "fps": self.fps_tensor}]
        gen_video = self.diffusion.ddim_sample_loop(
            noise=noise,
            model=self.unet,
            model_kwargs=model_kwargs,
            guide_scale=self.guide_scale,
            ddim_timesteps=self.ddim_timesteps,
            eta=0.0,
        )
        gen_video = 1.0 / self.scale_factor * gen_video
        gen_video = rearrange(gen_video, "b c f h w -> (b f) c h w")
        chunk_size = min(self.decoder_bs, gen_video.shape[0])
        gen_video_list = paddle.chunk(gen_video, gen_video.shape[0] // chunk_size, axis=0)
        decode_generator = []
        for vd_data in gen_video_list:
            gen_frames = self.vae.decode(vd_data).sample
            decode_generator.append(gen_frames)

        gen_video = paddle.concat(decode_generator, axis=0)
        gen_video = rearrange(gen_video, "(b f) c h w -> b c f h w", b=self.batch_size)

        video = tensor2vid(gen_video)

        if not return_dict:
            return (video,)
        return ImgToVideoSDPipelineOutput(frames=video)

    def build_noise(self):
        noise = paddle.randn([1, 4, self.max_frames, self.latent_hei, self.latent_wid])
        if self.noise_strength > 0:
            b, c, f, *_ = noise.shape
            offset_noise = paddle.randn([b, c, f, 1, 1])
            noise = noise + self.noise_strength * offset_noise
        return noise

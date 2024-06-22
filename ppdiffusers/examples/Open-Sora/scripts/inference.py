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

import paddle
from dataset.utils import save_sample
from models.stdit.stdit2 import STDiT2
from models.text_encoder import T5Encoder
from models.text_encoder.t5 import text_preprocessing
from models.vae import VideoAutoencoderKL
from paddlenlp.trainer import set_seed
from schedulers.iddpm import IDDPM
from utils.config_utils import parse_configs

IMG_FPS = 120


def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts


def main():
    # # ======================================================
    # # 1. cfg and init distributed env
    # # ======================================================

    cfg = parse_configs()
    print(cfg)

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # # init distributed
    if cfg.dtype == "fp16":
        dtype = paddle.float16
    else:
        dtype = paddle.float32

    # - Prompt handling
    if cfg.prompt is None:
        assert cfg.prompt_path is not None, "prompt or prompt_path must be provided"
        prompts = load_prompts(cfg.prompt_path)
    else:
        prompts = cfg.prompt

    input_size = (cfg.num_frames, *cfg.image_size)

    vae = VideoAutoencoderKL(from_pretrained=cfg.vae_pretrained_path, micro_batch_size=cfg.micro_batch_size)
    latent_size = vae.get_latent_size(input_size)

    text_encoder = T5Encoder(from_pretrained=cfg.text_encoder_pretrained_path, model_max_length=cfg.model_max_length)

    model = STDiT2.from_pretrained(pretrained_model_name_or_path=cfg.model_pretrained_path)

    text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

    vae = vae.astype(dtype).eval()
    model = model.astype(dtype).eval()

    # 3.3. build scheduler

    scheduler = IDDPM(
        num_sampling_steps=cfg.num_sampling_steps,
        cfg_scale=cfg.cfg_scale,
        cfg_channel=cfg.cfg_channel,
    )

    model_args = dict()

    image_size = cfg.image_size

    height = paddle.to_tensor([image_size[0]], dtype=dtype).tile((cfg.batch_size))
    width = paddle.to_tensor([image_size[1]], dtype=dtype).tile((cfg.batch_size))
    num_frames = paddle.to_tensor([cfg.num_frames], dtype=dtype).tile((cfg.batch_size))
    ar = paddle.to_tensor([image_size[0] / image_size[1]], dtype=dtype).tile((cfg.batch_size))
    if cfg.num_frames == 1:
        cfg.fps = IMG_FPS

    fps = paddle.to_tensor([cfg.fps], dtype=dtype).tile((cfg.batch_size))
    model_args["height"] = height
    model_args["width"] = width
    model_args["num_frames"] = num_frames
    model_args["ar"] = ar
    model_args["fps"] = fps

    # ======================================================
    # 4. inference
    # ======================================================
    sample_idx = 0
    if cfg.sample_name is not None:
        sample_name = cfg.sample_name
    elif cfg.prompt_as_path:
        sample_name = ""
    else:
        sample_name = "sample"
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 4.1. batch generation
    for i in range(0, len(prompts), cfg.batch_size):
        # 4.2 sample in hidden space
        batch_prompts_raw = prompts[i : i + cfg.batch_size]
        batch_prompts = [text_preprocessing(prompt) for prompt in batch_prompts_raw]

        # handle the last batch
        if len(batch_prompts_raw) < cfg.batch_size and cfg.multi_resolution == "STDiT2":
            model_args["height"] = model_args["height"][: len(batch_prompts_raw)]
            model_args["width"] = model_args["width"][: len(batch_prompts_raw)]
            model_args["num_frames"] = model_args["num_frames"][: len(batch_prompts_raw)]
            model_args["ar"] = model_args["ar"][: len(batch_prompts_raw)]
            model_args["fps"] = model_args["fps"][: len(batch_prompts_raw)]

        # 4.3. diffusion sampling
        old_sample_idx = sample_idx
        # generate multiple samples for each prompt
        for k in range(cfg.num_sample):
            sample_idx = old_sample_idx

            # Skip if the sample already exists
            # This is useful for resuming sampling VBench
            if cfg.prompt_as_path:
                skip = True
                for batch_prompt in batch_prompts_raw:
                    path = os.path.join(save_dir, f"{sample_name}{batch_prompt}")
                    if cfg.num_sample != 1:
                        path = f"{path}-{k}"
                    path = f"{path}.mp4"
                    if not os.path.exists(path):
                        skip = False
                        break
                if skip:
                    continue

            # sampling
            z = paddle.randn(shape=(len(batch_prompts), vae.out_channels, *latent_size))

            samples = scheduler.sample(
                model,
                text_encoder,
                z=z,
                prompts=batch_prompts,
                additional_args=model_args,
            )

            samples = vae.decode(samples.astype(dtype=dtype))

            # 4.4. save samples

            for idx, sample in enumerate(samples):
                print(f"Prompt: {batch_prompts_raw[idx]}")
                if cfg.prompt_as_path:
                    sample_name_suffix = batch_prompts_raw[idx]
                else:
                    sample_name_suffix = f"_{sample_idx}"
                save_path = os.path.join(save_dir, f"{sample_name}{sample_name_suffix}")
                if cfg.num_sample != 1:
                    save_path = f"{save_path}-{k}"

                save_sample(sample.astype(paddle.float32), fps=cfg.fps // cfg.frame_interval, save_path=save_path)
                sample_idx += 1


if __name__ == "__main__":
    main()

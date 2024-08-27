# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

os.environ["FLAGS_use_cuda_managed_memory"] = "true"
import argparse
import datetime

import paddle

from ppdiffusers import StableDiffusion3Pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description=" Use PaddleMIX to accelerate the Stable Diffusion3 image generation model."
    )
    parser.add_argument(
        "--benchmark",
        type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
        default=False,
        help="if benchmark is set to True, measure inference performance",
    )
    parser.add_argument(
        "--inference_optimize",
        type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
        default=False,
        help="If inference_optimize is set to True, all optimizations except Triton are enabled.",
    )
    parser.add_argument(
        "--inference_optimize_triton",
        type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
        default=False,
        help="If inference_optimize_triton is set to True, Triton operator optimized inference is enabled.",
    )
    parser.add_argument(
        "--inference_optimize_origin",
        type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
        default=False,
        help="If inference_optimize_origin is set to True, the original dynamic graph inference optimization is enabled.",
    )
    parser.add_argument("--height", type=int, default=512, help="Height of the generated image.")
    parser.add_argument("--width", type=int, default=512, help="Width of the generated image.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of inference steps.")
    return parser.parse_args()


args = parse_args()

if args.inference_optimize:
    os.environ["INFERENCE_OPTIMIZE"] = "True"
if args.inference_optimize_triton:
    os.environ["INFERENCE_OPTIMIZE_TRITON"] = "True"
if args.inference_optimize_origin:
    os.environ["INFERENCE_OPTIMIZE_ORIGIN"] = "True"


pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    paddle_dtype=paddle.float32,
    # from_hf_hub=True,
    # from_diffusers=True,
)

pipe.text_encoder = paddle.incubate.jit.inference(
    pipe.text_encoder,
    save_model_dir="./tmp/text_encoder",
    cache_static_model=True,
    with_trt=True,
    trt_precision_mode="float16",
    trt_use_static=True,
)

pipe.text_encoder_2 = paddle.incubate.jit.inference(
    pipe.text_encoder_2,
    save_model_dir="./tmp/text_encoder_2",
    cache_static_model=True,
    with_trt=True,
    trt_precision_mode="float16",
    trt_use_static=True,
)


pipe.text_encoder_3 = paddle.incubate.jit.inference(
    pipe.text_encoder_3,
    save_model_dir="./tmp/text_encoder_3_T5",
    cache_static_model=True,
    with_trt=True,
    trt_precision_mode="float16",
    trt_use_static=True,
)


# for vae model
pipe.vae.decode = paddle.incubate.jit.inference(
    pipe.vae.decode,
    save_model_dir="./tmp/vae_static_models",
    cache_static_model=True,
    with_trt=True,
    trt_precision_mode="float16",
    trt_use_static=True,
)

generator = paddle.Generator().manual_seed(42)
prompt = "A cat holding a sign that says hello world"


image = pipe(
    prompt, num_inference_steps=args.num_inference_steps, width=args.width, height=args.height, generator=generator
).images[0]

if args.benchmark:
    # warmup
    for i in range(3):
        image = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            width=args.width,
            height=args.height,
            generator=generator,
        ).images[0]

    repeat_times = 10
    sumtime = 0.0
    for i in range(repeat_times):
        paddle.device.synchronize()
        starttime = datetime.datetime.now()
        image = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            width=args.width,
            height=args.height,
            generator=generator,
        ).images[0]
        paddle.device.synchronize()
        endtime = datetime.datetime.now()
        duringtime = endtime - starttime
        duringtime = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
        sumtime += duringtime
        print("The this end to end time : ", duringtime, "ms")

    print("The ave end to end time : ", sumtime / repeat_times, "ms")
    cuda_mem_after_used = paddle.device.cuda.max_memory_allocated() / (1024**3)
    print(f"Max used CUDA memory : {cuda_mem_after_used:.3f} GiB")

image.save("text_to_image_generation-stable_diffusion_3-result.png")

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

import argparse
import datetime
import os

import paddle
from paddlenlp.trainer import set_seed

from ppdiffusers import DDIMScheduler, DiTPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description=" Use PaddleMIX to accelerate the Diffusion Transformer image generation model."
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
        default=True,
        help="If inference_optimize_triton is set to True, Triton operator optimized inference is enabled.",
    )
    return parser.parse_args()


args = parse_args()

if args.inference_optimize:
    os.environ["INFERENCE_OPTIMIZE"] = "True"
if args.inference_optimize_triton:
    os.environ["INFERENCE_OPTIMIZE_TRITON"] = "True"

dtype = paddle.float16
pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", paddle_dtype=dtype)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


if args.inference_optimize:
    # optimize the transformer using paddle.incubate.jit.inference
    pipe.transformer = paddle.incubate.jit.inference(
        pipe.transformer,
        enable_new_ir=True,
        save_model_dir="./tmp/dit",
        cache_static_model=True,
        exp_enable_use_cutlass=True,
        delete_pass_lists=["add_norm_fuse_pass"],
    )
    pipe.vae.decode = paddle.incubate.jit.inference(
        pipe.vae.decode,
        enable_new_ir=True,
        save_model_dir="./tmp/dit/vae",
        cache_static_model=True,
        exp_enable_use_cutlass=True,
    )
set_seed(42)
words = ["golden retriever"]  # class_ids [207]
class_ids = pipe.get_label_ids(words)
image = pipe(class_labels=class_ids, num_inference_steps=25).images[0]

if args.benchmark:

    # warmup
    for i in range(3):
        set_seed(42)
        image = pipe(class_labels=class_ids, num_inference_steps=25).images[0]

    repeat_times = 5

    for i in range(repeat_times):
        paddle.device.synchronize()
        starttime = datetime.datetime.now()
        set_seed(42)
        image = pipe(class_labels=class_ids, num_inference_steps=25).images[0]
        paddle.device.synchronize()
        endtime = datetime.datetime.now()

        duringtime = endtime - starttime
        time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
        print("DIT end to end time : ", time_ms, "ms")

image.save("class_conditional_image_generation-dit-result.png")

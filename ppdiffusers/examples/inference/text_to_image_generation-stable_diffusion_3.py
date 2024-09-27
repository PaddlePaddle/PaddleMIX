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
import argparse
import paddle
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
        "--inference_optimize_cfgp",
        type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
        default=False,
        help="If inference_optimize_cfgp is set to True, Classifier-Free Guidance Parallel is enabled and dual-GPU acceleration is used.",
    )
    parser.add_argument("--height", type=int, default=512, help="Height of the generated image.")
    parser.add_argument("--width", type=int, default=512, help="Width of the generated image.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--dtype", type=str, default="float32", help="Inference data types.")

    return parser.parse_args()


args = parse_args()

if args.inference_optimize:
    os.environ["INFERENCE_OPTIMIZE"] = "True"
    os.environ["INFERENCE_OPTIMIZE_TRITON"] = "True"
if args.inference_optimize_cfgp:
    os.environ["INFERENCE_OPTIMIZE_CFGP"] = "True"
if args.dtype == "float32":
    inference_dtype = paddle.float32
elif args.dtype == "float16":
    inference_dtype = paddle.float16


if args.inference_optimize_cfgp:
    # python3.8 -m paddle.distributed.launch --gpus "0,1,2,3" demo.py 

    from paddle.distributed import fleet
    from paddle.distributed.fleet.utils import recompute
    import numpy as np
    import random
    import paddle.distributed as dist
    import paddle.distributed.fleet as fleet
    strategy = fleet.DistributedStrategy()
    # 设置4路张量模型并行
    model_parallel_size = 2
    data_parallel_size = 1
    strategy.hybrid_configs = {
    "dp_degree": data_parallel_size,
    "mp_degree": model_parallel_size,
    "pp_degree": 1
    }
    # 注意 strategy 是这里传递的，动态图只能这里，静态图还可以在 distributed_optimizer 里传
    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    mp_id = hcg.get_model_parallel_rank()
    rank_id = dist.get_rank()



os.environ["FLAGS_use_cuda_managed_memory"] = "true"
import datetime
from ppdiffusers import StableDiffusion3Pipeline


pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    paddle_dtype=inference_dtype,
)

pipe.transformer = paddle.incubate.jit.inference(
    pipe.transformer,
    save_model_dir="./tmp/sd3",
    enable_new_ir=True,
    cache_static_model=True,
    exp_enable_use_cutlass=True,
    delete_pass_lists=["add_norm_fuse_pass"],
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
        print("SD3 end to end time : ", duringtime, "ms")

    print("SD3 ave end to end time : ", sumtime / repeat_times, "ms")
    cuda_mem_after_used = paddle.device.cuda.max_memory_allocated() / (1024**3)
    print(f"Max used CUDA memory : {cuda_mem_after_used:.3f} GiB")

if args.inference_optimize_cfgp:
    if rank_id == 0:
        print(f"Max used CUDA_rank_1 memory : {cuda_mem_after_used:.3f} GiB")
        image.save("text_to_image_generation-stable_diffusion_3-result0.png")
    else:
        print(f"Max used CUDA_rank_2 memory : {cuda_mem_after_used:.3f} GiB")
        image.save("text_to_image_generation-stable_diffusion_3-result1.png")
else:
    image.save("text_to_image_generation-stable_diffusion_3-result_single.png")

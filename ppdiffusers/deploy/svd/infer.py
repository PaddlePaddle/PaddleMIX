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

import argparse
import os
import time

import paddle

# isort: split
import numpy as np
import paddle.inference as paddle_infer
from paddlenlp.trainer.argparser import strtobool
from tqdm.auto import trange

from ppdiffusers import (  # noqa
    DiffusionPipeline,
    PaddleInferStableVideoDiffusionPipeline,
)
from ppdiffusers.utils import load_image


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="runwayml/stable-diffusion-v1-5@paddleinfer",
        help="The model directory of diffusion_model.",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=50,
        help="The number of unet inference steps.",
    )
    parser.add_argument(
        "--benchmark_steps",
        type=int,
        default=10,
        help="The number of performance benchmark steps.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="paddle_tensorrt",
        choices=["paddle", "paddle_tensorrt"],
        help="The inference runtime backend of unet model and text encoder model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=[
            "cpu",
            "gpu",
            "huawei_ascend_npu",
            "kunlunxin_xpu",
        ],
        help="The inference runtime device of models.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="img2video",
        choices=[
            "img2video",
            "all",
        ],
        help="The task can be one of [text2img, all]. ",
    )
    parser.add_argument("--use_fp16", type=strtobool, default=True, help="Wheter to use FP16 mode")
    parser.add_argument("--device_id", type=int, default=0, help="The selected gpu id. -1 means use cpu")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="preconfig-euler-ancestral",
        choices=[
            "pndm",
            "lms",
            "euler",
            "euler-ancestral",
            "preconfig-euler-ancestral",
            "dpm-multi",
            "dpm-single",
            "unipc-multi",
            "ddim",
            "ddpm",
            "deis-multi",
            "heun",
            "kdpm2-ancestral",
            "kdpm2",
        ],
        help="The scheduler type of stable diffusion.",
    )
    parser.add_argument("--height", type=int, default=576, help="Height of input image")
    parser.add_argument("--width", type=int, default=1024, help="Width of input image")
    parser.add_argument(
        "--tune",
        type=strtobool,
        default=False,
        help="Whether to tune the shape of tensorrt engine.",
    )

    return parser.parse_args()


def create_paddle_inference_runtime(
    model_dir="",
    model_name="",
    use_trt=False,
    precision_mode=paddle_infer.PrecisionType.Half,
    device_id=0,
    disable_paddle_trt_ops=[],
    disable_paddle_pass=[],
    workspace=24 * 1024 * 1024 * 1024,
    tune=False,
):
    config = paddle_infer.Config()
    config.enable_new_executor()
    config.enable_memory_optim()
    shape_file = f"{model_dir}/{model_name}/shape_range_info.pbtxt"
    if tune:
        config.collect_shape_range_info(shape_file)
        # config.switch_ir_optim(False)
    else:
        config.enable_new_executor()

    if device_id != -1:
        config.use_gpu()
        config.enable_use_gpu(memory_pool_init_size_mb=2000, device_id=device_id, precision_mode=precision_mode)
    for pass_name in disable_paddle_pass:
        config.delete_pass(pass_name)
    if use_trt:
        config.enable_tensorrt_engine(
            workspace_size=workspace,
            precision_mode=precision_mode,
            max_batch_size=1,
            min_subgraph_size=3,
            use_static=True,
        )
        config.enable_tensorrt_memory_optim()
        config.enable_tuned_tensorrt_dynamic_shape(shape_file, True)
        cache_file = os.path.join(model_dir, model_name, "_opt_cache/")
        config.set_optim_cache_dir(cache_file)
        if precision_mode != paddle_infer.PrecisionType.Half:
            only_fp16_passes = [
                "trt_cross_multihead_matmul_fuse_pass",
                "trt_flash_multihead_matmul_fuse_pass",
                "preln_elementwise_groupnorm_act_pass",
                "elementwise_groupnorm_act_pass",
            ]
            for curr_pass in only_fp16_passes:
                config.delete_pass(curr_pass)
    return config


def main(args):
    if args.device_id == -1:
        paddle.set_device("cpu")
    else:
        paddle.set_device(f"gpu:{args.device_id}")
    seed = 1024

    only_fp16_passes = [
        "trt_cross_multihead_matmul_fuse_pass",
        "trt_flash_multihead_matmul_fuse_pass",
        "preln_elementwise_groupnorm_act_pass",
        "elementwise_groupnorm_act_pass",
        "auto_mixed_precision_pass",
    ]
    no_need_passes = [
        "trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass",
        "add_support_int8_pass",
        "trt_cross_multihead_matmul_fuse_pass",
        "trt_flash_multihead_matmul_fuse_pass",
        "preln_elementwise_groupnorm_act_pass",
        "elementwise_groupnorm_act_pass",
        "groupnorm_act_pass",
        "conv_elementwise_add_fuse_pass",
    ]
    paddle_delete_passes = dict(  # noqa
        text_encoder=only_fp16_passes + no_need_passes if not args.use_fp16 else no_need_passes,
        text_encoder_2=only_fp16_passes + no_need_passes if not args.use_fp16 else no_need_passes,
        vae_encoder=only_fp16_passes + [] if args.use_fp16 else [],
        vae_decoder=only_fp16_passes + no_need_passes if not args.use_fp16 else no_need_passes,
        unet=only_fp16_passes + no_need_passes if not args.use_fp16 else no_need_passes,
        image_encoder=only_fp16_passes + no_need_passes if not args.use_fp16 else no_need_passes,
    )
    args.use_trt = args.backend == "paddle_tensorrt"
    precision_mode = paddle_infer.PrecisionType.Half if args.use_fp16 else paddle_infer.PrecisionType.Float32
    infer_configs = dict(
        vae_encoder=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            model_name="vae_encoder",
            use_trt=False,
            precision_mode=paddle_infer.PrecisionType.Half,
            device_id=args.device_id,
            disable_paddle_pass=paddle_delete_passes.get("vae_encoder", []),
            tune=False,
        ),
        vae_decoder=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            model_name="vae_decoder",
            use_trt=False,
            precision_mode=paddle_infer.PrecisionType.Float32,
            device_id=args.device_id,
            disable_paddle_pass=no_need_passes,
            tune=False,
        ),
        unet=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            model_name="unet",
            use_trt=args.use_trt,
            precision_mode=precision_mode,
            device_id=args.device_id,
            disable_paddle_pass=no_need_passes,
            tune=args.tune,
        ),
        image_encoder=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            model_name="image_encoder",
            use_trt=False,
            precision_mode=paddle_infer.PrecisionType.Half,
            device_id=args.device_id,
            disable_paddle_pass=paddle_delete_passes.get("image_encoder", []),
            tune=False,
        ),
    )
    pipe = PaddleInferStableVideoDiffusionPipeline.from_pretrained(
        args.model_dir,
        infer_configs=infer_configs,
        use_optim_cache=False,
    )
    pipe.set_progress_bar_config(disable=False)
    width = args.width
    height = args.height

    folder = f"results-{args.backend}"
    os.makedirs(folder, exist_ok=True)

    if args.task_name in ["img2video", "all"]:
        # img2video
        img_url = "https://paddlenlp.bj.bcebos.com/models/community/hf-internal-testing/diffusers-images/rocket.png"
        init_image = load_image(img_url)
        time_costs = []
        # warmup
        print("==> Warmup.")
        pipe(
            image=init_image,
            num_inference_steps=3,
            height=height,
            width=width,
            fps=7,
            decode_chunk_size=2,
        )
        print("==> Test img2video performance.")
        for step in trange(args.benchmark_steps):
            start = time.time()
            paddle.seed(seed)
            frames = pipe(
                image=init_image,
                num_inference_steps=args.inference_steps,
                height=height,
                width=width,
                fps=7,
                decode_chunk_size=2,
            ).frames
            latency = time.time() - start
            time_costs += [latency]
            # print(f"No {step:3d} time cost: {latency:2f} s")
        print(
            f"Use fp16: {'true' if args.use_fp16 else 'false'}, "
            f"Mean iter/sec: {1 / (np.mean(time_costs) / args.inference_steps):2f} it/s, "
            f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
            f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
        )
        frames[0][0].save(f"{folder}/test_svd.gif", save_all=True, append_images=frames[0][1:], loop=0)


if __name__ == "__main__":

    args = parse_arguments()
    main(args)

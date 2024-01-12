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
import random

# isort: split
import paddle.inference as paddle_infer
import numpy as np
from paddlenlp.trainer.argparser import strtobool
from tqdm.auto import trange

from ppdiffusers import (  # noqa
    DiffusionPipeline,
    PaddleInferStableDiffusionXLPipeline,
)
from ppdiffusers.utils import load_image


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="runwayml/stable-diffusion-v1-5@fastdeploy",
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
        default=1,
        help="The number of performance benchmark steps.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="paddle_tensorrt",
        # Note(zhoushunjie): Will support 'tensorrt' soon.
        choices=["onnx_runtime", "paddle", "paddlelite", "paddle_tensorrt"],
        help="The inference runtime backend of unet model and text encoder model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        # Note(shentanyue): Will support more devices.
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
        default="text2img",
        choices=[
            "text2img",
            "img2img",
            "inpaint",
            "all",
        ],
        help="The task can be one of [text2img, img2img, inpaint, pix2pix, all]. ",
    )
    parser.add_argument(
        "--parse_prompt_type",
        type=str,
        default="lpw",
        choices=[
            "raw",
            "lpw",
        ],
        help="The parse_prompt_type can be one of [raw, lpw]. ",
    )
    parser.add_argument("--use_fp16", type=strtobool, default=True, help="Wheter to use FP16 mode")
    parser.add_argument("--use_bf16", type=strtobool, default=False, help="Wheter to use BF16 mode")
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
    parser.add_argument(
        "--infer_op",
        type=str,
        default="zero_copy_infer",
        choices=[
            "zero_copy_infer",
            "raw",
            "all",
        ],
        help="The type of infer op.",
    )
    parser.add_argument("--height", type=int, default=1024, help="Height of input image")
    parser.add_argument("--width", type=int, default=1024, help="Width of input image")

    return parser.parse_args()

def create_paddle_inference_runtime(
    model_dir="",
    model_name="",
    use_trt=False,
    dynamic_shape=None,
    precision_mode=paddle_infer.PrecisionType.Half,
    device_id=0,
    disable_paddle_trt_ops=[],
    disable_paddle_pass=[],
    workspace=24*1024*1024*1024,
    tune=False,
):
    config = paddle_infer.Config()
    config.enable_new_executor()
    config.enable_memory_optim()
    shape_file = f"{model_dir}/{model_name}/shape_range_info.pbtxt"
    if tune:
        config.collect_shape_range_info(shape_file)
    if device_id != -1:
        config.use_gpu()
        config.enable_use_gpu(memory_pool_init_size_mb=2000, device_id=device_id, precision_mode=precision_mode)
    for pass_name in disable_paddle_pass:
        config.delete_pass(pass_name)
    if use_trt:
        config.enable_tensorrt_engine(workspace_size=workspace,
                                      precision_mode=precision_mode,
                                      max_batch_size=1,
                                      min_subgraph_size=3,
                                      uuse_static=True)
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
        paddle_stream = None
    else:
        paddle.set_device(f"gpu:{args.device_id}")
    seed = 1024
    vae_in_channels = 4
    text_encoder_max_length = 77
    unet_max_length = text_encoder_max_length * 3  # lpw support max_length is 77x3
    min_image_size = 1024
    max_image_size = 1024
    max_image_size = max(min_image_size, max_image_size)
    hidden_states = 2048
    unet_in_channels = 4
    bs = 2

    text_encoder_dynamic_shape = {
        "input_ids": {
            "min_shape": [1, text_encoder_max_length],
            "max_shape": [1, text_encoder_max_length],
            "opt_shape": [1, text_encoder_max_length],
        }
    }

    text_encoder_2_dynamic_shape = {
        "input_ids": {
            "min_shape": [1, text_encoder_max_length],
            "max_shape": [1, text_encoder_max_length],
            "opt_shape": [1, text_encoder_max_length],
        }
    }

    vae_encoder_dynamic_shape = {
        "sample": {
            "min_shape": [1, 3, min_image_size, min_image_size],
            "max_shape": [1, 3, max_image_size, max_image_size],
            "opt_shape": [1, 3, min_image_size, min_image_size],
        }
    }

    vae_decoder_dynamic_shape = {
        "latent_sample": {
            "min_shape": [1, vae_in_channels, min_image_size // 8, min_image_size // 8],
            "max_shape": [1, vae_in_channels, max_image_size // 8, max_image_size // 8],
            "opt_shape": [1, vae_in_channels, min_image_size // 8, min_image_size // 8],
        }
    }

    unet_dynamic_shape = {
        "sample": {
            "min_shape": [
                1,
                unet_in_channels,
                min_image_size // 8,
                min_image_size // 8,
            ],
            "max_shape": [
                bs,
                unet_in_channels,
                max_image_size // 8,
                max_image_size // 8,
            ],
            "opt_shape": [
                2,
                unet_in_channels,
                min_image_size // 8,
                min_image_size // 8,
            ],
        },
        "timestep": {
            "min_shape": [1],
            "max_shape": [1],
            "opt_shape": [1],
        },
        "encoder_hidden_states": {
            "min_shape": [1, text_encoder_max_length, hidden_states],
            "max_shape": [bs, unet_max_length, hidden_states],
            "opt_shape": [2, text_encoder_max_length, hidden_states],
        },
        "text_embeds": {
            "min_shape": [1, 1280],
            "max_shape": [bs, 1280],
            "opt_shape": [2, 1280],
        },
        "time_ids": {
            "min_shape": [1, 6],
            "max_shape": [bs, 6],
            "opt_shape": [2, 6],
        },
    }
    # 4. Init runtime
    disable_paddle_pass=['auto_mixed_precision_pass']
    infer_configs = dict(
        text_encoder=create_paddle_inference_runtime(
        model_dir=args.model_dir,
        use_trt=False,
        model_name="text_encoder",
        dynamic_shape=text_encoder_dynamic_shape,
        precision_mode=paddle_infer.PrecisionType.Half,
        device_id=7,
        disable_paddle_trt_ops=["range", "lookup_table_v2"],
        tune=False),
        text_encoder_2=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            use_trt=False,
            model_name="text_encoder_2",
            dynamic_shape=text_encoder_dynamic_shape,
            precision_mode=paddle_infer.PrecisionType.Half,
            device_id=7,
            disable_paddle_trt_ops=["range", "lookup_table_v2"],
            tune=False
        ),
        vae_encoder=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            model_name="vae_encoder",
            use_trt=False,
            precision_mode=paddle_infer.PrecisionType.Half,
            device_id=7,
            tune=False
        ),
        vae_decoder=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            model_name="vae_decoder",
            use_trt=False,
            precision_mode=paddle_infer.PrecisionType.Half,
            device_id=7,
            disable_paddle_pass=disable_paddle_pass,
            tune=False
        ),
        unet=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            model_name="unet",
            use_trt=False,
            precision_mode=paddle_infer.PrecisionType.Half,
            device_id=7,
            tune=False
        ),
    )
    pipe = PaddleInferStableDiffusionXLPipeline.from_pretrained(
        args.model_dir,
        infer_configs=infer_configs,
    )
    pipe.set_progress_bar_config(disable=True)
    # pipe.change_scheduler(args.scheduler)
    parse_prompt_type = args.parse_prompt_type
    width = args.width
    height = args.height

    if args.infer_op == "all":
        infer_op_list = ["zero_copy_infer", "raw"]
    else:
        infer_op_list = [args.infer_op]
    if args.device == "kunlunxin_xpu" or args.backend == "paddle":
        print("When device is kunlunxin_xpu or backend is paddle, we will use `raw` infer op.")
        infer_op_list = ["raw"]

    for infer_op in infer_op_list:
        infer_op_dict = {
            "vae_encoder": infer_op,
            "vae_decoder": infer_op,
            "text_encoder": infer_op,
            "unet": infer_op,
        }
        folder = f"infer_op_{infer_op}_fp16" if args.use_fp16 else f"infer_op_{infer_op}_fp32"
        os.makedirs(folder, exist_ok=True)
        if args.task_name in ["text2img", "all"]:
            # text2img
            prompt = "beautiful scenery nature glass bottle landscape, purple galaxy bottle"
            time_costs = []
            negative_prompt = "text, watermark"
            # warmup
            # pipe(
            #     prompt,
            #     num_inference_steps=20,
            #     height=height,
            #     width=width,
            #     # parse_prompt_type=parse_prompt_type,
            #     # infer_op_dict=infer_op_dict,
            #     negative_prompt=negative_prompt

            # )
            print("==> Test text2img performance.")
            for step in trange(args.benchmark_steps):
                start = time.time()
                paddle.seed(seed)
                images = pipe(
                    prompt,
                    output_type="pil",
                    num_inference_steps=args.inference_steps,
                    height=height,
                    width=width,
                    # parse_prompt_type=parse_prompt_type,
                    # infer_op_dict=infer_op_dict,
                    negative_prompt=negative_prompt
                ).images
                latency = time.time() - start
                time_costs += [latency]
                # print(f"No {step:3d} time cost: {latency:2f} s")
            print(
                f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
                f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
            )
            images[0].save(f"{folder}/text2img___1.png")

        if args.task_name in ["img2img", "all"]:
            # img2img
            img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/sketch-mountains-input.png"
            init_image = load_image(img_url)
            prompt = "A fantasy landscape, trending on artstation"
            time_costs = []
            # warmup
            pipe.img2img(
                prompt,
                image=init_image,
                num_inference_steps=20,
                height=height,
                width=width,
                # parse_prompt_type=parse_prompt_type,
                infer_op_dict=infer_op_dict,
            )
            print("==> Test img2img performance.")
            for step in trange(args.benchmark_steps):
                start = time.time()
                paddle.seed(seed)
                images = pipe.img2img(
                    prompt,
                    image=init_image,
                    num_inference_steps=args.inference_steps,
                    height=height,
                    width=width,
                    parse_prompt_type=parse_prompt_type,
                    infer_op_dict=infer_op_dict,
                ).images
                latency = time.time() - start
                time_costs += [latency]
                # print(f"No {step:3d} time cost: {latency:2f} s")
            print(
                f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
                f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
            )
            images[0].save(f"{folder}/img2img.png")

        if args.task_name in ["inpaint", "all"]:
            img_url = (
                "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
            )
            mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"
            init_image = load_image(img_url)
            mask_image = load_image(mask_url)
            prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
            time_costs = []
            pipe.inpaint(
                prompt,
                image=init_image,
                mask_image=mask_image,
                num_inference_steps=20,
                height=height,
                width=width,
                parse_prompt_type=parse_prompt_type,
                infer_op_dict=infer_op_dict,
            )
            print("==> Test inpaint performance.")
            for step in trange(args.benchmark_steps):
                start = time.time()
                paddle.seed(seed)
                images = pipe.inpaint(
                    prompt,
                    image=init_image,
                    mask_image=mask_image,
                    num_inference_steps=args.inference_steps,
                    height=height,
                    width=width,
                    parse_prompt_type=parse_prompt_type,
                    infer_op_dict=infer_op_dict,
                ).images
                latency = time.time() - start
                time_costs += [latency]
                # print(f"No {step:3d} time cost: {latency:2f} s")
            print(
                f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
                f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
            )

            images[0].save(f"{folder}/inpaint.png")


if __name__ == "__main__":
    seed=2024
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    args = parse_arguments()
    main(args)

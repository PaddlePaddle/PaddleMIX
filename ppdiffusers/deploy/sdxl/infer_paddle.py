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
        default="static_model/stable-diffusion-xl-base-1.0",
        help="The model directory of diffusion_model.",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=30,
        help="The number of unet inference steps.",
    )
    parser.add_argument(
        "--benchmark_steps",
        type=int,
        default=5,
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
        ],
        help="only [text2img]. ",
    )
    parser.add_argument("--use_fp16", type=strtobool, default=True, help="Wheter to use FP16 mode")
    parser.add_argument("--use_bf16", type=strtobool, default=False, help="Wheter to use BF16 mode")
    parser.add_argument("--device_id", type=int, default=7, help="The selected gpu id. -1 means use cpu")
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


'''
关于trt动态shape的使用:
为了使用trt动态shape，paddle_inference提供了三种方式来设置shape信息：
1. 直接使用set_trt_dynamic_shape_info设置输入的shape范围，后面op的shape自动推导。但是这种方式潜在很多未知bug,极不推荐使用。
2. 离线收集：首先在静态图模式下，（下面接口的tune==True）使用collect_shape_range_info收集各OP的shape范围到静态图目录下得到
shape_range_info.pbtxt，然后开启trt,使用enable_tuned_tensorrt_dynamic_shape(path_shape_file, True)使用收集到的shape信息。
强烈推荐使用这种方式。
3. 在线收集：直接开启trt 使用enable_tuned_tensorrt_dynamic_shape()接口，接口参数为空，会自动收集输入的shape信息，但这种方式
比离线收集要慢。


'''


def create_paddle_inference_runtime(
    model_dir="",
    model_name="",
    use_trt=False,
    dynamic_shape=None,
    precision_mode=paddle_infer.PrecisionType.Half,
    device_id=0,
    workspace=24*1024*1024*1024,
    tune=False, #离线收集shape信息
    auto_tune=False, #在线收集信息
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
    if use_trt:
        config.enable_tensorrt_engine(workspace_size=workspace,
                                      precision_mode=precision_mode,
                                      max_batch_size=1,
                                      min_subgraph_size=3,
                                      use_static=True,)
        config.enable_tensorrt_memory_optim()
        if dynamic_shape is None:
            if auto_tune:
                config.enable_tuned_tensorrt_dynamic_shape()
            else:
                if not os.path.exists(shape_file):
                    raise ValueError(f"shape_range_info.pbtxt not found in {model_dir}/{model_name}, you should set dyanmic_shape or collect shape_range_info by auto_tune firstly.")
                config.enable_tuned_tensorrt_dynamic_shape(shape_file, True)
        else:
            if dynamic_shape is None:
                raise ValueError("dynamic_shape should be set when use trt when you don's have shape-file.")
            config.set_trt_dynamic_shape_info(dynamic_shape[0], dynamic_shape[1], dynamic_shape[2])
        cache_path =  f"{model_dir}/{model_name}/_opt_cache"
        config.set_optim_cache_dir(cache_path)
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

    text_encoder_dynamic_min_shape = {
        "input_ids": [1, text_encoder_max_length],
    }
    text_encoder_dynamic_max_shape = {
        "input_ids": [1, text_encoder_max_length],
    }
    text_encoder_dynamic_opt_shape = {
        "input_ids": [1, text_encoder_max_length],
    }

    text_encoder_dynamic_shape = [text_encoder_dynamic_min_shape, text_encoder_dynamic_max_shape, text_encoder_dynamic_opt_shape]

    text_encoder_2_dynamic_min_shape = {
        "input_ids": [1, text_encoder_max_length],
    }
    text_encoder_2_dynamic_max_shape = {
        "input_ids": [1, text_encoder_max_length],
    }
    text_encoder_2_dynamic_opt_shape = {
        "input_ids": [1, text_encoder_max_length],
    }

    text_encoder_2_dynamic_shape = [text_encoder_2_dynamic_min_shape, text_encoder_2_dynamic_max_shape, text_encoder_2_dynamic_opt_shape]

    vae_encoder_dynamic_min_shape = {
        "sample": [1, 3, min_image_size // 8, min_image_size // 8],
    }
    vae_encoder_dynamic_max_shape = {
        "sample": [1, 3, max_image_size // 8, max_image_size // 8],
    }
    vae_encoder_dynamic_opt_shape = {
        "sample": [1, 3, min_image_size // 8, min_image_size // 8],
    }
    vae_encoder_dynamic_shape = [vae_encoder_dynamic_min_shape, vae_encoder_dynamic_max_shape, vae_encoder_dynamic_opt_shape]


    vae_decoder_dynamic_min_shape = {
        "latent_sample": [1, vae_in_channels, min_image_size // 8, min_image_size // 8],
    }
    vae_decoder_dynamic_max_shape = {
        "latent_sample": [1, vae_in_channels, min_image_size // 8, min_image_size // 8],
    }
    vae_decoder_dynamic_opt_shape = {
        "latent_sample": [1, vae_in_channels, min_image_size // 8, min_image_size // 8],
    }
    vae_decoder_dynamic_shape = [vae_decoder_dynamic_min_shape, vae_decoder_dynamic_max_shape, vae_decoder_dynamic_opt_shape]
    
    unet_min_input_shape ={
        "sample": [1, unet_in_channels, min_image_size // 8, min_image_size // 8],
        "timestep": [1],
        "encoder_hidden_states": [1, text_encoder_max_length, hidden_states],
        "text_embeds": [1, 1280],
        "time_ids": [1, 6],
    }
    unet_max_input_shape ={
        "sample": [bs, unet_in_channels, max_image_size // 8, max_image_size // 8],
        "timestep": [1],
        "encoder_hidden_states": [bs, unet_max_length, hidden_states],
        "text_embeds": [bs, 1280],
        "time_ids": [bs, 6],
    }
    unet_opt_input_shape ={
        "sample": [2, unet_in_channels, min_image_size // 8, min_image_size // 8],
        "timestep": [1],
        "encoder_hidden_states": [2, text_encoder_max_length, hidden_states],
        "text_embeds": [2, 1280],
        "time_ids": [2, 6],
    }
    unet_input_shape=[unet_min_input_shape, unet_max_input_shape, unet_opt_input_shape]
    # 4. Init runtime
    only_fp16_passes = [
                "trt_cross_multihead_matmul_fuse_pass",
                "trt_flash_multihead_matmul_fuse_pass",
                "preln_elementwise_groupnorm_act_pass",
                "elementwise_groupnorm_act_pass",
    
    ]
    no_need_passes = [
        'trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass',
        'add_support_int8_pass',
        'auto_mixed_precision_pass',
    ]
    paddle_delete_passes = dict(
        text_encoder=only_fp16_passes + no_need_passes if not args.use_fp16 else no_need_passes,
        text_encoder_2=only_fp16_passes + no_need_passes if not args.use_fp16 else no_need_passes,
        vae_encoder=only_fp16_passes + [] if args.use_fp16 else [],
        vae_decoder=only_fp16_passes + no_need_passes if not args.use_fp16 else no_need_passes,
        unet=only_fp16_passes + no_need_passes if not args.use_fp16 else no_need_passes,
    )
    infer_configs = dict(
            text_encoder=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            use_trt=False,
            model_name="text_encoder",
            dynamic_shape=None,
            precision_mode=paddle_infer.PrecisionType.Half,
            device_id=args.device_id,
            tune=False,
        ),
        text_encoder_2=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            use_trt=False,
            model_name="text_encoder_2",
            dynamic_shape=None,
            precision_mode=paddle_infer.PrecisionType.Half,
            device_id=args.device_id,
            tune=False,
        ),
        vae_encoder=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            model_name="vae_encoder",
            use_trt=False,
            precision_mode=paddle_infer.PrecisionType.Half,
            device_id=args.device_id,
            tune=False
        ),
        vae_decoder=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            model_name="vae_decoder",
            use_trt=False,
            precision_mode=paddle_infer.PrecisionType.Half,
            device_id=args.device_id,
            tune=False
        ),
        unet=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            model_name="unet",
            use_trt=True,
            precision_mode=paddle_infer.PrecisionType.Half,
            device_id=args.device_id,
            dynamic_shape=None,
            tune=False,
        ),
    )
    pipe = PaddleInferStableDiffusionXLPipeline.from_pretrained(
        args.model_dir,
        infer_configs=infer_configs,
        paddle_delete_passes=paddle_delete_passes,
    )
    pipe.set_progress_bar_config(disable=True)
    # pipe.change_scheduler(args.scheduler)
    width = args.width
    height = args.height
    folder = f"infer_fp16" if args.use_fp16 else f"infer_fp32"
    os.makedirs(folder, exist_ok=True)
    if args.task_name in ["text2img", "all"]:
        # text2img
        prompt = "beautiful scenery nature glass bottle landscape, purple galaxy bottle"
        time_costs = []
        negative_prompt = "text, watermark"
        # warmup
        pipe(
            prompt,
            num_inference_steps=20,
            height=height,
            width=width,
            # parse_prompt_type=parse_prompt_type,
            # infer_op_dict=infer_op_dict,
            negative_prompt=negative_prompt

        )
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
        images[0].save(f"{folder}/text2img_30step.png")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)

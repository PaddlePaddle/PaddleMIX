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

# isort: split
import paddle
import paddle.inference as paddle_infer

# isort: split
import cv2
import numpy as np
from paddlenlp.trainer.argparser import strtobool
from PIL import Image
from tqdm.auto import trange

from ppdiffusers import (  # noqa
    DiffusionPipeline,
    PaddleInferStableDiffusionMegaPipeline,
)
from ppdiffusers.utils import load_image


def get_canny_image(image, args):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = cv2.Canny(image, args.low_threshold, args.high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


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
            "inpaint_legacy",
            "hiresfix",
            "all",
        ],
        help="The task can be one of [text2img, img2img, inpaint_legacy, hiresfix, all]. ",
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
    parser.add_argument("--height", type=int, default=512, help="Height of input image")
    parser.add_argument("--width", type=int, default=512, help="Width of input image")
    parser.add_argument("--hr_resize_height", type=int, default=768, help="HR Height of input image")
    parser.add_argument("--hr_resize_width", type=int, default=768, help="HR Width of input image")
    parser.add_argument("--is_sd2_0", type=strtobool, default=False, help="Is sd2_0 model?")
    parser.add_argument(
        "--low_threshold",
        type=int,
        default=100,
        help="The value of Canny low threshold.",
    )
    parser.add_argument(
        "--high_threshold",
        type=int,
        default=200,
        help="The value of Canny high threshold.",
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
    if device_id != -1:
        config.use_gpu()
        config.enable_use_gpu(memory_pool_init_size_mb=2000, device_id=device_id, precision_mode=precision_mode)
    for pass_name in disable_paddle_pass:
        config.delete_pass(pass_name)
    if use_trt:
        # check if shape_range_info.pbtxt exists
        if not os.path.exists(shape_file):
            config.collect_shape_range_info(shape_file)

        config.enable_tensorrt_engine(
            workspace_size=workspace,
            precision_mode=precision_mode,
            max_batch_size=1,
            min_subgraph_size=3,
            use_static=True,
        )
        config.enable_tensorrt_memory_optim()
        config.enable_tuned_tensorrt_dynamic_shape(shape_file, True)
    return config


def main(args):
    if args.device_id == -1:
        paddle.set_device("cpu")
    else:
        paddle.set_device(f"gpu:{args.device_id}")

    infer_op_dict = {
        "vae_encoder": args.infer_op,
        "vae_decoder": args.infer_op,
        "text_encoder": args.infer_op,
        "unet": args.infer_op,
    }
    seed = 1024
    min_image_size = 512
    max_image_size = 768
    max_image_size = max(min_image_size, max_image_size)

    # 4. Init runtime
    only_fp16_passes = [
        "trt_cross_multihead_matmul_fuse_pass",
        "trt_flash_multihead_matmul_fuse_pass",
        "preln_elementwise_groupnorm_act_pass",
        "elementwise_groupnorm_act_pass",
    ]
    no_need_passes = [
        "trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass",
        "add_support_int8_pass",
        "auto_mixed_precision_pass",
        "conv_elementwise_add_fuse_pass",
    ]
    paddle_delete_passes = dict(
        text_encoder=only_fp16_passes + no_need_passes if not args.use_fp16 else no_need_passes,
        text_encoder_2=only_fp16_passes + no_need_passes if not args.use_fp16 else no_need_passes,
        vae_encoder=only_fp16_passes + [] if args.use_fp16 else [],
        vae_decoder=only_fp16_passes + no_need_passes if not args.use_fp16 else no_need_passes,
        unet=only_fp16_passes + no_need_passes if not args.use_fp16 else no_need_passes,
    )
    args.use_trt = args.backend == "paddle_tensorrt"
    infer_configs = dict(
        text_encoder=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            use_trt=False,
            model_name="text_encoder",
            precision_mode=paddle_infer.PrecisionType.Half,
            device_id=args.device_id,
            disable_paddle_trt_ops=["range", "lookup_table_v2"],
            disable_paddle_pass=paddle_delete_passes.get("text_encoder", []),
            tune=False,
        ),
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
            precision_mode=paddle_infer.PrecisionType.Half,
            device_id=args.device_id,
            disable_paddle_pass=paddle_delete_passes.get("vae_decoder", []),
            tune=False,
        ),
        unet=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            model_name="unet",
            use_trt=args.use_trt,
            precision_mode=paddle_infer.PrecisionType.Half,
            device_id=args.device_id,
            disable_paddle_pass=paddle_delete_passes.get("unet", []),
            tune=False,
        ),
    )
    pipe = PaddleInferStableDiffusionMegaPipeline.from_pretrained(
        args.model_dir,
        infer_configs=infer_configs,
        use_optim_cache=True,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.change_scheduler(args.scheduler)
    parse_prompt_type = args.parse_prompt_type
    width = args.width
    height = args.height
    hr_resize_width = args.hr_resize_width
    hr_resize_height = args.hr_resize_height

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
            init_image = load_image(
                "https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/control_bird_canny_demo.png"
            )
            controlnet_cond = get_canny_image(init_image, args)
            # text2img
            prompt = "bird"
            time_costs = []
            # warmup
            pipe.text2img(
                prompt,
                num_inference_steps=10,
                height=height,
                width=width,
                parse_prompt_type=parse_prompt_type,
                controlnet_cond=controlnet_cond,
                controlnet_conditioning_scale=1.0,
                infer_op_dict=infer_op_dict,
            )
            print("==> Test text2img performance.")
            for step in trange(args.benchmark_steps):
                start = time.time()
                paddle.seed(seed)
                images = pipe.text2img(
                    prompt,
                    num_inference_steps=args.inference_steps,
                    height=height,
                    width=width,
                    parse_prompt_type=parse_prompt_type,
                    controlnet_cond=controlnet_cond,
                    controlnet_conditioning_scale=1.0,
                    infer_op_dict=infer_op_dict,
                ).images
                latency = time.time() - start
                time_costs += [latency]
                # print(f"No {step:3d} time cost: {latency:2f} s")
            print(
                f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
                f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
            )
            images[0].save(f"{folder}/text2img.png")

        if args.task_name in ["img2img", "all"]:
            img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/sketch-mountains-input.png"
            init_image = load_image(img_url)
            controlnet_cond = get_canny_image(init_image, args)
            prompt = "A fantasy landscape, trending on artstation"
            time_costs = []
            # warmup
            pipe.img2img(
                prompt,
                image=init_image,
                num_inference_steps=20,
                height=height,
                width=width,
                parse_prompt_type=parse_prompt_type,
                controlnet_cond=controlnet_cond,
                controlnet_conditioning_scale=1.0,
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
                    controlnet_cond=controlnet_cond,
                    controlnet_conditioning_scale=1.0,
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

        if args.task_name in ["inpaint_legacy", "all"]:
            img_url = (
                "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
            )
            mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"
            init_image = load_image(img_url)
            mask_image = load_image(mask_url)
            controlnet_cond = get_canny_image(init_image, args)
            prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
            time_costs = []

            pipe.inpaint_legacy(
                prompt,
                image=init_image,
                mask_image=mask_image,
                num_inference_steps=20,
                height=height,
                width=width,
                parse_prompt_type=parse_prompt_type,
                controlnet_cond=controlnet_cond,
                controlnet_conditioning_scale=1.0,
                infer_op_dict=infer_op_dict,
            )
            print("==> Test inpaint_legacy performance.")
            for step in trange(args.benchmark_steps):
                start = time.time()
                paddle.seed(seed)
                images = pipe.inpaint_legacy(
                    prompt,
                    image=init_image,
                    mask_image=mask_image,
                    num_inference_steps=args.inference_steps,
                    height=height,
                    width=width,
                    parse_prompt_type=parse_prompt_type,
                    controlnet_cond=controlnet_cond,
                    controlnet_conditioning_scale=1.0,
                    infer_op_dict=infer_op_dict,
                ).images
                latency = time.time() - start
                time_costs += [latency]
                # print(f"No {step:3d} time cost: {latency:2f} s")
            print(
                f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
                f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
            )
            if args.task_name == "all":
                task_name = "inpaint_legacy"
            else:
                task_name = args.task_name
            images[0].save(f"{folder}/{task_name}.png")

        if args.task_name in ["hiresfix", "all"]:
            hiresfix_pipe = DiffusionPipeline.from_pretrained(
                args.model_dir,
                vae_encoder=pipe.vae_encoder,
                vae_decoder=pipe.vae_decoder,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=pipe.safety_checker,
                feature_extractor=pipe.feature_extractor,
                requires_safety_checker=pipe.requires_safety_checker,
                custom_pipeline="pipeline_PaddleInfer_stable_diffusion_hires_fix",
            )
            # custom_pipeline
            # https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/examples/community/pipeline_PaddleInfer_stable_diffusion_hires_fix.py
            hiresfix_pipe._progress_bar_config = pipe._progress_bar_config
            # hiresfix
            init_image = load_image(
                "https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/control_bird_canny_demo.png"
            )
            controlnet_cond = get_canny_image(init_image, args)
            # hiresfix
            prompt = "a red bird"
            time_costs = []
            # warmup
            hiresfix_pipe(
                prompt,
                height=height,
                width=width,
                num_inference_steps=20,
                hires_ratio=0.5,
                hr_resize_width=hr_resize_width,
                hr_resize_height=hr_resize_height,
                enable_hr=True,
                controlnet_cond=controlnet_cond,
                controlnet_conditioning_scale=1.0,
                parse_prompt_type=parse_prompt_type,
                infer_op_dict=infer_op_dict,
            )
            print("==> Test hiresfix performance.")
            for step in trange(args.benchmark_steps):
                start = time.time()
                paddle.seed(seed)
                images = hiresfix_pipe(
                    prompt,
                    height=height,
                    width=width,
                    num_inference_steps=args.inference_steps,
                    hires_ratio=0.5,
                    hr_resize_width=hr_resize_width,
                    hr_resize_height=hr_resize_height,
                    enable_hr=True,
                    controlnet_cond=controlnet_cond,
                    controlnet_conditioning_scale=1.0,
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
            images[0].save(f"{folder}/hiresfix.png")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

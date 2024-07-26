import argparse
import os
import time
import json
import paddle
# Note(wangbojun), open pir new executor
# paddle.set_flags({"FLAGS_enable_pir_in_executor": 1})
# isort: split
import numpy as np
import paddle.inference as paddle_infer
from paddlenlp.trainer.argparser import strtobool
from tqdm.auto import trange
import nvtx
from deploy.custom_ddim import DDIMScheduler
# from ppdiffusers.pipelines import (  # noqa
#     DiTInferencePipeline,
# )
from ppdiffusers.pipelines import (  # noqa
    DiTInferencePipeline,
)

def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

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
    enable_new_pir=True,
):
    config = paddle_infer.Config()
    # config.enable_memory_optim()
    shape_file = f"{model_dir}/{model_name}/shape_range_info.pbtxt"
    if tune:
        config.collect_shape_range_info(shape_file)
        config.switch_ir_optim(False)
    else:
        if enable_new_pir:
            config.enable_new_ir()
            config.switch_ir_optim(True)
        config.enable_new_executor()
        # config.switch_ir_debug(True)
    # config.enable_custom_passes([], true)
    if device_id != -1:
        config.use_gpu()
        config.enable_use_gpu(memory_pool_init_size_mb=2000, device_id=device_id, precision_mode=precision_mode)
    # for pass_name in disable_paddle_pass:
    config.delete_pass(disable_paddle_pass)
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
            ]
            for curr_pass in only_fp16_passes:
                config.delete_pass(curr_pass)
    # config.enable_custom_passes([], True)
    return config

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="./static_model",
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
    parser.add_argument("--use_fp16", type=strtobool, default=True, help="Wheter to use FP16 mode")
    parser.add_argument("--device_id", type=int, default=4, help="The selected gpu id. -1 means use cpu")
    
    parser.add_argument(
        "--tune",
        type=strtobool,
        default=False,
        help="Whether to tune the shape of tensorrt engine.",
    )

    return parser.parse_args()

def main(args):
    if args.device_id == -1:
        paddle.set_device("cpu")
    else:
        paddle.set_device(f"gpu:{args.device_id}")


    no_need_passes = [
        # "trt_prompt_tuning_embedding_eltwise_layernorm_fuse_pass",
        # "add_support_int8_pass",
        # "auto_mixed_precision_pass",
        # "conv2d_add_act_fuse_pass",
        "group_norm_silu_fuse_pass",
        "transfer_layout_pass",
        "add_norm_fuse_pass",
    ]
    paddle_delete_passes = dict(  # noqa
        vae_decoder= no_need_passes,
        transformer=no_need_passes
    )
    precision_mode = paddle_infer.PrecisionType.Half
    infer_configs = dict(
        vae_decoder=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            model_name="vae_decoder",
            precision_mode=paddle_infer.PrecisionType.Half,
            device_id=args.device_id,
            disable_paddle_pass=paddle_delete_passes.get("vae_decoder", []),
            tune=False,
            enable_new_pir=False,
        ),
        transformer=create_paddle_inference_runtime(
            model_dir=args.model_dir,
            model_name="transformer",
            precision_mode=precision_mode,
            device_id=args.device_id,
            disable_paddle_pass=paddle_delete_passes.get("unet", []),
            tune=args.tune,
        ),
    )

    scheduler = DDIMScheduler(
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        # Make sure the scheduler compatible with DDIM
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    pipe = DiTInferencePipeline.from_pretrained(
        pretrained_model_name_or_path=args.model_dir,
        infer_configs=infer_configs,
        scheduler=scheduler
    )

    
    transformer_config = read_json(args.model_dir + "/transformer/config.json")
    pipe.transformer_config=transformer_config
    
    vae_config = read_json(args.model_dir + "/vae_decoder/config.json")
    pipe.vae_config=vae_config
    
    id2label = read_json(args.model_dir + "/id2label.json")
    pipe.set_labels(id2label)
    words = ["golden retriever"]  # class_ids [207]
    class_ids = pipe.get_label_ids(words)
    pipe(class_labels=class_ids, num_inference_steps=25).images[0]
    total_time = 0.0
    test_time = 13
    warm_up_time = 3
    for i in range(test_time):
        paddle.device.cuda.synchronize()
        start_time = time.time()
        image=pipe(class_labels=class_ids, num_inference_steps=25).images[0]
        if i > warm_up_time:
            total_time += time.time() - start_time
    print(f"Average time: {total_time / (test_time-warm_up_time)}")

    image.save("golden_retriever.png")

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
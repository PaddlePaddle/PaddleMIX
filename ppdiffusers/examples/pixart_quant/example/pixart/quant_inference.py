import os
import sys
import time
import shutil
import argparse
import logging
import ppdiffusers

import paddle
from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging

from models.customize_pixart_alpha_pipeline import CustomizePixArtAlphaPipeline
from models.customize_transformer_2d import CustomizeTransformer2DModel

ppdiffusers.models.Transformer2DModel = CustomizeTransformer2DModel
ppdiffusers.PixArtAlphaPipeline = CustomizePixArtAlphaPipeline
from ppdiffusers import PixArtAlphaPipeline
from omegaconf import OmegaConf, ListConfig

def main(args):
    seed_everything(args.seed)
    paddle.set_grad_enabled(False)
    device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"

    if args.log is not None:
        if not os.path.exists(args.log):
            os.makedirs(args.log)
    log_file = os.path.join(args.log, 'run.log')
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    pipe = PixArtAlphaPipeline.from_pretrained("/mnt/public/wujunyi_tsinghua/huggingface_cache/hub/models--PixArt-alpha--PixArt-XL-2-1024-MS/snapshots/b89adadeccd9ead2adcb9fa2825d3fabec48d404", from_diffusers=True, from_hf_hub=True)

    # ---- assign quant configs ------
    quant_config = OmegaConf.load(args.quant_config)
    #pipe.convert_quant(quant_config)
    pipe = pipe.to(dtype=paddle.float16).to(device)
    #quant_param_ckpt = paddle.load(os.path.join(args.log, args.quant_param_ckpt))
    
    model = pipe.transformer
    #model.load_quant_param_dict(quant_param_ckpt)
    

    logger.info(str(model))

    # read the promts
    prompt_path = args.prompt if args.prompt is not None else "./prompts.txt"
    prompts = []
    with open(prompt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            prompts.append(line.strip())
                    
    N_batch = len(prompts) // args.batch_size # drop_last
    for i in range(N_batch):
        images = pipe(
            prompt=prompts[i*args.batch_size: (i+1)*args.batch_size],
            num_inference_steps=args.num_sampling_steps
        ).images
        print(f"Export image of batch {i}")

        save_path = os.path.join(args.log, "generated_images")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        for i_image in range(args.batch_size):
            images[i_image].save(os.path.join(save_path, f"output_{i_image + args.batch_size*i}.jpg"))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str)
    parser.add_argument('--quant-config', required=True, type=str)
    parser.add_argument("--quant_param_ckpt", type=str, default="./quant_params.pth")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hardware", action='store_true', help='whether to use_cuda_kernel')
    parser.add_argument("--profile", action='store_true', help='profile mode, measure the e2e latency')
    parser.add_argument("--quant_weight_ckpt", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    main(args)

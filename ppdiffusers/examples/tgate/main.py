import os
import argparse
import paddle

from tgate import TgateSDXLLoader, TgateSDXLDeepCacheLoader, TgatePixArtAlphaLoader, TgateSVDLoader
from ppdiffusers import StableDiffusionXLPipeline, PixArtAlphaPipeline, StableVideoDiffusionPipeline
from ppdiffusers import UNet2DConditionModel, LCMScheduler
from ppdiffusers import DPMSolverMultistepScheduler
from ppdiffusers.utils import load_image, export_to_video

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of TGATE V2.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="the input prompts",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="the dir of input image to generate video",
    )
    parser.add_argument(
        "--saved_path",
        type=str,
        default=None,
        required=True,
        help="Path to save the generated results.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='pixart',
        help="[pixart_alpha,sdxl,lcm_sdxl,lcm_pixart_alpha,svd]",
    )
    parser.add_argument(
        "--gate_step",
        type=int,
        default=10,
        help="When re-using the cross-attention",
    )
    parser.add_argument(
        '--sp_interval',
        type=int,
        default=5,
        help="The time-step interval to cache self attention before gate_step (Semantics-Planning Phase).",
    )
    parser.add_argument(
        '--fi_interval',
        type=int,
        default=1,
        help="The time-step interval to cache self attention after gate_step (Fidelity-Improving Phase).",
    )
    parser.add_argument(
        '--warm_up',
        type=int,
        default=2,
        help="The time step to warm up the model inference",
    )
    parser.add_argument(
        "--inference_step",
        type=int,
        default=25,
        help="total inference steps",
    )
    parser.add_argument(
        '--deepcache', 
        action='store_true', 
        default=False, 
        help='do deep cache',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for generation. Set for reproducible results.',
    )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.saved_path, exist_ok=True)
    if args.prompt:
        saved_path = os.path.join(args.saved_path, 'test.png')
    elif args.image:
        saved_path = os.path.join(args.saved_path, 'test.mp4')

    # Create generator if seed is provided
    generator = None
    if args.seed is not None:
        generator = paddle.Generator().manual_seed(args.seed)

    if args.model == 'sdxl':
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            paddle_dtype=paddle.float16, 
            variant="fp16", 
        )
        if args.deepcache:
            pipe = TgateSDXLDeepCacheLoader(
                pipe,
                cache_interval=3,
                cache_branch_id=0
            )
        else:
            pipe = TgateSDXLLoader(pipe)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        image = pipe.tgate(
            prompt=args.prompt,
            gate_step=args.gate_step,
            sp_interval=args.sp_interval if not args.deepcache else 1,
            fi_interval=args.fi_interval,
            warm_up=args.warm_up if not args.deepcache else 0,
            num_inference_steps=args.inference_step,
            generator=generator,
        ).images[0]
        image.save(saved_path)

    elif args.model == 'lcm_sdxl':
        unet = UNet2DConditionModel.from_pretrained(
            "latent-consistency/lcm-sdxl",
            paddle_dtype=paddle.float16,
            variant="fp16",
        )
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            unet=unet, 
            paddle_dtype=paddle.float16, 
            variant="fp16",
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe = TgateSDXLLoader(pipe)

        image = pipe.tgate(
            prompt=args.prompt, 
            gate_step=args.gate_step,
            sp_interval=1,
            fi_interval=args.fi_interval,
            warm_up=0, 
            num_inference_steps=args.inference_step,
            lcm=True,
            generator=generator,
        ).images[0]
        image.save(saved_path)

    elif args.model == 'pixart_alpha':
        pipe = PixArtAlphaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS", 
            paddle_dtype=paddle.float16,
        )
        pipe = TgatePixArtAlphaLoader(pipe)

        image = pipe.tgate(
            prompt=args.prompt,
            gate_step=args.gate_step,
            sp_interval=args.sp_interval,
            fi_interval=args.fi_interval,
            warm_up=args.warm_up,   
            num_inference_steps=args.inference_step,
            generator=generator,
        ).images[0]
        image.save(saved_path)

    elif args.model == 'lcm_pixart':
        pipe = PixArtAlphaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-LCM-XL-2-1024-MS", 
            paddle_dtype=paddle.float16,
        )
        pipe = TgatePixArtAlphaLoader(pipe)

        image = pipe.tgate(
            args.prompt,
            gate_step=args.gate_step,
            sp_interval=1,
            fi_interval=args.fi_interval,
            warm_up=0,
            num_inference_steps=args.inference_step,
            lcm=True,
            guidance_scale=0.,
            generator=generator,
        ).images[0]
        image.save(saved_path)

    elif args.model == 'svd':
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", 
            paddle_dtype=paddle.float16, 
            variant="fp16",
        )
        pipe = TgateSVDLoader(pipe)

        image = load_image(args.image)

        frames = pipe.tgate(
            image,
            gate_step=args.gate_step,
            num_inference_steps=args.inference_step,
            warm_up=args.warm_up,
            sp_interval=args.sp_interval,
            fi_interval=args.fi_interval,
            num_frames=25,
            decode_chunk_size=8,
            generator=generator,
        ).frames[0]
        export_to_video(frames, saved_path, fps=7)

    else:
        raise Exception('Please sepcify the model name!')
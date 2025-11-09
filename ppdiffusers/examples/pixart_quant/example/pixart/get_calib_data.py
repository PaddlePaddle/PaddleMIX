import os
import sys
import time
import shutil
import argparse
import logging
import ppdiffusers

import paddle
import paddle.nn as nn
from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging

from models.customize_pixart_alpha_pipeline import CustomizePixArtAlphaPipeline
from models.customize_transformer_2d import CustomizeTransformer2DModel

ppdiffusers.models.Transformer2DModel = CustomizeTransformer2DModel
ppdiffusers.PixArtAlphaPipeline = CustomizePixArtAlphaPipeline
from ppdiffusers import PixArtAlphaPipeline
from omegaconf import OmegaConf, ListConfig

class SaveActivationHook:
    def __init__(self):
        self.hook_handle = None
        self.outputs = []

    def __call__(self, layer, inputs):
        """
        Paddle 前向前钩子 (forward_pre_hook) 的签名通常是 (layer, inputs)
        inputs 通常是一个 tuple，inputs[0] 是实际的输入 Tensor。

        支持输入形状 [BS, C] 或 [BS, N_token, C]，
        仅保留通道维的最大值（以减小存储）。
        """
        # 取第一个输入（通常是我们需要的 Tensor）
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs

        # 保证是 Paddle Tensor
        if x is None:
            return

        # 取 channel 维度大小
        C = x.shape[-1]

        # 将前两维合并后取每列的绝对值最大值 -> shape [C]
        data = paddle.abs(x.reshape([-1, C])).max(axis=0)

        # 保存
        self.outputs.append(data)

    def clear(self):
        self.outputs = []
        
def add_hook_to_module_(module, hook_cls, **kwargs):
    """
    注册一个 paddle 钩子到 module 并返回 hook 实例。
    - module: paddle.nn.Layer
    - hook_cls: 钩子类（实例的 __call__ 应兼容 Paddle 的钩子签名）
    - when: 'pre' or 'post'（默认 'pre'）
    """
    hook = hook_cls()
    handle = module.register_forward_pre_hook(hook)
    hook.hook_handle = handle
    return hook

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
    pipe.convert_quant(quant_config)
    pipe = pipe.to(dtype=paddle.float16).to(device)
    model = pipe.transformer

    '''
    INFO: add the hook for hooking the activations
    '''
    kwargs = {
        'hook_cls': SaveActivationHook,
    }
    hook_d = apply_func_to_submodules(model,
                            class_type=nn.Linear,  # add hook to all objects of this cls
                            function=add_hook_to_module_,
                            return_d={},
                            **kwargs
                            )

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

    save_d = {}
    for k, v in hook_d.items():
        # 如果没有采集到任何 activation，跳过并记录警告
        if not getattr(v, "outputs", None):
            logger.warning(f'layer_name: {k} has no saved outputs, skipping.')
            continue

        # 将 list of paddle.Tensor ([C]) -> stacked Tensor shape [N_timestep*B, C]
        save_d[k] = paddle.stack(v.outputs, axis=0)

        # logging: v.outputs[0].shape 在 Paddle 中是 tuple，格式化打印也没问题
        logger.info(f'layer_name: {k}, hook_input_shape: {v.outputs[0].shape}')

        # 安全移除 hook（兼容不同 Paddle 版本）
        handle = getattr(v, "hook_handle", None)
        if handle is not None:
            try:
                # 新版本可能提供可调用的 handle.remove()
                handle.remove()
            except Exception:
                # 退而求其次：尝试从 module 的私有 hook dict 中移除（如果 hook 保存了 module 引用）
                try:
                    module = getattr(v, "module", None)
                    if module is not None:
                        # 可能是 pre 或 post 钩子 id
                        if hasattr(module, "_forward_pre_hooks"):
                            module._forward_pre_hooks.pop(handle, None)
                        if hasattr(module, "_forward_post_hooks"):
                            module._forward_post_hooks.pop(handle, None)
                    else:
                        # 无 module 引用时无法进一步移除（记录警告）
                        logger.warning(f"hook handle for {k} could not be removed automatically (no module reference).")
                except Exception as e:
                    logger.warning(f"failed to remove hook for {k}: {e}")
        else:
            logger.warning(f'no hook_handle found for {k}')

    # 保存到文件（Paddle 的保存格式）
    save_path = os.path.join(args.log, quant_config.calib_data.save_path)
    paddle.save(save_d, save_path)
    logger.info(f'saved calib data in {save_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument('--quant-config', required=True, type=str)
    parser.add_argument("--num-sampling-steps", type=int, default=10)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    main(args)

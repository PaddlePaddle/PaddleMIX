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

from qdiff.smooth_quant.sq_quant_layer import SQQuantizedLinear

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
    INFO: The PTQ process:
    for simple PTQ with dynamic act quant: 
    the weight are quantized with quant_model initialization.
    the act quant params are calculated online. 
    '''
    def init_sq_channel_mask_(module, full_name, calib_data, **kwargs):
        """
        module: SQQuantizedLinear（Paddle 版本）
        calib_data[full_name]: Tensor of shape [T, C] (Paddle Tensor)
        """
        assert isinstance(module, SQQuantizedLinear)
        # calib_data[full_name] 形状为 [T, C]，按第0维取 max -> [C]
        act_mask = paddle.max(calib_data[full_name], axis=0)
        module.get_channel_mask(act_mask)  # 设置 module.channel_mask
        module.update_quantized_weight_scaled()


    def init_rotation_matrix_(module, full_name):
        """
        module: QuarotQuantizedLinear（Paddle 版本）
        这里保持对外部工具函数的导入不变，假定在 Paddle 库中也有相应实现。
        """
        # 若这些类在你的环境中是 paddle 实现，断言有效；否则调整为类名检查或移除断言
        assert isinstance(module, QuarotQuantizedLinear)
        from qdiff.quarot.quarot_utils import random_hadamard_matrix, matmul_hadU_cuda
        module.get_rotation_matrix()
        module.update_quantized_weight_rotated()


    def init_rotation_and_channel_mask_(module, full_name, calib_data):
        """
        module: ViDiTQuantizedLinear（Paddle 版本）
        先基于 calib_data 计算 act mask，再计算 rotation，并更新量化权重。
        """
        assert isinstance(module, ViDiTQuantizedLinear)
        act_mask = paddle.max(calib_data[full_name], axis=0)
        module.get_channel_mask(act_mask)
        module.get_rotation_matrix()
        module.update_quantized_weight_rotated_and_scaled()

    '''
    INFO: the smooth_quant quantization.
    load act channel mask from the calib data
    '''
    if quant_config.get("smooth_quant",None) is not None:
        # INFO: the SQQuantizedLayer are initialized with the quant_layer_refactor_ in quant_dit.py
        from qdiff.smooth_quant.sq_quant_layer import SQQuantizedLinear

        assert quant_config.calib_data.save_path is not None
        calib_path = os.path.join(args.log, quant_config.calib_data.save_path)
        calib_data = paddle.load(calib_path)

        # get the channel mask, iter through all layers
        kwargs = {}
        apply_func_to_submodules(model,
                            class_type=SQQuantizedLinear,  # add hook to all objects of this cls
                            function=init_sq_channel_mask_,
                            calib_data = calib_data,
                            full_name='',
                            **kwargs
                            )

    '''
    INFO: the quarot quantization.
    init and apply the rotation matrix
    '''
    if quant_config.get("quarot",None) is not None:
        
        from qdiff.quarot.quarot_quant_layer import QuarotQuantizedLinear
        # get the rotation matrix, iter through all layers
        kwargs = {}
        apply_func_to_submodules(model,
                            class_type=QuarotQuantizedLinear,  # add hook to all objects of this cls
                            function=init_rotation_matrix_,
                            full_name='',
                            **kwargs
                            )
    '''
    INFO: combining both
    '''
    if quant_config.get("viditq",None) is not None:
        from qdiff.viditq.viditq_quant_layer import ViDiTQuantizedLinear
        
        assert quant_config.calib_data.save_path is not None
        calib_data = torch.load(os.path.join(args.log, quant_config.calib_data.save_path), weights_only=True)  # default wtih 
        kwargs = {}
        apply_func_to_submodules(model,
                            class_type=ViDiTQuantizedLinear,  # add hook to all objects of this cls
                            function=init_rotation_and_channel_mask_,
                            full_name='',
                            calib_data = calib_data,
                            **kwargs
                            )
        
    model.set_init_done()
    model.save_quant_param_dict()
    paddle.save(pipe.transformer.quant_param_dict, os.path.join(args.log, 'quant_params.pth'))
    logger.info(f'saved quant params into {args.log}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str)
    parser.add_argument('--quant-config', required=True, type=str)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=10)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    main(args)

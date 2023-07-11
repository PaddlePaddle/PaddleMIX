import paddle
""" OpenAI pretrained model functions

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import os
import warnings
from typing import List, Optional, Union
from .model import build_model_from_openai_state_dict, convert_weights_to_lp, get_cast_dtype
from .pretrained import get_pretrained_url, list_pretrained_models_by_tag, download_pretrained_from_url
__all__ = ['list_openai_models', 'load_openai_model']


def list_openai_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list_pretrained_models_by_tag('openai')


def load_openai_model(name: str,
                      precision: Optional[str]=None,
                      device: Optional[str]=None,
                      jit: bool=False,
                      cache_dir: Optional[str]=None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    precision: str
        Model precision, if None defaults to 'fp32' if device == 'cpu' else 'fp16'.
    device : str
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model (default) or more hackable non-JIT model.
    cache_dir : Optional[str]
        The directory to cache the downloaded model weights

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if device is None:
        device = 'cuda' if paddle.device.cuda.device_count() >= 1 else 'cpu'
    if precision is None:
        precision = 'fp32' if device == 'cpu' else 'fp16'
    if get_pretrained_url(name, 'openai'):
        model_path = download_pretrained_from_url(
            get_pretrained_url(name, 'openai'), cache_dir=cache_dir)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f'Model {name} not found; available models = {list_openai_models()}'
        )
    state_dict = paddle.load(model_path)
    if not jit:
        cast_dtype = get_cast_dtype(precision)
        try:
            model = build_model_from_openai_state_dict(
                state_dict or model.state_dict(), cast_dtype=cast_dtype)
        except KeyError:
            sd = {k[7:]: v for k, v in state_dict['state_dict'].items()}
            model = build_model_from_openai_state_dict(
                sd, cast_dtype=cast_dtype)
        if isinstance(device, paddle.dtype):
            dtype = device
        elif isinstance(device,
                        str) and device not in ['cpu', 'cuda', 'ipu', 'xpu']:
            dtype = device
        elif isinstance(device, paddle.Tensor):
            dtype = device.dtype
        else:
            dtype = model.dtype
        model = model.cast(dtype)
        if precision.startswith('amp') or precision == 'fp32':
            model.astype(dtype='float32')
        elif precision == 'bf16':
            convert_weights_to_lp(model, dtype='bfloat16')
        return model

    #paddle do not support jit mode

    # device_node = [n for n in device_holder.graph.findAllNodes(
    #     'prim::Constant') if 'Device' in repr(n)][-1]

    # def patch_device(module):
    #     try:
    #         graphs = [module.graph] if hasattr(module, 'graph') else []
    #     except RuntimeError:
    #         graphs = []
    #     if hasattr(module, 'forward1'):
    #         graphs.append(module.forward1.graph)
    #     for graph in graphs:
    #         for node in graph.findAllNodes('prim::Constant'):
    #             if 'value' in node.attributeNames() and str(node['value']
    #                 ).startswith('cuda'):
    #                 node.copyAttributes(device_node)
    # model.apply(fn=patch_device)
    # patch_device(model.encode_image)
    # patch_device(model.encode_text)
    # if precision == 'fp32':
    #     float_holder = torch.jit.trace(lambda : paddle.ones(shape=[]).
    #         astype(dtype='float32'), example_inputs=[])
    #     float_input = list(float_holder.graph.findNode('aten::to').inputs())[1]
    #     float_node = float_input.node()

    #     def patch_float(module):
    #         try:
    #             graphs = [module.graph] if hasattr(module, 'graph') else []
    #         except RuntimeError:
    #             graphs = []
    #         if hasattr(module, 'forward1'):
    #             graphs.append(module.forward1.graph)
    #         for graph in graphs:
    #             for node in graph.findAllNodes('aten::to'):
    #                 inputs = list(node.inputs())
    #                 for i in [1, 2]:
    #                     if inputs[i].node()['value'] == 5:
    #                         inputs[i].node().copyAttributes(float_node)
    #     model.apply(fn=patch_float)
    #     patch_float(model.encode_image)
    #     patch_float(model.encode_text)
    #     model.astype(dtype='float32')
    # model.visual.image_size = model.input_resolution.item()
    # return model

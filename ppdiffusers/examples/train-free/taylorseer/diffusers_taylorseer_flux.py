from typing import Any, Dict, Optional, Tuple, Union
import time
from ppdiffusers import DiffusionPipeline
from ppdiffusers.pipelines.flux import FluxPipeline
from ppdiffusers.models import FluxTransformer2DModel
from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.utils import USE_PEFT_BACKEND, is_paddle_version, logging, scale_lora_layers, unscale_lora_layers
import paddle
import numpy as np
from forwards import (taylorseer_flux_single_block_forward, 
                        taylorseer_flux_double_block_forward, 
                        taylorseer_flux_forward)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

num_inference_steps = 50
seed = 42
prompt = "An image of a squirrel in Picasso style"
#
pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
#pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# TaylorSeer settings
pipeline.transformer.__class__.num_steps = num_inference_steps

pipeline.transformer.__class__.forward = taylorseer_flux_forward

for double_transformer_block in pipeline.transformer.transformer_blocks:
    double_transformer_block.__class__.forward = taylorseer_flux_double_block_forward
    
for single_transformer_block in pipeline.transformer.single_transformer_blocks:
    single_transformer_block.__class__.forward = taylorseer_flux_single_block_forward



parameter_peak_memory = paddle.device.cuda.max_memory_allocated()

paddle.device.cuda.max_memory_reserved()
#start_time = time.time()
start = paddle.device.cuda.Event(enable_timing=True)
end = paddle.device.cuda.Event(enable_timing=True)
for i in range(2):
    start.record()
    img = pipeline(
        prompt, 
        num_inference_steps=num_inference_steps,
        generator=paddle.Generator("cpu").manual_seed(seed)
        ).images[0]

    end.record()
    paddle.device.synchronize()
    elapsed_time = start.elapsed_time(end) * 1e-3
    peak_memory = paddle.device.cuda.max_memory_allocated()

    img.save("{}.png".format('taylorseer_' + prompt))

    print(
        f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
    )
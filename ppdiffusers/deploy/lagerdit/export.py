import os
import time 
import json
import paddle
import argparse

from lagerdit_inference_model import LagerDitInferenceModel
from ppdiffusers import DDIMScheduler
from ppdiffusers.pipelines import DiTPipeline

paddle.set_device('gpu:7')
from paddlenlp.utils.import_utils import import_module

from types import MethodType

def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def dump_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f)

from ppdiffusers.pipelines.paddleinfer_xl_utils import (
    PaddleInferDiffusionXLPipelineMixin,
    PaddleInferRuntimeModel,
)
from ppdiffusers.pipelines.pipeline_utils import DiffusionPipeline
from ppdiffusers.schedulers import KarrasDiffusionSchedulers
from ppdiffusers.transformers import CLIPTokenizer
from ppdiffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class PaddleInferDitPipelineHousing(DiffusionPipeline, PaddleInferDiffusionXLPipelineMixin):
    _optional_components = [
        "vae_encoder",
    ]

    def __init__(
        self,
        transformer: PaddleInferRuntimeModel,
        vae_decoder: PaddleInferRuntimeModel,
        scheduler: KarrasDiffusionSchedulers,
        force_zeros_for_empty_prompt: bool = True,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae_decoder=vae_decoder,
            scheduler=scheduler,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.post_init()



def save_dit_model(
    model_path: str,
    output_path: str,
):
    
    pipe = DiTPipeline.from_pretrained(model_path, paddle_dtype="bfloat16")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    #init transformer 
    transformer_path = os.path.join("/root/.cache/paddlenlp/ppdiffusers/", model_path, "transformer")
    transformer_config_file = os.path.join(transformer_path, "config.json")
    paddle.set_default_dtype("bfloat16")
    inference_model = LagerDitInferenceModel(**read_json(transformer_config_file), export=True)
    dit_inference_state_dict = pipe.transformer.state_dict()
    inference_model.set_state_dict(dit_inference_state_dict)
    inference_model.eval()
    # 1. convert transformer
    input_spec = [
        paddle.static.InputSpec(shape=[2, 4, 32, 32], dtype="bfloat16", name="hidden_states"),
        paddle.static.InputSpec(shape=[2], dtype="int64", name="time_step"),
        paddle.static.InputSpec(shape=[2], dtype="int64", name="class_labels"),
    ]
    model = paddle.jit.to_static(inference_model, input_spec=input_spec)
    save_path = os.path.join(output_path, "transformer", "inference")
    paddle.jit.save(model, save_path)
    dump_json(pipe.transformer.config, output_path + "/transformer/config.json")
    

    #2. convert vae-decoder
    vae_decoder = pipe.vae
    def forward_vae_decoder(self, z):
            return self.decode(z, True).sample

    vae_decoder.forward = MethodType(forward_vae_decoder, vae_decoder)
    vae_decoder = paddle.jit.to_static(
        vae_decoder,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, 4, 32, 32],
                dtype="bfloat16",
                name="latent_sample",
            ),  # latent_sample
        ],
    )
    # Save vae_decoder in static graph model.
    save_path = os.path.join(output_path, "vae_decoder", "inference")
    paddle.jit.save(vae_decoder, save_path)
    dump_json(pipe.vae.config, output_path + "/vae_decoder/config.json")
    print(f"Save vae_decoder model in {save_path} successfully.")
    del pipe.vae
    paddle_infer_pipe_cls = PaddleInferDitPipelineHousing(
        transformer=PaddleInferRuntimeModel.from_pretrained(output_path +"/transformer"),
        vae_decoder=PaddleInferRuntimeModel.from_pretrained(output_path +"/vae_decoder"),
        scheduler=pipe.scheduler,
    )
    print("start saving")
    output_path = str(output_path)
    paddle_infer_pipe_cls.save_pretrained(output_path)
    dump_json(pipe.id2label, output_path+ "/id2label.json")
    print("PaddleInfer pipeline saved to", output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="Alpha-VLLM/Large-DiT-3B-256",
        type=str,
        help="Path to the `ppdiffusers` checkpoint to convert (either a local directory or on the bos).",
    )
    parser.add_argument("--output_path", default="./static_model" ,type=str, help="Path to the output model.")
    args = parser.parse_args()

    save_dit_model(
        args.pretrained_model_name_or_path,
        args.output_path,
    )
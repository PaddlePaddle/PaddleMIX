import paddle
from ppdiffusers import StableUnCLIPImg2ImgPipeline
from PIL import Image
import paddlemix.models.imagebind as ib

import paddle
import sys
import os
from paddlemix.datasets import *
from paddlemix import ImageBindModel, ImageBindProcessor
from paddlemix.models import *
import numpy as np
import argparse
import requests
from PIL import Image
from dataclasses import dataclass, field
from paddlenlp.trainer import PdArgumentParser

from paddlemix.utils.log import logger
from paddlemix.models.imagebind.modeling import ImageBindModel
from paddlemix.models.imagebind.utils import *
# from paddlemix.models.imagebind.utils.resample import *
# from paddlemix.models.imagebind.utils.paddle_aux import *

class Predictor:
    def __init__(self, model_args):
        self.processor = ImageBindProcessor.from_pretrained(model_args.model_name_or_path)
        self.predictor = ImageBindModel.from_pretrained(model_args.model_name_or_path)
        self.predictor.eval()
       
    def run(self, inputs):
        with paddle.no_grad():
            embeddings = self.predictor(inputs)

        return embeddings    
    
def main(model_args,data_args):

    #bulid model
    logger.info("imagebind_model: {}".format(model_args.model_name_or_path))

    url = (data_args.input_image)
    if os.path.isfile(url):
        #read image
        image_pil = Image.open(data_args.input_image).convert("RGB")
    elif url:
        image_pil = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    else:
        image_pil = None
        
    predictor = Predictor(model_args)
        
    encoding = predictor.processor(images=image_pil,text="", audios=data_args.input_audio, return_tensors='pd')
    inputs = {}

    if image_pil:
        image_processor = encoding["pixel_values"]
        inputs.update({ModalityType.VISION: image_processor})
    if data_args.input_audio:
        audio_processor = encoding["audio_values"]
        inputs.update({ModalityType.AUDIO:audio_processor})

    embeddings = predictor.run(inputs)
    image_proj_embeds = embeddings[ModalityType.AUDIO]
    

    if image_pil: 
        logger.info("Generate vision embedding: {}".format(embeddings[ModalityType.VISION]))
        image_proj_embeds +=  embeddings[ModalityType.VISION]

    if data_args.input_audio:
        logger.info("Generate audio embedding: {}".format(embeddings[ModalityType.AUDIO]))
        
    prompt = data_args.input_text
    
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        model_args.stable_unclip_model_name_or_path)
    pipe.set_progress_bar_config(disable=None)

    output = pipe(image_embeds=image_proj_embeds, prompt=prompt)    
    os.makedirs(model_args.output_dir, exist_ok=True)

    save_path = os.path.join(model_args.output_dir, "audio2img_imagebind_output.jpg")
    logger.info("Generate image to: {}".format(save_path))
    output.images[0].save(save_path)
    
@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    input_text: str = field(
        default = "",
        metadata={"help": "The name of prompt input."}
        
    )  
    input_image: str = field(
        default = "",
        #wget https://github.com/facebookresearch/ImageBind/blob/main/.assets/bird_image.jpg
        metadata={"help": "The name of image input."}
        
    )  
    input_audio: str = field(
        default = "",
        #wget https://github.com/facebookresearch/ImageBind/blob/main/.assets/bird_audio.wav
        metadata={"help": "The name of audio input."}
        
    )  

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="imagebind-1.2b/",
        metadata={"help": "Path to pretrained model or model identifier"},
    )
    
    stable_unclip_model_name_or_path: str = field(
        default="stabilityai/stable-diffusion-2-1-unclip",
        metadata={"help": "Path to pretrained model or model identifier in stable_unclip_model_name_or_path"},
    )
    
    output_dir: str = field(
        default = "vis_audio2img",
        metadata={"help": "The name of imagebind audio input."}
        
    )  

    device: str = field(
        default="GPU",
        metadata={
            "help": "Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU."
        },
    )


if __name__ == '__main__':
  
    parser = PdArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    model_args.device = model_args.device.upper()
    assert model_args.device in ['CPU', 'GPU', 'XPU', 'NPU'
                            ], "device should be CPU, GPU, XPU or NPU"


    main(model_args,data_args)


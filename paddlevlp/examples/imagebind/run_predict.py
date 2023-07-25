import paddle
import sys
import os
currentPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(currentPath + '/../../../')
from paddlevlp.datasets import *
from paddlevlp import ImageBindModel, ImageBindProcessor
from paddlevlp.models import ModalityType
import numpy as np
import argparse
import requests
from PIL import Image

from paddlevlp.utils.log import logger
from paddlevlp.models.imagebind.modeling import ImageBindModel

def predict(args):


    processor = ImageBindProcessor.from_pretrained(args.pretrained_name_or_path)
    
    #bulid model
    logger.info("imagebind_model: {}".format(args.pretrained_name_or_path))
    imagebind_model = ImageBindModel.from_pretrained(args.pretrained_name_or_path)
    imagebind_model.eval()


    url = (args.input_image)
    if os.path.isfile(url):
        #read image
        image_pil = Image.open(args.input_image).convert("RGB")
    else:
        image_pil = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        
    encoding = processor(images=image_pil,text=args.input_text, audios=args.input_audio, return_tensors='pd')
    if args.input_text:
        tokenized_processor = encoding['input_ids']
    if image_pil:
        image_processor = encoding["pixel_values"]
    if args.input_audio:
        audio_processor = encoding["audio_values"]

    
    inputs = {ModalityType.TEXT: tokenized_processor, ModalityType.VISION: image_processor, ModalityType.AUDIO:audio_processor}
    with paddle.no_grad():
        embeddings = imagebind_model(inputs)
    
    print(embeddings[ModalityType.TEXT])
    print(embeddings[ModalityType.VISION])
    print(embeddings[ModalityType.AUDIO])
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_name_or_path",
        default="./imagebind/",
        type=str,
        help="The dir name of imagebind checkpoint.",
    )
    parser.add_argument(
        "--input_text",
        default='A dog.',
        type=str,
        help="The dir name of imagebind text input.",
    )
    parser.add_argument(
        "--input_image",
        default='.assets/dog_image.jpg',
        type=str,
        help="The dir name of imagebind image input.",
    )
    parser.add_argument(
        "--input_audio",
        default='.assets/dog_audio.wav',
        type=str,
        help="The dir name of imagebind audio input.",
    )
    args = parser.parse_args()

    predict(args)

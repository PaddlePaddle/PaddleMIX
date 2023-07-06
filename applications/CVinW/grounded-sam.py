from dataclasses import dataclass, field
import os
import sys
import numpy as np
from typing import List

import paddle
import paddle.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from paddlenlp.trainer import PdArgumentParser
from paddlevlp.utils.log import logger

from paddlevlp.processors.groundingdino_processing import GroudingDinoProcessor
from paddlevlp.models.groundingdino.modeling import GroundingDinoModel
from paddlevlp.models.sam.modeling import SamModel
from paddlevlp.processors.sam_processing import SamProcessor


def postprocess(mask):
    masks = np.array(mask[:,0,:,:])
    init_mask = np.zeros(masks.shape[-2:])
    for mask in masks:
        mask = mask.reshape(mask.shape[-2:])
        mask[mask == False] = 0
        mask[mask == True] = 1
        init_mask += mask

    init_mask[init_mask == 0] = 0
    init_mask[init_mask != 0] = 255
    #init_mask = 255 - init_mask
    init_mask = Image.fromarray(init_mask).convert('L')

    return init_mask

def mask_image(image, mask):
    """Mask an image.
    """
    mask_data = np.array(mask, dtype="int32")
    if len(mask_data.shape) == 2: # mode L
        mask_data = np.expand_dims(mask_data, 2)
    masked = np.array(image, dtype="int32") - mask_data
    masked = masked.clip(0, 255).astype("uint8")
    masked = Image.fromarray(masked)
    return masked


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    input_image: str = field(
        metadata={"help": "The name of input image."}
    )  
    prompt: str = field(
        default=None, metadata={"help": "The prompt of the image to be generated."}
    )  


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    dino_model_name_or_path: str = field(
        default="GroundingDino/groundingdino-swint-ogc",
        metadata={"help": "Path to pretrained model or model identifier"},
    )
    sam_model_name_or_path: str = field(
        default="Sam/SamVitH",
        metadata={"help": "Path to pretrained model or model identifier"},
    )
    box_threshold: float = field(
        default=0.3,
        metadata={
            "help": "box threshold."
        },
    )
    text_threshold: float = field(
        default=0.25,
        metadata={
            "help": "text threshold."
        },
    )
    output_dir: str = field(
        default="grounded_sam_output",
        metadata={
            "help": "output directory."
        },
    )
    visual: bool = field(
        default=True,
        metadata={
            "help": "save visual image."
        },
    )

def main():
    parser = PdArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    #bulid dino processor
    dino_processor = GroudingDinoProcessor.from_pretrained(
        'bert-base-uncased'
    ) 

    #bulid dino model
    logger.info("dino_model: {}".format(model_args.dino_model_name_or_path))
    dino_model = GroundingDinoModel.from_pretrained(model_args.dino_model_name_or_path)
    
    #buidl sam processor
    sam_processor = SamProcessor.from_pretrained(
        'Sam'
    ) 
    #bulid model
    logger.info("SamModel: {}".format(model_args.sam_model_name_or_path))
    sam_model = SamModel.from_pretrained(model_args.sam_model_name_or_path,input_type="boxs")

    #read image
    image_pil = Image.open(data_args.input_image).convert("RGB")
    #preprocess image text_prompt
    image_tensor,mask,tokenized_out = dino_processor(images=image_pil,text=data_args.prompt)

    with paddle.no_grad():
        outputs = dino_model(image_tensor,mask, input_ids=tokenized_out['input_ids'],
                        attention_mask=tokenized_out['attention_mask'],text_self_attention_masks=tokenized_out['text_self_attention_masks'],
                        position_ids=tokenized_out['position_ids'])

    logits = F.sigmoid(outputs["pred_logits"])[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

     # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(axis=1) > model_args.box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

     # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = dino_processor.decode(logit > model_args.text_threshold)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

    
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    logger.info("dino output{}".format(pred_dict))
    
    H,W = size[1], size[0]
    boxes = []
    for box in zip(boxes_filt):
        box = box[0] * paddle.to_tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        x0, y0, x1, y1 = box.numpy()
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        boxes.append([x0, y0, x1, y1])
    boxes = np.array(boxes)
    image_seg,prompt = sam_processor(image_pil,input_type="boxs",box=boxes,point_coords=None) 
    seg_masks = sam_model(img=image_seg,prompt=prompt)
    seg_masks = sam_processor.postprocess_masks(seg_masks)
    
    if model_args.visual:
        # make dir
        os.makedirs(model_args.output_dir, exist_ok=True)
        init_mask = postprocess(seg_masks)
        
        image_masked = mask_image(image_pil, init_mask)
        image_masked.save(os.path.join(model_args.output_dir, "image_masked.jpg"))


if __name__ == "__main__":
    main()

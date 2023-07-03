import argparse
import os
import numpy as np
import paddle
import paddle.nn.functional as F

from paddlevlp.processors.groundingdino_processing import GroudingDinoProcessor
from paddlevlp.models.groundingdino.modeling import GroundingDinoModel
from PIL import Image, ImageDraw, ImageFont


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * paddle.to_tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box.numpy()
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def main():
    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--dino_type", "-dt", type=str, default="groundingdino-swint-ogc", help="dino type")
    parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument(
        "--visual",
        type=eval,
        default=True,
    )
    

    args = parser.parse_args()


    #bulid processor
    processor = GroudingDinoProcessor.from_pretrained(
        'bert-base-uncased'
    ) 
    #bulid model
    print(f'dino_model {args.dino_type}')
    dino_model = GroundingDinoModel.from_pretrained(args.dino_type)

    #read image
    image_pil = Image.open(args.image_path).convert("RGB")
    #preprocess image text_prompt
    image_tensor,mask,tokenized_out = processor(images=image_pil,text=args.text_prompt)

    with paddle.no_grad():
        outputs = dino_model(image_tensor,mask, input_ids=tokenized_out['input_ids'],
                        attention_mask=tokenized_out['attention_mask'],text_self_attention_masks=tokenized_out['text_self_attention_masks'],
                        position_ids=tokenized_out['position_ids'])

    logits = F.sigmoid(outputs["pred_logits"])[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

     # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(axis=1) > args.box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

     # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = processor.decode(logit > args.text_threshold)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

   
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    print("output:",pred_dict)

    if args.visual:
        # make dir
        os.makedirs(args.output_dir, exist_ok=True)
        image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
        image_with_box.save(os.path.join(args.output_dir, "pred.jpg"))


if __name__ == "__main__":
    main()
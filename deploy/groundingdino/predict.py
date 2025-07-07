# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass, field

import numpy as np
import paddle
import paddle.nn.functional as F
import requests
from paddle.inference import Config, create_predictor
from paddle.utils.cpp_extension import load
from paddlenlp.trainer import PdArgumentParser
from PIL import Image, ImageDraw, ImageFont

from paddlemix.processors.groundingdino_processing import GroundingDinoProcessor

ms_deformable_attn = load(
    name="deformable_detr_ops",
    sources=[
        "../../paddlemix/models/groundingdino/csrc/ms_deformable_attn_op.cc",
        "../../paddlemix/models/groundingdino/csrc/ms_deformable_attn_op.cu",
    ],
)


def load_predictor(
    model_dir,
    run_mode="paddle",
    batch_size=1,
    device="GPU",
    cpu_threads=1,
    enable_mkldnn=False,
    enable_mkldnn_bfloat16=False,
    delete_shuffle_pass=False,
):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16/trt_int8)
        use_dynamic_shape (bool): use dynamic shape or not
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        delete_shuffle_pass (bool): whether to remove shuffle_channel_detect_pass in TensorRT.
                                    Used by action model.
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need device == 'GPU'.
    """
    if device != "GPU" and run_mode != "paddle":
        raise ValueError(
            "Predict by TensorRT mode: {}, expect device=='GPU', but device == {}".format(run_mode, device)
        )
    infer_model = os.path.join(model_dir, "groundingdino_model.pdmodel")
    infer_params = os.path.join(model_dir, "groundingdino_model.pdiparams")

    config = Config(infer_model, infer_params)
    if device == "GPU":
        # initial GPU memory(M), device ID
        config.enable_use_gpu(200, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    elif device == "XPU":
        if config.lite_engine_enabled():
            config.enable_lite_engine()
        config.enable_xpu(10 * 1024 * 1024)
    elif device == "NPU":
        if config.lite_engine_enabled():
            config.enable_lite_engine()
        config.enable_npu()
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)
        if enable_mkldnn:
            try:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
                if enable_mkldnn_bfloat16:
                    config.enable_mkldnn_bfloat16()
            except Exception:
                print("The current environment does not support `mkldnn`, so disable mkldnn.")
                pass

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    if delete_shuffle_pass:
        config.delete_pass("shuffle_channel_detect_pass")
    predictor = create_predictor(config)
    return predictor, config


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
        box = box * paddle.to_tensor([W, H, W, H]).astype(paddle.float32)
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


class Predictor(object):
    def __init__(self, model_args, data_args):
        self.processor = GroundingDinoProcessor.from_pretrained(model_args.text_encoder_type)
        self.box_threshold = model_args.box_threshold
        self.text_threshold = model_args.text_threshold
        self.predictor, self.config = load_predictor(model_args.model_path)

        self.image = None
        self.mask = None
        self.tokenized_input = {}
        self.input_map = {}

    def create_inputs(self):

        self.input_map["x"] = self.image.numpy()
        self.input_map["m"] = np.array(self.mask.numpy(), dtype="int64")

        for key in self.tokenized_input.keys():
            self.input_map[key] = np.array(self.tokenized_input[key].numpy(), dtype="int64")

        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(self.input_map[input_names[i]])

    def preprocess(self, image, text):

        self.image, self.mask, self.tokenized_input = self.processor(images=image, text=text)

    def run(self, image, prompt):
        self.preprocess(image, data_args.prompt)

        self.create_inputs()
        self.predictor.run()
        output_names = self.predictor.get_output_names()
        pred_boxes = self.predictor.get_output_handle(output_names[0]).copy_to_cpu()
        pred_logits = self.predictor.get_output_handle(output_names[1]).copy_to_cpu()

        pred_dict = {
            "pred_logits": paddle.to_tensor(pred_logits),
            "pred_boxes": paddle.to_tensor(pred_boxes),
        }
        boxes_filt, pred_phrases = self.postprocess(pred_dict)
        return boxes_filt, pred_phrases

    def postprocess(self, outputs, with_logits=True):

        logits = F.sigmoid(outputs["pred_logits"])[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(axis=1) > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = self.processor.decode(logit > self.text_threshold)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases


def main(model_args, data_args):
    predictor = Predictor(model_args, data_args)
    url = data_args.input_image
    # read image
    if os.path.isfile(url):
        # read image
        image_pil = Image.open(data_args.input_image).convert("RGB")
    else:
        image_pil = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    if model_args.benchmark:
        import time

        start = 0.0
        total = 0.0
        for i in range(20):
            if i > 10:
                start = time.time()
            boxes_filt, pred_phrases = predictor.run(image_pil, data_args.prompt)
            if i > 10:
                total += time.time() - start

        print("Time:", total / 10)

    boxes_filt, pred_phrases = predictor.run(image_pil, data_args.prompt)

    # make dir
    os.makedirs(model_args.output_dir, exist_ok=True)

    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }

    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    image_with_box.save(os.path.join(model_args.output_dir, "pred.jpg"))


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    input_image: str = field(metadata={"help": "The name of input image."})
    prompt: str = field(default=None, metadata={"help": "The prompt of the image to be generated."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_path: str = field(
        default="output_groundingdino/",
        metadata={"help": "Path to pretrained model or model identifier"},
    )
    text_encoder_type: str = field(
        default="GroundingDino/groundingdino-swint-ogc",
        metadata={"help": "type for text encoder ."},
    )
    box_threshold: float = field(
        default=0.3,
        metadata={"help": "box threshold."},
    )
    text_threshold: float = field(
        default=0.25,
        metadata={"help": "text threshold."},
    )
    output_dir: str = field(
        default="output",
        metadata={"help": "output directory."},
    )
    run_mode: str = field(
        default="paddle",
        metadata={"help": "mode of running(paddle/trt_fp32/trt_fp16/trt_int8)."},
    )
    device: str = field(
        default="GPU",
        metadata={"help": "Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU."},
    )
    benchmark: bool = field(default=False, metadata={"help": "benchmark"})


if __name__ == "__main__":

    parser = PdArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    model_args.device = model_args.device.upper()
    assert model_args.device in [
        "CPU",
        "GPU",
        "XPU",
        "NPU",
    ], "device should be CPU, GPU, XPU or NPU"

    main(model_args, data_args)

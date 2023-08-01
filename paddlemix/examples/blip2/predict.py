# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import os
import numpy as np
import requests
from dataclasses import dataclass, field
from typing import List
import codecs
import yaml
import paddle
import paddle.nn.functional as F
from paddle.inference import Config as PredictConfig
from paddle.inference import create_predictor, PrecisionType

from PIL import Image, ImageDraw, ImageFont

from paddlemix.processors.blip_processing import Blip2Processor, BlipImageProcessor
from paddle.utils.cpp_extension import load
from paddlenlp.trainer import PdArgumentParser
from paddlemix.utils.log import logger

import matplotlib.pyplot as plt


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


class DeployConfig:
    def __init__(self, path):
        with codecs.open(path, 'r', 'utf-8') as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)

        self._dir = os.path.dirname(path)

    @property
    def model(self):
        return os.path.join(self._dir, self.dic['Deploy']['model'])

    @property
    def params(self):
        return os.path.join(self._dir, self.dic['Deploy']['params'])


def use_auto_tune(args):
    return hasattr(PredictConfig, "collect_shape_range_info") \
        and hasattr(PredictConfig, "enable_tuned_tensorrt_dynamic_shape") \
        and args.device == "gpu" and args.use_trt and args.enable_auto_tune


def auto_tune(args, imgs, img_nums):
    """
    Use images to auto tune the dynamic shape for trt sub graph.
    The tuned shape saved in args.auto_tuned_shape_file.

    Args:
        args(dict): input args.
        imgs(str, list[str], numpy): the path for images or the origin images.
        img_nums(int): the nums of images used for auto tune.
    Returns:
        None
    """
    logger.info("Auto tune the dynamic shape for GPU TRT.")

    assert use_auto_tune(args), "Do not support auto_tune, which requires " \
        "device==gpu && use_trt==True && paddle >= 2.2"

    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    num = min(len(imgs), img_nums)

    cfg = DeployConfig(args.cfg)
    pred_cfg = PredictConfig(cfg.model, cfg.params)
    pass_builder = pred_cfg.pass_builder()
    pass_builder.delete_pass("identity_op_clean_pass")
    pred_cfg.enable_use_gpu(100, 0)
    if not args.print_detail:
        pred_cfg.disable_glog_info()
    pred_cfg.collect_shape_range_info(args.auto_tuned_shape_file)

    # todo
    predictor = create_predictor(pred_cfg)
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    for i in range(0, num):
        if isinstance(imgs[i], str):
            data = {'img': imgs[i]}
            data = np.array([cfg.transforms(data)['img']])
        else:
            data = imgs[i]
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)
        try:
            predictor.run()
        except Exception as e:
            logger.info(str(e))
            logger.info(
                "Auto tune failed. Usually, the error is out of GPU memory "
                "for the model or image is too large. \n")
            del predictor
            if os.path.exists(args.auto_tuned_shape_file):
                os.remove(args.auto_tuned_shape_file)
            return

    logger.info("Auto tune success.\n")


class Predictor:
    def __init__(self, args):
        """
        Prepare for prediction.
        The usage and docs of paddle inference, please refer to
        https://paddleinference.paddlepaddle.org.cn/product_introduction/summary.html
        """
        self.args = args
        self.cfg = DeployConfig(args.cfg)
        self.processor = BlipImageProcessor.from_pretrained(
            args.model_name_or_path)

        self._init_base_config()

        if args.device == 'cpu':
            self._init_cpu_config()
        elif args.device == 'npu':
            self.pred_cfg.enable_custom_device('npu')
        elif args.device == 'xpu':
            self.pred_cfg.enable_xpu()
        else:
            self._init_gpu_config()

        try:
            self.predictor = create_predictor(self.pred_cfg)
        except Exception as e:
            logger.info(str(e))
            logger.info(
                "If the above error is '(InvalidArgument) some trt inputs dynamic shape info not set, "
                "..., Expected all_dynamic_shape_set == true, ...', "
                "please set --enable_auto_tune=True to use auto_tune. \n")
            exit()

    def _init_base_config(self):
        self.pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        pass_builder = self.pred_cfg.pass_builder()
        pass_builder.delete_pass("identity_op_clean_pass")
        self.pred_cfg.enable_memory_optim()
        self.pred_cfg.switch_ir_optim(True)

    def _init_cpu_config(self):
        """
        Init the config for x86 cpu.
        """
        logger.info("Use CPU")
        self.pred_cfg.disable_gpu()
        if self.args.enable_mkldnn:
            logger.info("Use MKLDNN")
            # cache 10 different shapes for mkldnn
            self.pred_cfg.set_mkldnn_cache_capacity(10)
            self.pred_cfg.enable_mkldnn()
        self.pred_cfg.set_cpu_math_library_num_threads(self.args.cpu_threads)

    def _init_gpu_config(self):
        """
        Init the config for nvidia gpu.
        """
        logger.info("Use GPU")
        self.pred_cfg.enable_use_gpu(100, 0)
        precision_map = {
            "fp16": PrecisionType.Half,
            "fp32": PrecisionType.Float32,
            "int8": PrecisionType.Int8
        }
        precision_mode = precision_map[self.args.precision]

    def run(self, image):
        image = self.preprocess(image)
        image['pixel_values'] = np.stack(image['pixel_values'], axis=0)
        input_names = self.predictor.get_input_names()
        input_handle1 = {
            input_names[0]: self.predictor.get_input_handle(input_names[0])
        }
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])

        input_handle1[input_names[0]].reshape(image['pixel_values'].shape)
        input_handle1[input_names[0]].copy_from_cpu(image['pixel_values'])
        self.predictor.run()
        results = output_handle.copy_to_cpu()
        return results

    def preprocess(self, image):

        image_seg = self.processor(image)

        return image_seg

    def postprocess(self, results):
        return self.processor.postprocess_masks(results)


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    input_image: str = field(
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        metadata={"help": "The name of input image."
                  })  # "http://images.cocodataset.org/val2017/000000039769.jpg"
    prompt: str = field(
        default="a photo of ",
        metadata={"help": "The prompt of the image to be generated."
                  })  # "Question: how many cats are there? Answer:"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="Salesforce/blip2-opt-2.7b",
        metadata={"help": "Path to pretrained model or model identifier"}, )
    cfg: str = field(
        default="blip2_export/deploy.yaml",
        metadata={"help": "The config file."}, )
    use_trt: bool = field(
        default=False,
        metadata={
            "help": "Whether to use Nvidia TensorRT to accelerate prediction."
        }, )
    precision: str = field(
        default="fp32",
        metadata={"help": "The tensorrt precision."}, )
    enable_auto_tune: bool = field(
        default=False,
        metadata={
            "help":
            "Whether to enable tuned dynamic shape. We uses some images to collect \
             the dynamic shape for trt sub graph, which avoids setting dynamic shape manually."
        }, )
    device: str = field(
        default="GPU",
        metadata={
            "help":
            "Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU."
        }, )
    cpu_threads: int = field(
        default=10,
        metadata={"help": "Number of threads to predict when using cpu."}, )
    enable_mkldnn: bool = field(
        default=False,
        metadata={"help": "Enable to use mkldnn to speed up when using cpu."}, )

    output_dir: str = field(
        default="seg_output",
        metadata={"help": "output directory."}, )


def main(model_args, data_args):

    url = (data_args.input_image)
    #read image
    if os.path.isfile(url):
        #read image
        image_pil = Image.open(data_args.input_image).convert("RGB")
    else:
        image_pil = Image.open(requests.get(url, stream=True).raw).convert(
            "RGB")

    predictor = Predictor(model_args)

    # image_pil = Image.open(data_args.input_image).convert("RGB")
    results = predictor.run(image_pil)
    logger.info("the image feature is:{}".format(results))


if __name__ == '__main__':

    parser = PdArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    model_args.device = model_args.device.upper()
    assert model_args.device in ['CPU', 'GPU', 'XPU', 'NPU'
                                 ], "device should be CPU, GPU, XPU or NPU"

    main(model_args, data_args)

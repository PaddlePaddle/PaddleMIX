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

import codecs
import os
from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import requests
import yaml
from paddle.inference import Config as PredictConfig
from paddle.inference import create_predictor
from paddlenlp.trainer import PdArgumentParser
from PIL import Image

from paddlemix.processors.sam_processing import SamProcessor
from paddlemix.utils.log import logger


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
        with codecs.open(path, "r", "utf-8") as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)

        self._dir = os.path.dirname(path)

    @property
    def model(self):
        return os.path.join(self._dir, self.dic["Deploy"]["model"])

    @property
    def params(self):
        return os.path.join(self._dir, self.dic["Deploy"]["params"])


def use_auto_tune(args):
    return (
        hasattr(PredictConfig, "collect_shape_range_info")
        and hasattr(PredictConfig, "enable_tuned_tensorrt_dynamic_shape")
        and args.device == "gpu"
        and args.use_trt
        and args.enable_auto_tune
    )


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

    assert use_auto_tune(args), (
        "Do not support auto_tune, which requires " "device==gpu && use_trt==True && paddle >= 2.2"
    )

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
            data = {"img": imgs[i]}
            data = np.array([cfg.transforms(data)["img"]])
        else:
            data = imgs[i]
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)
        try:
            predictor.run()
        except Exception as e:
            logger.info(str(e))
            logger.info(
                "Auto tune failed. Usually, the error is out of GPU memory " "for the model or image is too large. \n"
            )
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
        self.processor = SamProcessor.from_pretrained(args.model_name_or_path)

        self._init_base_config()

        if args.device == "cpu":
            self._init_cpu_config()
        elif args.device == "npu":
            self.pred_cfg.enable_custom_device("npu")
        elif args.device == "xpu":
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
                "please set --enable_auto_tune=True to use auto_tune. \n"
            )
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

    def run(self, image, prompt_out):
        image, prompt_out = self.preprocess(image, prompt_out)
        input_names = self.predictor.get_input_names()
        input_handle1 = self.predictor.get_input_handle(input_names[0])
        input_handle2 = self.predictor.get_input_handle(input_names[1])
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])

        input_handle1.reshape(image.shape)
        input_handle1.copy_from_cpu(image.numpy())
        if self.args.input_type == "boxs":
            prompt_out = prompt_out.reshape([-1, 4])
        input_handle2.reshape(prompt_out.shape)
        input_handle2.copy_from_cpu(prompt_out.numpy())

        self.predictor.run()

        results = output_handle.copy_to_cpu()

        results = self.postprocess(results)

        return results

    def preprocess(self, image, prompts):

        image_seg, prompt = self.processor(
            image,
            input_type=self.args.input_type,
            box=prompts["boxs"],
            point_coords=prompts["points"],
        )

        return [image_seg, prompt]

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

    input_image: str = field(metadata={"help": "The name of input image."})
    box_prompt: List[int] = field(default=None, metadata={"help": "box prompt format as xyxyxyxy...]."})
    points_prompt: List[int] = field(default=None, metadata={"help": "point prompt format as [[xy],[xy]...]."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="Sam/SamVitH-1024",
        metadata={"help": "Path to pretrained model or model identifier"},
    )
    input_type: str = field(
        default="boxs",
        metadata={"help": "The model prompt type, choices ['boxs', 'points', 'points_grid']."},
    )
    cfg: str = field(
        default=None,
        metadata={"help": "The config file."},
    )
    use_trt: bool = field(
        default=False,
        metadata={"help": "Whether to use Nvidia TensorRT to accelerate prediction."},
    )
    precision: str = field(
        default="fp32",
        metadata={"help": "The tensorrt precision."},
    )
    min_subgraph_size: int = field(
        default=3,
        metadata={"help": "The min subgraph size in tensorrt prediction.'"},
    )
    enable_auto_tune: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable tuned dynamic shape. We uses some images to collect \
             the dynamic shape for trt sub graph, which avoids setting dynamic shape manually."
        },
    )
    device: str = field(
        default="GPU",
        metadata={"help": "Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU."},
    )
    cpu_threads: int = field(
        default=10,
        metadata={"help": "Number of threads to predict when using cpu."},
    )
    enable_mkldnn: bool = field(
        default=False,
        metadata={"help": "Enable to use mkldnn to speed up when using cpu."},
    )

    output_dir: str = field(
        default="seg_output",
        metadata={"help": "output directory."},
    )
    visual: bool = field(
        default=True,
        metadata={"help": "save visual image."},
    )
    benchmark: bool = field(
        default=False,
        metadata={"help": "benchmark"}
    )


def main(model_args, data_args):

    url = data_args.input_image
    # read image
    if os.path.isfile(url):

        image_pil = Image.open(data_args.input_image).convert("RGB")
    else:
        image_pil = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    if data_args.box_prompt is not None:
        data_args.box_prompt = np.array(data_args.box_prompt)
    if data_args.points_prompt is not None:
        data_args.points_prompt = np.array([data_args.points_prompt])

    if use_auto_tune(model_args):
        tune_img_nums = 10
        auto_tune(model_args, [image_pil], tune_img_nums)

    predictor = Predictor(model_args)
    
    if model_args.benchmark:
        import time
        start = 0.0
        total = 0.0
        for i in range(20):
            if i>10:
                start = time.time()
            seg_masks = predictor.run(image_pil, {"points": data_args.points_prompt, "boxs": data_args.box_prompt})
            if i > 10:
                total += time.time()-start

        print("Time:",total/10)

    seg_masks = predictor.run(image_pil, {"points": data_args.points_prompt, "boxs": data_args.box_prompt})

    if model_args.visual:
        # make dir
        os.makedirs(model_args.output_dir, exist_ok=True)
        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image_pil)
        for mask in seg_masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)

        plt.axis("off")
        plt.savefig(
            os.path.join(model_args.output_dir, "mask_pred.jpg"),
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.0,
        )

    if use_auto_tune(model_args) and os.path.exists(model_args.auto_tuned_shape_file):
        os.remove(model_args.auto_tuned_shape_file)


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

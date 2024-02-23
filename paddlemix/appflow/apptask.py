# coding:utf-8
# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import math
import os
from multiprocessing import cpu_count

import paddle
from paddlenlp.taskflow.utils import dygraph_mode_guard

from paddlemix.utils.env import PPMIX_HOME
from paddlemix.utils.log import logger


class AppTask(object):
    """
    The meta class of task in Taskflow. The meta class has the five abstract function,
        the subclass need to inherit from the meta class.
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.
    """

    def __init__(self, model, task, priority_path=None, **kwargs):

        self.model = model
        self.task = task
        self.kwargs = kwargs
        self._priority_path = priority_path
        self.is_static_model = kwargs.get("is_static_model", False)

        self._home_path = self.kwargs["home_path"] if "home_path" in self.kwargs else PPMIX_HOME

        if "task_path" in self.kwargs:
            self._task_path = self.kwargs["task_path"]
            self._model_dir = self._task_path
        elif self._priority_path:
            self._task_path = os.path.join(self._home_path, "models", self._priority_path)
            self._model_dir = os.path.join(self._home_path, "models")
        else:
            self._task_path = os.path.join(self._home_path, "models", self.model)
            self._model_dir = os.path.join(self._home_path, "models")

        self._infer_precision = self.kwargs["precision"] if "precision" in self.kwargs else "fp32"
        # Default to use Paddle Inference
        self._predictor_type = "paddle-inference"
        self._num_threads = self.kwargs["num_threads"] if "num_threads" in self.kwargs else math.ceil(cpu_count() / 2)

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """

    def _construct_input_spec(self):
        """
        Construct the input spec for the predictor.
        """

    def _get_static_model_name(self):
        names = []
        for file_name in os.listdir(self._task_path):
            if ".pdparams" in file_name:
                names.append(file_name[:-9])
        if len(names) == 0:
            raise IOError(f"{self._task_path} should include '.pdparams' file.")
        if len(names) > 1:
            logger.warning(f"{self._task_path} includes more than one '.pdparams' file.")
        return names[0]

    def _convert_dygraph_to_static(self):
        """
        Convert the dygraph model to static model.
        """
        assert (
            self._model is not None
        ), "The dygraph model must be created before converting the dygraph model to static model."
        assert (
            self._input_spec is not None
        ), "The input spec must be created before converting the dygraph model to static model."
        logger.info("Converting to the inference model cost a little time.")
        static_model = paddle.jit.to_static(self._model, input_spec=self._input_spec)

        paddle.jit.save(static_model, self.inference_model_path)
        logger.info("The inference model save in the path:{}".format(self.inference_model_path))

    def _prepare_static_mode(self):
        """
        Construct the input data and predictor in the PaddlePaddle static mode.
        """
        if paddle.get_device() == "cpu":
            self._config.disable_gpu()
            self._config.enable_mkldnn()
            if self._infer_precision == "int8":
                # EnableMKLDNN() only works when IR optimization is enabled.
                self._config.switch_ir_optim(True)
                self._config.enable_mkldnn_int8()
                logger.info((">>> [InferBackend] INT8 inference on CPU ..."))
        elif paddle.get_device().split(":", 1)[0] == "npu":
            self._config.disable_gpu()
            self._config.enable_custom_device("npu", self.kwargs["device_id"])
        else:
            precision_map = {
                "trt_int8": paddle.inference.PrecisionType.Int8,
                "trt_fp32": paddle.inference.PrecisionType.Float32,
                "trt_fp16": paddle.inference.PrecisionType.Half,
            }
            self._config.enable_use_gpu(5000, self.kwargs["device_id"])
            self._config.set_cpu_math_library_num_threads(self._num_threads)
            self._config.switch_use_feed_fetch_ops(False)
            self._config.disable_glog_info()
            self._config.switch_ir_optim(True)
            self._config.enable_memory_optim(True)
            if self._infer_precision in precision_map.keys():
                self._config.enable_tensorrt_engine(
                    workspace_size=(1 << 40),
                    max_batch_size=0,
                    min_subgraph_size=30,
                    precision_mode=precision_map[self._infer_precision],
                    use_static=True,
                    use_calib_mode=False,
                )

                if not os.path.exists(self._tuned_trt_shape_file):
                    self._config.collect_shape_range_info(self._tuned_trt_shape_file)
                else:
                    logger.info(f"Use dynamic shape file: " f"{self._tuned_trt_shape_file} for TRT...")
                self._config.enable_tuned_tensorrt_dynamic_shape(self._tuned_trt_shape_file, True)

            if self.task == "openset_det_sam":
                self._config.delete_pass("add_support_int8_pass")

                if self.model == "GroundingDino/groundingdino-swint-ogc":
                    self._config.exp_disable_tensorrt_ops(
                        [
                            "pad3d",
                            "set_value",
                            "reduce_all",
                            "cumsum_8.tmp_0",
                            "linear_296.tmp_1",
                        ]
                    )

                if self.model == "Sam/SamVitH-1024" or self.model == "Sam/SamVitH-512":
                    self._config.delete_pass("shuffle_channel_detect_pass")
                    self._config.delete_pass("trt_skip_layernorm_fuse_pass")
                    self._config.delete_pass("preln_residual_bias_fuse_pass")
                    self._config.exp_disable_tensorrt_ops(
                        [
                            "concat_1.tmp_0",
                            "set_value",
                            "empty_0.tmp_0",
                            "concat_55.tmp_0",
                        ]
                    )

        self.predictor = paddle.inference.create_predictor(self._config)
        self.input_names = [name for name in self.predictor.get_input_names()]
        self.input_handles = [self.predictor.get_input_handle(name) for name in self.predictor.get_input_names()]
        self.output_handle = [self.predictor.get_output_handle(name) for name in self.predictor.get_output_names()]

    def _get_inference_model(self):
        """
        Return the inference program, inputs and outputs in static mode.
        """

        # When the user-provided model path is already a static model, skip to_static conversion
        if self.is_static_model:
            self.inference_model_path = os.path.join(self._task_path, self._static_model_name)
            if not os.path.exists(self.inference_model_path + ".pdmodel") or not os.path.exists(
                self.inference_model_path + ".pdiparams"
            ):
                raise IOError(
                    f"{self._task_path} should include {self._static_model_name + '.pdmodel'} and {self._static_model_name + '.pdiparams'} while is_static_model is True"
                )
            if self.paddle_quantize_model(self.inference_model_path):
                self._infer_precision = "int8"
                self._predictor_type = "paddle-inference"

        else:
            # Since 'self._task_path' is used to load the HF Hub path when 'from_hf_hub=True', we construct the static model path in a different way
            self.inference_model_path = os.path.join(self._task_path, self._static_model_name)
            self._tuned_trt_shape_file = self.inference_model_path + "_shape.txt"
            if not os.path.exists(self.inference_model_path + ".pdiparams"):
                with dygraph_mode_guard():
                    self._construct_model(self.model)
                    self._construct_input_spec()
                    self._convert_dygraph_to_static()

        self._static_model_file = self.inference_model_path + ".pdmodel"
        self._static_params_file = self.inference_model_path + ".pdiparams"

        if paddle.get_device().split(":", 1)[0] == "npu" and self._infer_precision == "fp16":
            # transform fp32 model tp fp16 model
            self._static_fp16_model_file = self.inference_model_path + "-fp16.pdmodel"
            self._static_fp16_params_file = self.inference_model_path + "-fp16.pdiparams"
            if not os.path.exists(self._static_fp16_model_file) and not os.path.exists(self._static_fp16_params_file):
                logger.info("Converting to the inference model from fp32 to fp16.")
                paddle.inference.convert_to_mixed_precision(
                    os.path.join(self._static_model_file),
                    os.path.join(self._static_params_file),
                    os.path.join(self._static_fp16_model_file),
                    os.path.join(self._static_fp16_params_file),
                    backend=paddle.inference.PlaceType.CUSTOM,
                    mixed_precision=paddle.inference.PrecisionType.Half,
                    # Here, npu sigmoid will lead to OOM and cpu sigmoid don't support fp16.
                    # So, we add sigmoid to black list temporarily.
                    black_list={"sigmoid"},
                )
                logger.info(
                    "The inference model in fp16 precision save in the path:{}".format(self._static_fp16_model_file)
                )
            self._static_model_file = self._static_fp16_model_file
            self._static_params_file = self._static_fp16_params_file

        if self._predictor_type == "paddle-inference":
            self._config = paddle.inference.Config(self._static_model_file, self._static_params_file)
            self._prepare_static_mode()
        else:
            self._prepare_onnx_mode()

    def __call__(self, *args, **kwargs):
        inputs = self._preprocess(*args)
        outputs = self._run_model(inputs, **kwargs)
        results = self._postprocess(outputs)
        return results

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import datetime
import os

import numpy as np
import requests

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["FLAGS_use_cuda_managed_memory"] = "true"

import paddle
from paddle import inference
from paddlenlp.transformers import MiniGPT4Processor
from PIL import Image
from utils import load_real_time_tokens


class Predictor(object):
    def __init__(self, args):
        self.args = args
        self.first_predictor, self.first_input_handles, self.first_output_handles = self.create_predictor(
            args.first_model_path
        )
        self.second_predictor, self.second_input_handles, self.second_output_handles = self.create_predictor(
            args.second_model_path
        )
        print(f"first_model_path: {args.first_model_path}, {self.first_predictor}")
        print(f"second_model_path: {args.second_model_path}, {self.second_predictor}")
        self.processor = MiniGPT4Processor.from_pretrained(args.minigpt4_path)

    def create_predictor(self, model_path):

        from paddlenlp.utils.import_utils import import_module

        import_module("paddlenlp_ops.encode_rotary_qk")
        import_module("paddlenlp_ops.get_padding_offset")
        import_module("paddlenlp_ops.qkv_transpose_split")
        import_module("paddlenlp_ops.rebuild_padding")
        import_module("paddlenlp_ops.transpose_remove_padding")
        import_module("paddlenlp_ops.write_cache_kv")

        model_file = model_path + ".pdmodel"
        params_file = model_path + ".pdiparams"
        if not os.path.exists(model_file):
            raise ValueError("not find model file path {}".format(model_file))
        if not os.path.exists(params_file):
            raise ValueError("not find params file path {}".format(params_file))
        config = paddle.inference.Config(model_file, params_file)

        shape_range_file = model_file + "shape.txt"
        # 第一次运行的时候需要收集shape信息，请打开下面的注释
        # config.collect_shape_range_info(shape_range_file)

        config.switch_ir_optim(True)
        # 第一个模型跑TRT
        if model_file.find("llama") == -1:
            self.args.use_tensorrt = False
        else:
            self.args.use_tensorrt = False

        if self.args.device == "gpu":
            # set GPU configs accordingly
            # such as initialize the gpu memory, enable tensorrt
            config.enable_use_gpu(100, 0)
            precision_mode = inference.PrecisionType.Half
            # 第一个模型是要跑TRT的
            if self.args.use_tensorrt:
                config.enable_tuned_tensorrt_dynamic_shape(shape_range_file, True)
                config.enable_tensorrt_engine(
                    max_batch_size=-1, min_subgraph_size=30, precision_mode=precision_mode, use_static=True
                )

        config.switch_use_feed_fetch_ops(False)
        predictor = paddle.inference.create_predictor(config)
        input_handles = [predictor.get_input_handle(name) for name in predictor.get_input_names()]
        output_handle = [predictor.get_output_handle(name) for name in predictor.get_output_names()]

        return predictor, input_handles, output_handle

    @paddle.no_grad()
    def encode_images(self, pixel_values):
        # pixel_values 已经在GPU上了
        [language_model_inputs, language_model_attention_mask] = self.first_predictor.run([pixel_values])
        return language_model_inputs, language_model_attention_mask

    @paddle.no_grad()
    def generate_with_image_features(
        self,
        image_features,
        first_input_ids,
        second_input_ids,
        image_attention_mask=None,
        first_attention_mask=None,
        second_attention_mask=None,
        **generate_kwargs,
    ):
        batch, seq, _ = image_features.shape
        seq = image_features.shape[1] + first_input_ids.shape[1] + second_input_ids.shape[1]
        max_len = 204
        dtype = "float16"
        tgt_generation_mask = paddle.full([batch, 1, 1, max_len], 0, dtype=dtype)
        tgt_generation_mask[:, 0, 0, :seq] = 1

        attention_mask = paddle.full([batch, 1, max_len, max_len], 0, dtype=dtype)
        attention_mask[:, 0, :seq, :seq] = paddle.tril(paddle.ones(shape=(seq, seq), dtype=dtype))
        position_ids = paddle.full([batch, seq], 0, dtype="int64")
        for i in range(batch):
            position_ids[i, :] = paddle.to_tensor([i for i in range(seq)], dtype="int64")

        inputs = [
            image_features,
            first_input_ids,
            second_input_ids,
            attention_mask,
            # image_attention_mask,
            # first_attention_mask,
            # second_attention_mask,
            position_ids,  # position_ids
            paddle.full([batch, 1], 1.0, dtype="float32"),  # penalty_score
            paddle.full([batch, 1], 0.0, dtype="float32"),  # frequency_score,
            paddle.full([batch, 1], 0.0, dtype="float32"),  # presence_score,
            paddle.full([batch, 1], 1, dtype="int64"),  # min_length,
            paddle.full([batch, 1], max_len - seq, dtype="int64"),  # max_length,
            paddle.full([batch, 1], 1.0, dtype="float32"),  # temperature,
            paddle.full([batch, 1], 0.0, dtype="float32"),  # top_p,
            paddle.full([1], 2277, dtype="int64"),  # eos_token_id,
            paddle.full([batch, 1], seq, dtype="int32"),  # seq_len_encoder,
            paddle.full([batch, 1], seq, dtype="int32"),  # seq_len_decoder,
            paddle.full([batch, 1], 0, dtype="int64"),  # step_idx,
            paddle.full([batch, 1], False, dtype="bool"),  # stop_flags,
            paddle.full([batch, 1], -123, dtype="int64"),  # tgt_ids can be be initialized arbitrarily
            paddle.full([batch, 1], seq - 1, dtype="int64"),  # tgt_pos,
            tgt_generation_mask,  # tgt_generation_mask,
            paddle.full([batch, max_len], -100, dtype="int64"),  # pre_ids, can be initialized arbitrarily
            paddle.full([1], batch, dtype="int64"),  # stop_nums, be batch
        ]
        for i in range(40):
            tmp = paddle.rand(shape=[2, batch, 40, max_len, 128], dtype=dtype)
            print(tmp.shape)
            inputs.append(tmp)

        self.second_predictor.run(inputs)
        tokens: np.ndarray = load_real_time_tokens()
        generate_ids = tokens.tolist()
        return generate_ids, None

    def pre_processing(self, images, text, prompt=None):
        processed_contents = self.processor(images, text, prompt=prompt)
        return processed_contents

    def post_processing(self, generate_ids):
        msg = self.processor.batch_decode(generate_ids)
        return msg

    def predict(self, images, text, prompt=None):
        processed_contents = self.pre_processing(images, text, prompt=prompt)
        batch = 1
        processed_contents["pixel_values"] = paddle.tile(
            processed_contents["pixel_values"], repeat_times=[batch, 1, 1, 1]
        )
        image_features, image_attention_mask = self.encode_images(processed_contents["pixel_values"])
        print(image_attention_mask.shape)
        processed_contents["first_input_ids"] = paddle.tile(
            processed_contents["first_input_ids"], repeat_times=[batch, 1]
        )
        processed_contents["second_input_ids"] = paddle.tile(
            processed_contents["second_input_ids"], repeat_times=[batch, 1]
        )
        processed_contents["first_attention_mask"] = paddle.tile(
            processed_contents["first_attention_mask"], repeat_times=[batch, 1]
        )
        processed_contents["second_attention_mask"] = paddle.tile(
            processed_contents["second_attention_mask"], repeat_times=[batch, 1]
        )
        generate_ids, _ = self.generate_with_image_features(
            image_features,
            processed_contents["first_input_ids"],
            processed_contents["second_input_ids"],
            image_attention_mask,
            processed_contents["first_attention_mask"],
            processed_contents["second_attention_mask"],
        )

        msg = self.post_processing(generate_ids)

        return msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--first_model_path",
        default="The dir name of image encoder model",
        type=str,
        help="",
    )
    parser.add_argument(
        "--second_model_path",
        default="The dir name of language model",
        type=str,
        help="",
    )
    parser.add_argument(
        "--minigpt4_path",
        type=str,
        default="The minigpt4 dir name of saving tokenizer",
        help="The path of extraction model path that you want to load.",
    )
    parser.add_argument("--use_tensorrt", action="store_true", help="Whether to use inference engin TensorRT.")
    parser.add_argument(
        "--precision", default="fp32", type=str, choices=["fp32", "fp16", "int8"], help="The tensorrt precision."
    )
    parser.add_argument(
        "--device", default="gpu", choices=["gpu", "cpu", "xpu"], help="Device selected for inference."
    )
    parser.add_argument("--cpu_threads", default=10, type=int, help="Number of threads to predict when using cpu.")
    parser.add_argument(
        "--enable_mkldnn",
        default=False,
        type=eval,
        choices=[True, False],
        help="Enable to use mkldnn to speed up when using cpu.",
    )
    args = parser.parse_args()

    predictor = Predictor(args)

    url = "https://paddlenlp.bj.bcebos.com/data/images/mugs.png"
    image = Image.open(requests.get(url, stream=True).raw)

    text = "describe this image"
    prompt = "Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.###Human: <Img><ImageHere></Img> <TextHere>###Assistant:"

    # warm up
    warm_up_times = 2
    repeat_times = 10
    for i in range(warm_up_times):
        msg = predictor.predict(image, text, prompt)

    # 测试50次
    starttime = datetime.datetime.now()
    for i in range(repeat_times):
        msg = predictor.predict(image, text, prompt)

    endtime = datetime.datetime.now()
    duringtime = endtime - starttime
    time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0

    print(
        "Reference: The image shows two black and white cats sitting next to each other on a blue background. The cats have black fur and white fur with black noses, eyes, and paws. They are both looking at the camera with a curious expression. The mugs are also blue with the same design of the cats on them. There is a small white flower on the left side of the mug. The background is a light blue color."
    )
    print("Outputs: ", msg)
    print("The whole time on average: ", time_ms / repeat_times, "ms")

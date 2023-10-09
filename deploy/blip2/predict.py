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
import argparse
import os

import numpy as np
import paddle
import requests
from PIL import Image

from paddlemix.models.blip2.utils import create_tokenizer, load_real_time_tokens
from paddlemix.processors.blip_processing import (
    Blip2Processor,
    BlipImageProcessor,
    BlipTextProcessor,
)
from paddlemix.utils.downloader import is_url


class Predictor(object):
    def __init__(self, args):
        self.args = args
        self.first_predictor = self.create_predictor(args.first_model_path)
        self.second_predictor = self.create_predictor(args.second_model_path)
        tokenizer_class = create_tokenizer(args.text_model_name_or_path)
        image_processor_eval = BlipImageProcessor.from_pretrained(
            os.path.join(args.model_name_or_path, "processor", "eval")
        )
        text_processor_class_eval = BlipTextProcessor.from_pretrained(
            os.path.join(args.model_name_or_path, "processor", "eval")
        )
        eval_processor = Blip2Processor(image_processor_eval, text_processor_class_eval, tokenizer_class)
        self.processor = eval_processor

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

        config.switch_ir_optim(False)

        if self.args.device == "gpu":
            config.enable_use_gpu(100, 0)

        config.switch_use_feed_fetch_ops(False)
        predictor = paddle.inference.create_predictor(config)
        return predictor

    @paddle.no_grad()
    def encode_images(self, pixel_values):
        [language_model_inputs] = self.first_predictor.run([pixel_values])
        return language_model_inputs

    @paddle.no_grad()
    def generate_with_image_features(self, image_features, second_input_ids):

        batch, seq, _ = image_features.shape
        seq = image_features.shape[1] + second_input_ids.shape[1]
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
            second_input_ids,
            attention_mask,
            position_ids,  # position_ids
            paddle.full([batch, 1], 1.0, dtype="float32"),  # penalty_score
            paddle.full([batch, 1], 0.0, dtype="float32"),  # frequency_score,
            paddle.full([batch, 1], 0.0, dtype="float32"),  # presence_score,
            paddle.full([batch, 1], 1, dtype="int64"),  # min_length,
            paddle.full([batch, 1], max_len - seq, dtype="int64"),  # max_length,
            paddle.full([batch, 1], 1.0, dtype="float32"),  # temperature,
            paddle.full([batch, 1], 0.0, dtype="float32"),  # top_p,
            paddle.full([1], 50118, dtype="int64"),  # eos_token_id,
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
        for i in range(32):
            tmp = paddle.rand(shape=[2, batch, 32, max_len, 80], dtype=dtype)
            print(tmp.shape)
            inputs.append(tmp)

        self.second_predictor.run(inputs)

        import datetime

        starttime = datetime.datetime.now()

        tokens: np.ndarray = load_real_time_tokens()
        generate_ids = tokens.tolist()

        print(generate_ids[0])

        endtime = datetime.datetime.now()
        duringtime = endtime - starttime
        ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
        print("读取磁盘时间，", ms)
        return generate_ids, None

    def pre_processing(self, images, text, prompt=None):
        processed_contents = self.processor(images, text, prompt=prompt)
        return processed_contents

    def post_processing(self, generate_ids):
        msg = self.processor.batch_decode(generate_ids)
        return msg

    def predict(self, images, prompt=None):
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pd",
            return_attention_mask=True,
            mode="test",
        )
        pixel_values = inputs["pixel_values"]
        second_input_ids = inputs["input_ids"]

        image_features = self.encode_images(pixel_values)
        generate_ids, _ = self.generate_with_image_features(
            image_features,
            second_input_ids,
        )

        msg = self.post_processing(generate_ids)

        return msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--first_model_path",
        default="blip2_export/image_encoder",
        type=str,
        help="",
    )
    parser.add_argument(
        "--second_model_path",
        default="opt-2.7b-infer_static/opt",
        type=str,
        help="",
    )
    parser.add_argument(
        "--image_path",
        default="https://paddlenlp.bj.bcebos.com/data/images/mugs.png",
        type=str,
        help="",
    )
    parser.add_argument("--prompt", default="a photo of ", type=str, help="The input prompt.")
    parser.add_argument(
        "--text_model_name_or_path",
        default="facebook/opt-2.7b",
        type=str,
        help="The type of text model to use (OPT, T5).",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="paddlemix/blip2-caption-opt2.7b",
        type=str,
        help="Path to pretrained model or model identifier.",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
    )
    args = parser.parse_args()

    predictor = Predictor(args)

    image_path = args.image_path
    # url = "https://paddlenlp.bj.bcebos.com/data/images/female.png"
    if is_url(image_path):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)

    prompt = args.prompt
    # warp up
    warm_up_times = 1
    repeat_times = 5
    for i in range(warm_up_times):
        msg = predictor.predict(image, prompt)

    # 测试50次
    import datetime

    starttime = datetime.datetime.now()

    for i in range(repeat_times):
        msg = predictor.predict(image, prompt)

    endtime = datetime.datetime.now()
    duringtime = endtime - starttime
    time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0

    print("Reference: two coffee mugs with cats on them.")
    print("Outputs: ", msg)
    print("infer OK")
    print("The whoel end to end time : ", time_ms / repeat_times, "ms")

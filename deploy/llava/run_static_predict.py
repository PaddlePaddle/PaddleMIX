# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from utils import load_real_time_tokens

from paddlemix.auto import AutoConfigMIX, AutoProcessorMIX, AutoTokenizerMIX
from paddlemix.models.llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from paddlemix.models.llava.conversation import conv_templates
from paddlemix.utils.log import logger


class Predictor(object):
    def __init__(self, args):

        self.compute_dtype = "float16" if args.fp16 else "bfloat16"
        if not paddle.amp.is_bfloat16_supported() and self.compute_dtype == "bfloat16":
            logger.warning("bfloat16 is not supported on your device,change to float32")
            self.compute_dtype = "float32"

        self.args = args
        self.config = AutoConfigMIX.from_pretrained(args.model_name_or_path)

        self.tokenizer = AutoTokenizerMIX.from_pretrained(args.model_name_or_path, use_fast=False)
        self.processor, _ = AutoProcessorMIX.from_pretrained(args.model_name_or_path, eval="eval")

        self.first_predictor = self.create_predictor(args.first_model_path)
        print(f"first_model_path: {args.first_model_path}, {self.first_predictor}")
        self.second_predictor = self.create_predictor(args.second_model_path)
        print(f"second_model_path: {args.second_model_path}, {self.second_predictor}")

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

        config.switch_ir_optim(True)

        if self.args.device == "gpu":
            config.enable_use_gpu(100, 0)

        config.switch_use_feed_fetch_ops(False)
        predictor = paddle.inference.create_predictor(config)
        return predictor

    @paddle.no_grad()
    def encode_images(self, pixel_values):
        language_model_inputs = self.first_predictor.run(pixel_values)
        return language_model_inputs

    @paddle.no_grad()
    def generate_with_image_features(self, image_features, input_ids):
        max_len = 2048
        total_max_length = max_len + 1024
        batch, seq, _ = image_features.shape
        seq += input_ids.shape[1] - 1

        _attention_mask = paddle.ones_like(x=input_ids, dtype="bool")
        input_ids = [
            cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, _attention_mask)
        ]
        cur_image_idx = 0
        new_input_ids = []
        img_pos = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            image_token_indices = (
                [-1]
                + paddle.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].squeeze(axis=1).tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []

            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])

            split_sizes = [x.shape[0] for x in cur_input_ids_noim]

            split_start = 0
            cur_new_input_ids = []
            cur_img_pos = []

            for i in range(num_images + 1):
                cur_new_input_ids.append(cur_input_ids_noim[i])

                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_ids.append(paddle.full([cur_image_features.shape[0]], 1, dtype="int64"))
                    split_start += split_sizes[i - 1] if i > 0 else split_sizes[i]
                    cur_img_pos.append([split_start, split_start + cur_image_features.shape[0]])
                    split_start += cur_image_features.shape[0]

            cur_new_input_ids = paddle.concat(x=cur_new_input_ids)
            new_input_ids.append(cur_new_input_ids)
            img_pos.append(cur_img_pos)

        new_input_ids = paddle.to_tensor(new_input_ids)
        img_pos = paddle.to_tensor(img_pos)

        tgt_generation_mask = paddle.full([batch, 1, 1, total_max_length], 1)

        attention_mask = paddle.zeros(
            shape=(batch, 1, total_max_length, total_max_length),
            dtype="int64",
        )
        length = seq
        attention_mask[:, 0, :length, :length] = paddle.tril(paddle.ones(shape=(length, length), dtype="int64"))

        position_ids = paddle.full([batch, total_max_length], 0, dtype="int64")
        position_ids[:, :seq] = paddle.arange(0, seq)

        inputs = [
            new_input_ids,  # input_ids
            image_features,  # image_features
            img_pos,
            attention_mask,
            position_ids,
            paddle.full([batch, 1], 1.0, dtype="float32"),  # penalty_score
            paddle.full([batch, 1], 0.0, dtype="float32"),  # frequency_score,
            paddle.full([batch, 1], 0.0, dtype="float32"),  # presence_score,
            paddle.full([batch, 1], 1, dtype="int64"),  # min_length,
            paddle.full([batch, 1], max_len, dtype="int64"),  # max_length,
            paddle.full([batch, 1], 0.7, dtype="float32"),  # temperature,
            paddle.full([batch, 1], 0.95, dtype="float32"),  # top_p,
            paddle.full([1], self.config.eos_token_id, dtype="int64"),  # eos_token_id,
            paddle.full([batch, 1], seq, dtype="int32"),  # seq_len_encoder,
            paddle.full([batch, 1], seq, dtype="int32"),  # seq_len_decoder,
            paddle.full([batch, 1], 0, dtype="int64"),  # step_idx,
            paddle.full([batch, 1], False, dtype="bool"),  # stop_flags,
            paddle.full([batch, 1], 29962, dtype="int64"),  # tgt_ids can be be initialized arbitrarily
            paddle.full([batch, 1], seq - 1, dtype="int64"),  # tgt_pos,
            tgt_generation_mask,  # tgt_generation_mask,
            paddle.full([batch, total_max_length], -1, dtype="int64"),  # pre_ids, can be initialized arbitrarily
            paddle.full([1], batch, dtype="int64"),  # stop_nums, be batch
        ]

        for i in range(self.config.num_hidden_layers):
            tmp = paddle.zeros(
                shape=[
                    2,
                    batch,
                    self.config.num_attention_heads,
                    total_max_length,
                    self.config.hidden_size // self.config.num_attention_heads,
                ],
                dtype=self.compute_dtype,
            )

            inputs.append(tmp)

        self.second_predictor.run(inputs)
        tokens = load_real_time_tokens()
        generate_ids = tokens.tolist()

        return generate_ids, None

    def pre_processing(self, inp, first_message):
        model_name = self.args.model_name_or_path
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        if self.args.conv_mode is not None and conv_mode != self.args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, self.args.conv_mode, self.args.conv_mode
                )
            )
        else:
            self.args.conv_mode = conv_mode
        conv = conv_templates[self.args.conv_mode].copy()

        if self.args.image_file is not None and first_message:
            if self.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
            conv.append_message(conv.roles[0], inp)
            first_message = False
        else:
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        record = {"image": self.args.image_file, "conversations": prompt}

        data_dict = self.processor(record=record, image_aspect_ratio=self.config.image_aspect_ratio)

        return data_dict

    def post_processing(self, generate_ids):
        msg = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return msg

    def predict(self):
        roles = "user", "assistant"
        first_message = True
        while True:
            try:
                inp = input(f"{roles[0]}: ")
            except EOFError:
                inp = ""
            if not inp:
                print("exit...")
                break
            print(f"{roles[1]}: ", end="")
            data_dict = self.pre_processing(inp, first_message)
            image = paddle.cast(data_dict["images"], self.compute_dtype)

            image_features = self.encode_images(image)[0]

            generate_ids, _ = self.generate_with_image_features(
                image_features,
                data_dict["input_ids"],
            )

            msg = self.post_processing(generate_ids)
            print("Outputs: ", msg)


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
        "--model_name_or_path",
        type=str,
        default="qwen-vl/qwen-vl-7b",
        help="The path of extraction model path that you want to load.",
    )
    parser.add_argument(
        "--device", default="gpu", choices=["gpu", "cpu", "xpu"], help="Device selected for inference."
    )
    parser.add_argument("--seed", default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--image_file", type=str, required=True)
    parser.add_argument("--conv_mode", type=str, default=None)

    args = parser.parse_args()

    paddle.seed(args.seed)

    predictor = Predictor(args)
    predictor.predict()

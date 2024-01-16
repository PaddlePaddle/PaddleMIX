import argparse
import os
import numpy as np
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["FLAGS_use_cuda_managed_memory"] = "true"

import paddle
from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlemix import QWenTokenizer, QwenVLProcessor

from utils import load_real_time_tokens


class Predictor(object):
    def __init__(self, args):
        self.args = args
        self.config = PretrainedConfig.from_pretrained(args.qwen_vl_config_path)
        self.tokenizer = QWenTokenizer.from_pretrained(args.qwen_vl_config_path)
        self.processor = QwenVLProcessor(tokenizer=self.tokenizer)
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
        [language_model_inputs] = self.first_predictor.run([pixel_values])
        return language_model_inputs

    @paddle.no_grad()
    def generate_with_image_features(self, image_features, input_ids):
        batch, seq,_ = image_features.shape
        seq = input_ids.shape[1]
        max_len = 1024
        dtype = "float16"
        tgt_generation_mask = paddle.full([batch, 1, 1, max_len], 0, dtype=dtype)
        tgt_generation_mask[:,0,0,:seq] = 1

        img_pos = None
        if paddle.any(input_ids == self.config.visual["image_start_id"]):
            bos_pos = paddle.where(input_ids == self.config.visual["image_start_id"])
            eos_pos = paddle.where(input_ids == self.config.visual["image_start_id"] + 1)
            assert (bos_pos[0] == eos_pos[0]).astype("bool").all()
            img_pos = paddle.concat((bos_pos[0], bos_pos[1], eos_pos[1]), axis=1)

        attention_mask = paddle.full([batch, 1, max_len, max_len], 0, dtype=dtype)
        attention_mask[:,0,:seq,:seq] = paddle.tril(paddle.ones(shape=(seq, seq), dtype=dtype))
        position_ids = paddle.full([batch, seq], 0, dtype="int64")
        for i in range(batch):
            position_ids[i,:] = paddle.to_tensor([i for i in range(seq)], dtype="int64")

        inputs = [
            input_ids, # input_ids
            image_features, # image_features
            img_pos, # img_pos
            attention_mask, # attention_mask
            position_ids,    # position_ids
            paddle.full([batch, 1], 1.0, dtype="float32"),  # penalty_score
            paddle.full([batch, 1], 0.0, dtype="float32"),  # frequency_score,
            paddle.full([batch, 1], 0.0, dtype="float32"),  # presence_score,
            paddle.full([batch, 1], 1, dtype="int64"),    # min_length,
            paddle.full([batch, 1], max_len - seq, dtype="int64"), # max_length,
            paddle.full([batch, 1], 1.0, dtype="float32"), # temperature,
            paddle.full([batch, 1], 0.0, dtype="float32"), # top_p,
            paddle.full([1], 151643, dtype="int64"),   # eos_token_id,
            paddle.full([batch, 1], seq, dtype="int32"),  # seq_len_encoder,
            paddle.full([batch, 1], seq, dtype="int32"), # seq_len_decoder,
            paddle.full([batch, 1], 0, dtype="int64"), # step_idx,
            paddle.full([batch, 1], False, dtype="bool"), # stop_flags,
            paddle.full([batch, 1], -123, dtype="int64"), # tgt_ids can be be initialized arbitrarily
            paddle.full([batch, 1], seq - 1, dtype="int64"), # tgt_pos,
            tgt_generation_mask, # tgt_generation_mask,
            paddle.full([batch, max_len], -100, dtype="int64"), # pre_ids, can be initialized arbitrarily
            paddle.full([1], batch, dtype="int64") # stop_nums, be batch 
        ]
        for i in range(32):
            tmp = paddle.rand(shape=[2, batch, 32, max_len, 128], dtype=dtype)
            inputs.append(tmp)

        self.second_predictor.run(inputs)
        tokens = load_real_time_tokens()
        generate_ids = tokens.tolist()
        return generate_ids, None

    def pre_processing(self, url, prompt):
        # input query
        query = []
        query.append({"image": url})
        query.append({"text": prompt})
        inputs = self.processor(query=query, return_tensors="pd")
        return inputs

    def post_processing(self, generate_ids):
        msg = self.processor.batch_decode(generate_ids)
        return msg

    def predict(self, url, prompt):
        inputs = self.pre_processing(url, prompt)
        images = inputs["images"]
        second_input_ids = inputs["input_ids"]

        image_features = self.encode_images(images)
        generate_ids, _ = self.generate_with_image_features(
            image_features,
            second_input_ids,
        )

        msg = self.post_processing(generate_ids)

        return msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_model_path", default='The dir name of image encoder model', type=str, help="", )
    parser.add_argument("--second_model_path", default='The dir name of language model', type=str, help="", )
    parser.add_argument("--qwen_vl_config_path", type=str,
                        default="The qwen vl config dir name of saving config",
                        help="The path of extraction model path that you want to load.")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu", "xpu"], help="Device selected for inference.")
    parser.add_argument("--seed", default=1234)
    args = parser.parse_args()

    paddle.seed(args.seed)

    predictor = Predictor(args)

    url = "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg"
    prompt = "Generate the caption in English with grounding:"

    # warp up
    warm_up_times = 2
    repeat_times = 10
    for i in range(warm_up_times):
        msg = predictor.predict(url, prompt)

    # test
    import time
    start_time = time.time()
    for i in range(repeat_times):
        msg = predictor.predict(url, prompt)
    end_time = time.time()
    during_time = end_time - start_time

    print("Outputs: ", msg)
    print("The whole end to end time : ", during_time / repeat_times * 1000, "ms")
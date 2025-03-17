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

import datetime
import sys
from dataclasses import dataclass, field

import numpy as np
import paddle
from paddle.distributed import fleet
from paddlenlp.generation import GenerationConfig
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import AutoConfig, AutoInferenceModelForCausalLM
from paddlenlp.trl import llm_utils

from paddlemix.models.qwen2_vl import MIXQwen2Tokenizer
from paddlemix.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLRotaryEmbedding,
)
from paddlemix.processors.qwen2_vl_processing import (
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    process_vision_info,
)

sys.path.append("PaddleNLP/llm/predict")
from predictor import ModelArgument, PredictorArgument


@dataclass
class Mix_PredictorArgument(PredictorArgument):
    question: str = field(default="Describe this image.", metadata={"help": "The question for the model."})
    image_file: str = field(
        default="paddlemix/demo_images/examples_image1.jpg", metadata={"help": "The image file for the model."}
    )


@dataclass
class Mix_ModelArgument(ModelArgument):
    pass


# NOTE: (zhoukangkang、changwenbin) Copied from PaddleMIX/paddlemix/models/qwen2_vl/modeling_qwen2_vl.py,
# for calculating M-ROPE.
def use_m_rope(vision_model_inputs):

    position_ids, _ = vl_model.get_rope_index(
        config.vision_config["spatial_merge_size"],
        config.image_token_id,
        config.video_token_id,
        config.vision_start_token_id,
        vision_model_inputs.get("input_ids"),
        vision_model_inputs.get("image_grid_thw"),
        vision_model_inputs.get("video_grid_thw", None),
        vision_model_inputs.get("attention_mask"),
    )
    position_start = position_ids[0][0][-1].item()
    position_end = config.max_position_embeddings - position_ids.shape[-1] + position_start
    position_value = (
        paddle.arange(position_start, position_end).reshape([1, 1, -1]).expand([position_ids.shape[0], 1, -1])
    )
    position_ids = paddle.concat([position_ids, position_value], axis=-1)

    head_dim = config.hidden_size // config.num_attention_heads
    qwen2_Embedding = Qwen2VLRotaryEmbedding(head_dim, config.max_position_embeddings, config.rope_theta)
    cos = qwen2_Embedding.cos_cached
    sin = qwen2_Embedding.sin_cached

    # NOTE: (zhoukangkang、changwenbin) Copied from PaddleMIX/paddlemix/models/qwen2_vl/modeling_qwen2_vl.py,
    # for calculating M-ROPE.
    cos = cos[position_ids]
    sin = sin[position_ids]
    mrope_section = config.rope_scaling["mrope_section"] * 2
    cos = paddle.concat(x=[m[i % 3] for i, m in enumerate(cos.split(mrope_section, axis=-1))], axis=-1)
    sin = paddle.concat(x=[m[i % 3] for i, m in enumerate(sin.split(mrope_section, axis=-1))], axis=-1)

    rope_emb = paddle.stack([cos, sin], axis=0)
    rope_emb = rope_emb.reshape([rope_emb.shape[0], 1, rope_emb.shape[2], 1, rope_emb.shape[-1]])

    return rope_emb


def init_llm_model_inputs(vision_model_inputs, inputs_embeds, arg_config: Mix_PredictorArgument):
    assert len(inputs_embeds.shape) == 3
    batch_size = inputs_embeds.shape[0]

    model_inputs = {}
    model_inputs["input_ids"] = paddle.zeros(shape=[batch_size, arg_config.total_max_length], dtype="int64")
    model_inputs["inputs_embeds"] = inputs_embeds

    # I dislike write (arg_config.total_max_length + arg_config.block_size -1 ) // arg_config.block_size
    assert arg_config.total_max_length % arg_config.block_size == 0

    model_inputs["top_p"] = paddle.full(shape=[batch_size, 1], fill_value=arg_config.top_p, dtype="float32")
    model_inputs["temperature"] = paddle.full(
        shape=[batch_size, 1], fill_value=arg_config.temperature, dtype="float32"
    )
    model_inputs["eos_token_id"] = paddle.to_tensor(
        np.array(llm_utils.get_eos_token_id(tokenizer, generation_config)).reshape(-1, 1).astype("int64")
    )
    model_inputs["penalty_score"] = paddle.full(
        shape=[batch_size, 1], fill_value=arg_config.repetition_penalty, dtype="float32"
    )
    model_inputs["frequency_score"] = paddle.full(shape=[batch_size, 1], fill_value=0.0, dtype="float32")
    model_inputs["presence_score"] = paddle.full(shape=[batch_size, 1], fill_value=0.0, dtype="float32")
    model_inputs["min_length"] = paddle.full(shape=[batch_size, 1], fill_value=arg_config.min_length, dtype="int64")
    model_inputs["max_length"] = paddle.full(shape=[batch_size, 1], fill_value=arg_config.max_length, dtype="int64")

    model_inputs["rope_emb"] = use_m_rope(vision_model_inputs)

    model_inputs["bad_tokens"] = paddle.to_tensor([-1], dtype="int64")
    model_inputs["is_block_step"] = paddle.full(shape=[batch_size], fill_value=False, dtype="bool")

    cache_kvs_shape = vl_model.model.get_cache_kvs_shape(vl_model.model.config, batch_size)
    cachekv_dtype = config.dtype if arg_config.cachekv_int8_type is None else "uint8"
    model_inputs["cache_kvs"] = [paddle.zeros(shape, dtype=cachekv_dtype) for shape in cache_kvs_shape]

    block_nums = arg_config.total_max_length // arg_config.block_size
    model_inputs["block_tables"] = paddle.arange(block_nums, dtype="int32").tile([batch_size, 1])

    seq_lens = inputs_embeds.shape[1]
    model_inputs["seq_lens_this_time"] = paddle.to_tensor(np.array(seq_lens).astype("int32").reshape(-1, 1))
    model_inputs["seq_lens_encoder"] = paddle.to_tensor(np.array(seq_lens).astype("int32").reshape(-1, 1))
    model_inputs["seq_lens_decoder"] = paddle.full(shape=[batch_size, 1], fill_value=0, dtype="int32")
    model_inputs["step_idx"] = paddle.full(shape=[batch_size, 1], fill_value=0, dtype="int64")
    model_inputs["not_need_stop"] = paddle.full(shape=[1], fill_value=True, dtype="bool")
    model_inputs["stop_flags"] = paddle.full(shape=[batch_size, 1], fill_value=False, dtype="bool")
    model_inputs["stop_nums"] = paddle.full(shape=[1], fill_value=batch_size, dtype="int64")
    model_inputs["pre_ids"] = paddle.full(shape=[batch_size, arg_config.max_length], fill_value=-1, dtype="int64")
    model_inputs["next_tokens"] = paddle.full(shape=[batch_size, 1], fill_value=-1, dtype="int64")

    return model_inputs


def run_model(predictor_args):

    question = "Describe this image."
    image_pad_token = "<|vision_start|><|image_pad|><|vision_end|>"
    text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{image_pad_token}{question}<|im_end|>\n<|im_start|>assistant\n"

    vision_model_inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pd",
    )
    input_tokens_len = vision_model_inputs.input_ids.shape[1]
    with paddle.no_grad():
        inputs_embeds = vl_model.vision_forward(**vision_model_inputs)
    llm_model_inputs = init_llm_model_inputs(vision_model_inputs, inputs_embeds, arg_config=predictor_args)
    generated_text = ""
    generated_ids = paddle.to_tensor([], dtype="int64").reshape([1, 0])
    while llm_model_inputs["not_need_stop"]:
        generated_id = vl_model.model.generate(**llm_model_inputs)  # already trimmed in paddle
        llm_model_inputs["input_ids"] = generated_id
        llm_model_inputs["inputs_embeds"] = None
        generated_ids = paddle.concat([generated_ids, generated_id], axis=1)
        if paddle.any(generated_id == 151645).item():
            break
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    output_tokens_len = generated_ids.shape[1]
    return generated_text, input_tokens_len, output_tokens_len


parser = PdArgumentParser((Mix_PredictorArgument, Mix_ModelArgument))
predictor_args, model_args = parser.parse_args_into_dataclasses()

paddle.set_default_dtype(predictor_args.dtype)
tensor_parallel_degree = paddle.distributed.get_world_size()
tensor_parallel_rank = paddle.distributed.get_rank()
if tensor_parallel_degree > 1:
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": tensor_parallel_degree,
        "pp_degree": 1,
        "sharding_degree": 1,
    }
    fleet.init(is_collective=True, strategy=strategy)

vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
    predictor_args.model_name_or_path,
    tensor_parallel_degree=tensor_parallel_degree,
    tensor_parallel_rank=tensor_parallel_rank,
    dtype=predictor_args.dtype,
    tensor_parallel_output=False,
)

vl_model.eval()

# NOTE: (zhoukangkang、changwenbin) Because we only use the visual model here,
# in order to reduce video memory,we delete the language model.
del vl_model.model
paddle.device.cuda.empty_cache()


image_processor = Qwen2VLImageProcessor()
tokenizer = MIXQwen2Tokenizer.from_pretrained(predictor_args.model_name_or_path)
processor = Qwen2VLProcessor(image_processor, tokenizer)
# min_pixels = 256*28*28 # 200704
# max_pixels = 1280*28*28 # 1003520

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": predictor_args.image_file,
            },
            {"type": "text", "text": predictor_args.question},
        ],
    }
]
# Preparation for inference
image_inputs, video_inputs = process_vision_info(messages)

config = AutoConfig.from_pretrained(predictor_args.model_name_or_path)
config.tensor_parallel_degree = tensor_parallel_degree
config.tensor_parallel_rank = tensor_parallel_rank
predictor_args.total_max_length = config.max_position_embeddings

# NOTE: (changwenbin) This is for using the inference optimization of paddlenlp qwen2.
config.model_type = "qwen2"
generation_config = GenerationConfig.from_pretrained(predictor_args.model_name_or_path)
fast_llm_model = AutoInferenceModelForCausalLM.from_pretrained(
    predictor_args.model_name_or_path,
    config=config,
    predictor_args=predictor_args,
    model_args=model_args,
    dtype=predictor_args.dtype,
    tensor_parallel_degree=tensor_parallel_degree,
    tensor_parallel_rank=tensor_parallel_rank,
)
fast_llm_model.eval()

vl_model.model = fast_llm_model

if predictor_args.benchmark:
    print(f"Benchmarking {predictor_args.model_name_or_path} ...")
    warm_up = 3
    repeat_times = 10
    sumtime = 0.0
    times = repeat_times + warm_up
    for i in range(times):
        if i > 2:
            paddle.device.synchronize()
            starttime = datetime.datetime.now()
        generated_text = run_model(predictor_args)
        if i > 2:
            paddle.device.synchronize()
            endtime = datetime.datetime.now()
            print("Final output_text:\n", generated_text[0])

        if i > 2:
            duringtime = endtime - starttime
            duringtime = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
            sumtime += duringtime
            print(f"Single Image Inference: {predictor_args.model_name_or_path} end-to-end time : ", duringtime, "ms")
    print(
        f"Single Image Inference: {predictor_args.model_name_or_path} average end-to-end time : ",
        sumtime / repeat_times,
        "ms",
    )
    print(f"GPU max_memory_allocated: {paddle.device.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
    print("input_tokens_len is :", generated_text[1], "tokens")
    print("output_tokens_len is :", generated_text[2], "tokens")

else:
    generated_text = run_model(predictor_args)
    print("Final output_text:\n", generated_text[0])

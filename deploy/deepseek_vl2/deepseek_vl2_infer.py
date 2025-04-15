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
from typing import Dict, List

import numpy as np
import paddle
import PIL.Image
from paddle.distributed import fleet
from paddlenlp.generation import GenerationConfig
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import AutoInferenceModelForCausalLM, DeepseekTokenizerFast
from paddlenlp.trl import llm_utils

from paddlemix.models.deepseek_vl2 import DeepseekVLV2Config, DeepseekVLV2ForCausalLM
from paddlemix.processors.deepseek_vl2_processing import DeepseekVLV2Processor

sys.path.append("PaddleNLP/llm/predict")
from predictor import ModelArgument, PredictorArgument


def load_pil_images(conversations: List[Dict[str, str]]) -> List[PIL.Image.Image]:
    """

        Args:
            conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
                [
                    {
                        "role": "User",
                        "content": "<image>
    Extract all information from this image and convert them into markdown format.",
                        "images": ["./examples/table_datasets.png"]
                    },
                    {"role": "Assistant", "content": ""},
                ]

        Returns:
            pil_images (List[PIL.Image.Image]): the list of PIL images.

    """
    pil_images = []
    for message in conversations:
        if "images" not in message:
            continue
        for image_path in message["images"]:
            pil_img = PIL.Image.open(image_path)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)
    return pil_images


@dataclass
class Mix_PredictorArgument(PredictorArgument):
    question: str = field(default="Describe this image.", metadata={"help": "The question for the model."})
    image_file: str = field(
        default="paddlemix/demo_images/examples_image1.jpg", metadata={"help": "The image file for the model."}
    )
    llm_mode: str = field(default="dynamic", metadata={"help": "The mode of llm. Supported values: dynamic, static"})


@dataclass
class Mix_ModelArgument(ModelArgument):
    pass


def init_llm_model_inputs(inputs_embeds, arg_config: Mix_PredictorArgument):
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

    model_inputs["bad_tokens"] = paddle.to_tensor([-1], dtype="int64")
    model_inputs["is_block_step"] = paddle.full(shape=[batch_size], fill_value=False, dtype="bool")

    cache_k_shapes, cache_v_shapes = vl_model.language.get_cache_kvs_shape(vl_model.language.config, batch_size)
    cachekv_dtype = config.dtype if arg_config.cachekv_int8_type is None else "uint8"
    cache_kvs = []
    if cache_k_shapes and cache_v_shapes:
        for cache_k_shape, cache_v_shape in zip(cache_k_shapes, cache_v_shapes):
            cache_kvs.append(paddle.zeros(cache_k_shape, dtype=cachekv_dtype))
            cache_kvs.append(paddle.zeros(cache_v_shape, dtype=cachekv_dtype))
    else:
        # for mla's absorption
        assert cache_v_shapes is None
        cache_kvs = [paddle.zeros(shape, dtype=cachekv_dtype) for shape in cache_k_shapes]

    model_inputs["cache_kvs"] = cache_kvs

    block_nums = arg_config.total_max_length // arg_config.block_size
    model_inputs["block_tables"] = paddle.arange(block_nums, dtype="int32").tile([batch_size, 1])

    seq_lens = inputs_embeds.shape[1]
    model_inputs["seq_lens_this_time"] = paddle.to_tensor(np.array(seq_lens).astype("int32").reshape(-1, 1))
    model_inputs["seq_lens_encoder"] = paddle.to_tensor(np.array(seq_lens).astype("int32").reshape(-1, 1))
    model_inputs["seq_lens_decoder"] = paddle.full(shape=[batch_size, 1], fill_value=0, dtype="int32")
    model_inputs["step_idx"] = paddle.full(shape=[batch_size, 1], fill_value=0, dtype="int64")
    model_inputs["not_need_stop"] = paddle.full(shape=[1], fill_value=True, dtype="bool").cpu()  # must at cpu place
    model_inputs["stop_flags"] = paddle.full(shape=[batch_size, 1], fill_value=False, dtype="bool")
    model_inputs["stop_nums"] = paddle.full(shape=[1], fill_value=batch_size, dtype="int64")
    model_inputs["pre_ids"] = paddle.full(shape=[batch_size, arg_config.max_length], fill_value=-1, dtype="int64")
    model_inputs["next_tokens"] = paddle.full(shape=[batch_size, 1], fill_value=-1, dtype="int64")

    return model_inputs


def run_model(predictor_args):

    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(conversations=conversation, images=pil_images, force_batchify=True, system_prompt="")
    prepare_inputs.images = prepare_inputs.images.astype(predictor_args.dtype)
    with paddle.no_grad():
        inputs_embeds = vl_model.prepare_inputs_embeds(**prepare_inputs)
    input_tokens_len = inputs_embeds.shape[1]
    llm_model_inputs = init_llm_model_inputs(inputs_embeds, arg_config=predictor_args)

    generated_text = ""
    generated_ids = paddle.to_tensor([], dtype="int64").reshape([1, 0])
    while llm_model_inputs["not_need_stop"]:
        generated_id = vl_model.language.generate(**llm_model_inputs)

        # NOTE: (changwenbin) , Get inputs_embeds from the visual model or input_ids.
        # Here we uniformly set the input of the language model to inputs_embeds
        llm_model_inputs["inputs_embeds"] = fast_llm_model.deepseek_v2.embed_tokens(generated_id)

        generated_ids = paddle.concat([generated_ids, generated_id], axis=1)
        if paddle.any(generated_id == tokenizer.eos_token_id).item():
            break
    generated_text = tokenizer.batch_decode(
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


paddle.set_default_dtype(predictor_args.dtype)
vl_model = DeepseekVLV2ForCausalLM.from_pretrained(
    predictor_args.model_name_or_path,
    tensor_parallel_degree=tensor_parallel_degree,
    tensor_parallel_rank=tensor_parallel_rank,
    dtype=predictor_args.dtype,
    tensor_parallel_output=False,
).eval()

# NOTE: (changwenbin) Because we only use the visual model here,
# in order to reduce video memory, we delete the language model.
del vl_model.language
paddle.device.cuda.empty_cache()


model_path = predictor_args.model_name_or_path
tokenizer = DeepseekTokenizerFast.from_pretrained(predictor_args.model_name_or_path)
config = DeepseekVLV2Config.from_pretrained(predictor_args.model_name_or_path)
processor = DeepseekVLV2Processor(
    tokenizer=tokenizer,
    candidate_resolutions=config["candidate_resolutions"],
    patch_size=config.vision_config["patch_size"],
    downsample_ratio=config["downsample_ratio"],
)


conversation = [
    {
        "role": "<|User|>",
        "content": f"<image>\n{predictor_args.question}",
        "images": [predictor_args.image_file],
    },
    {"role": "<|Assistant|>", "content": ""},
]


# register llm config
llm_config = config.language_config
llm_config.architectures = ["DeepseekVLV2ForCausalLM"]
llm_config.rope_scaling = {"factor": 1}
llm_config.rope_scaling_type = {}
llm_config.qk_rope_head_dim = 64
llm_config.rope_theta = 10000

generation_config = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    top_p=predictor_args.top_p,
    top_k=predictor_args.top_k,
    repetition_penalty=predictor_args.repetition_penalty,
    temperature=predictor_args.temperature,
    do_sample=False,
    trunc_input=True,
    use_cache=True,  # must true for infer
    return_dict=True,
)

fast_llm_model = AutoInferenceModelForCausalLM.from_pretrained(
    predictor_args.model_name_or_path,
    config=llm_config,
    predictor_args=predictor_args,
    model_args=model_args,
    dtype=predictor_args.dtype,
    tensor_parallel_degree=tensor_parallel_degree,
    tensor_parallel_rank=tensor_parallel_rank,
).eval()

# NOTE: (changwenbin) We convert the language model into a static graph
if predictor_args.llm_mode == "static":
    fast_llm_model = paddle.incubate.jit.inference(
        fast_llm_model,
        save_model_dir=f"./tmp/{predictor_args.model_name_or_path}/{predictor_args.quant_type}",
        enable_new_ir=True,
        cache_static_model=True,
        skip_prune_program=True,
        exp_enable_use_cutlass=False,
    )

vl_model.language = fast_llm_model

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

        # NOTE: (changwenbin) We delete the transformer_block to reduce the memory usage.
        if (fast_llm_model.deepseek_v2.transformer_block is not None) and (predictor_args.llm_mode == "static"):
            fast_llm_model.deepseek_v2.transformer_block = None
            paddle.device.cuda.empty_cache()

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

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

import paddle
from paddlenlp.transformers import Qwen2Tokenizer

from paddlemix.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from paddlemix.processors.qwen2_vl_processing import (
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    process_vision_info,
)
import argparse
from paddlenlp.experimental.transformers.qwen2.modeling import Qwen2ForCausalLMBlockInferenceModel
from paddlenlp.transformers import (
    AutoConfig,
    AutoInferenceModelForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedModel,
    PretrainedTokenizer,
)
from paddlenlp.generation import GenerationConfig
from paddlenlp.trl import llm_utils
import numpy as np

from dataclasses import dataclass, field
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.utils.log import logger


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description=" Use PaddleMIX to accelerate the Stable Diffusion3 image generation model."
#     )
#     parser.add_argument(
#         "--benchmark",
#         type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
#         default=False,
#         help="if set to True, measure inference performance",
#     )
#     parser.add_argument(
#         "--inference_optimize",
#         type=(lambda x: str(x).lower() in ["true", "1", "yes"]),
#         default=False,
#         help="If set to True, all optimizations except Triton are enabled.",
#     )
#     return parser.parse_args()

# args = parse_args()



MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
model_vision = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_NAME, dtype="bfloat16")

image_processor = Qwen2VLImageProcessor()
tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_NAME)
processor = Qwen2VLProcessor(image_processor, tokenizer)

# min_pixels = 256*28*28 # 200704
# max_pixels = 1280*28*28 # 1003520
# processor = Qwen2VLProcessor(image_processor, tokenizer, min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "paddlemix/demo_images/examples_image1.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
image_inputs, video_inputs = process_vision_info(messages)

question = "Describe this image."
image_pad_token = "<|vision_start|><|image_pad|><|vision_end|>"
text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{image_pad_token}{question}<|im_end|>\n<|im_start|>assistant\n"


@dataclass
class PredictorArgument:
    model_name_or_path: str = field(default=None, metadata={"help": "The directory of model."})
    model_prefix: str = field(default="model", metadata={"help": "the prefix name of static model"})
    src_length: int = field(default=1024, metadata={"help": "The max length of source text."})
    min_length: int = field(default=1, metadata={"help": "the min length for decoding."})
    max_length: int = field(default=128, metadata={"help": "the max length for decoding."})
    top_k: int = field(default=0, metadata={"help": "top_k parameter for generation"})
    top_p: float = field(default=0.7, metadata={"help": "top_p parameter for generation"})
    temperature: float = field(default=0.95, metadata={"help": "top_p parameter for generation"})
    repetition_penalty: float = field(default=1.0, metadata={"help": "repetition penalty parameter for generation"})
    device: str = field(default="gpu", metadata={"help": "Device"})
    dtype: str = field(default=None, metadata={"help": "Model dtype"})
    lora_path: str = field(default=None, metadata={"help": "The directory of LoRA parameters. Default to None"})
    export_precache: bool = field(default=False, metadata={"help": "whether use prefix weight to do infer"})
    prefix_path: str = field(
        default=None, metadata={"help": "The directory of Prefix Tuning parameters. Default to None"}
    )
    decode_strategy: str = field(
        default="sampling",
        metadata={
            "help": "the decoding strategy of generation, which should be one of ['sampling', 'greedy_search', 'beam_search']. Default to sampling"
        },
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention"},
    )

    mode: str = field(
        default="dynamic", metadata={"help": "the type of predictor, it should be one of [dynamic, static]"}
    )
    inference_model: bool = field(default=False, metadata={"help": "whether use InferenceModel to do generation"})
    quant_type: str = field(
        default="",
        metadata={
            "help": "Quantization type. Supported values: a8w8, a8w8c8, a8w8_fp8, a8w8c8_fp8, weight_only_int4, weight_only_int8"
        },
    )
    avx_model: bool = field(
        default=False, metadata={"help": "whether use AvxModel to do generation when using cpu inference"}
    )
    avx_type: str = field(
        default=None,
        metadata={
            "help": "avx compute type. Supported values: fp16, bf16,fp16_int8\
        fp16: first_token and next_token run in fp16\
        fp16_int8 : first_token run in fp16, next token run in int8"
        },
    )
    avx_cachekv_type: str = field(
        default="fp16",
        metadata={"help": "avx cachekv type. Supported values: fp16,int8"},
    )
    batch_size: int = field(default=1, metadata={"help": "The batch size of data."})
    benchmark: bool = field(
        default=False,
        metadata={
            "help": "If benchmark set as `True`, we will force model decode to max_length, which is helpful to compute throughput. "
        },
    )
    use_fake_parameter: bool = field(default=False, metadata={"help": "use fake parameter, for ptq scales now."})
    block_attn: bool = field(default=False, metadata={"help": "whether use block attention"})
    block_size: int = field(default=64, metadata={"help": "the block size for cache_kvs."})
    cachekv_int8_type: str = field(
        default=None,
        metadata={
            "help": "If cachekv_int8_type set as `dynamic`, cache kv would be quantized to int8 dynamically. If cachekv_int8_type set as `static`, cache kv would be quantized to int8 Statically."
        },
    )

    append_attn: bool = field(default=False, metadata={"help": "whether use append attention"})

    chat_template: str = field(
        default=None,
        metadata={
            "help": "the path of `chat_template.json` file to handle multi-rounds conversation. "
            "If is None(do not set --chat_template argument), it will use the default `chat_template.json`;"
            "If is equal with `model_name_or_path`, it will use the default loading; "
            "If is directory, it will find the `chat_template.json` under the directory; If is file, it will load it."
            "If is none string, it will not use chat_template.json."
        },
    )

    total_max_length: int = field(
        default=4096, metadata={"help": "Super parameter. Maximum sequence length(encoder+decoder)."}
    )

    def __post_init__(self):
        if self.append_attn:
            self.block_attn = True
        assert (
            self.src_length + self.max_length <= self.total_max_length
        ), "src_length + max_length should smaller than total_max_length."

@dataclass
class ModelArgument:
    model_type: str = field(
        default=None,
        metadata={"help": "the type of the model, which can be one of ['gpt-3', 'ernie-3.5-se', 'llama-img2txt']"},
    )
    data_file: str = field(default=None, metadata={"help": "data file directory"})
    output_file: str = field(default="output.json", metadata={"help": "predict result file directory"})

def init_model_inputs(arg_config: PredictorArgument):


    model_inputs = {}


    model_inputs["block_tables"] = paddle.full(
        shape=[arg_config.batch_size, (arg_config.total_max_length + arg_config.block_size - 1) // arg_config.block_size],
        fill_value=-1,
        dtype="int32",
    )
    model_inputs["top_p"] = paddle.full(
        shape=[arg_config.batch_size, 1], fill_value=arg_config.top_p, dtype="float32"
    )
    model_inputs["temperature"] = paddle.full(
        shape=[arg_config.batch_size, 1], fill_value=arg_config.temperature, dtype="float32"
    )
    model_inputs["eos_token_id"] = paddle.to_tensor(
        np.array(llm_utils.get_eos_token_id(tokenizer, generation_config)).reshape(-1, 1).astype("int64")
    )
    model_inputs["penalty_score"] = paddle.full(
        shape=[arg_config.batch_size, 1], fill_value=arg_config.repetition_penalty, dtype="float32"
    )
    model_inputs["frequency_score"] = paddle.full(
        shape=[arg_config.batch_size, 1], fill_value=0.0, dtype="float32"
    )
    model_inputs["presence_score"] = paddle.full(
        shape=[arg_config.batch_size, 1], fill_value=0.0, dtype="float32"
    )
    model_inputs["min_length"] = paddle.full(
        shape=[arg_config.batch_size, 1], fill_value=arg_config.min_length, dtype="int64"
    )
    model_inputs["max_length"] = paddle.full(
        shape=[arg_config.batch_size, 1], fill_value=arg_config.max_length, dtype="int64"
    )
    
    cache_kvs_shape = model.get_cache_kvs_shape(model.config, arg_config.batch_size)
    
    head_dim =cache_kvs_shape[0][-1]
    model_inputs["rope_emb"] = llm_utils.get_rotary_position_embedding(
        paddle.arange(arg_config.total_max_length).reshape((1, -1)), head_dim, config.rope_theta, config.rope_scaling
    )
    model_inputs["bad_tokens"] = paddle.to_tensor([-1], dtype="int64")
    model_inputs["is_block_step"] = paddle.full(shape=[arg_config.batch_size], fill_value=False, dtype="bool")
    

    cachekv_dtype = config.dtype if arg_config.cachekv_int8_type is None else "uint8"
    model_inputs["cache_kvs"] = [paddle.zeros(shape, dtype=cachekv_dtype) for shape in cache_kvs_shape]
    model_inputs["block_tables"][:][:] = -1
    seq_lens = [len(inputs["input_ids"][0])]

    max_block_nums = cache_kvs_shape[0][0]
    free_list = list(range(max_block_nums))
    for i in range(arg_config.batch_size):
        for j in range(
            (seq_lens[i] + arg_config.max_length + arg_config.block_size - 1) // arg_config.block_size
        ):
            used_block_id = free_list.pop()
            model_inputs["block_tables"][i, j] = used_block_id
    model_inputs["seq_lens_this_time"] = paddle.to_tensor(np.array(seq_lens).astype("int32").reshape(-1, 1))
    model_inputs["seq_lens_encoder"] = paddle.to_tensor(np.array(seq_lens).astype("int32").reshape(-1, 1))
    model_inputs["seq_lens_decoder"] = paddle.full(
        shape=[arg_config.batch_size, 1], fill_value=0, dtype="int32"
    )
    model_inputs["step_idx"] = paddle.full(shape=[arg_config.batch_size, 1], fill_value=0, dtype="int64")
    model_inputs["not_need_stop"] = paddle.full(shape=[1], fill_value=True, dtype="bool")
    model_inputs["stop_flags"] = paddle.full(
        shape=[arg_config.batch_size, 1], fill_value=False, dtype="bool"
    )
    model_inputs["stop_nums"] = paddle.full(shape=[1], fill_value=arg_config.batch_size, dtype="int64")
    model_inputs["pre_ids"] = paddle.full(
        shape=[arg_config.batch_size, arg_config.max_length], fill_value=-1, dtype="int64"
    )
    model_inputs["next_tokens"] = paddle.full(shape=[arg_config.batch_size, 1], fill_value=-1, dtype="int64")
    
    return model_inputs

parser = PdArgumentParser((PredictorArgument, ModelArgument))
predictor_args, model_args = parser.parse_args_into_dataclasses()

paddle.set_device(predictor_args.device)
paddle.set_default_dtype(predictor_args.dtype)

config = AutoConfig.from_pretrained(MODEL_NAME)
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
model = AutoInferenceModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    predictor_args=predictor_args,
    model_args=model_args,
    dtype=predictor_args.dtype,
    tensor_parallel_degree=1,
    tensor_parallel_rank=0,
)
model.eval()



if False:
    print("Benchmarking {MODEL_NAME}...")
    # warm_up = 3
    # for _ in range(warm_up):
    #     # Inference: Generation of the output
    #     inputs_embeds = vision_model.vision_forward(**inputs_vision)
    #     inputs["inputs_embeds"]=inputs_embeds
    #     generated_text = ""
    #     while inputs["not_need_stop"]:
    #         generated_ids = model.generate(**inputs)
    #         inputs["input_ids"] = generated_ids
    #         new_text_piece = processor.batch_decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #         if new_text_piece == "<|im_end|>":
    #             continue
    #         generated_text += new_text_piece
    #     print("Final output_text:\n", generated_text)
    import nvtx
    repeat_times = 10
    sumtime = 0.0
    for i in range(repeat_times):
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pd",
        )
        model_inputs = init_model_inputs(arg_config=predictor_args)

        paddle.device.synchronize()
        starttime = datetime.datetime.now()
        vision_nvtx = nvtx.start_range(message="vision", color="green")
        
        inputs_embeds = model_vision.vision_forward(**inputs)
        inputs.update(model_inputs)
        inputs["inputs_embeds"]=inputs_embeds
        
        paddle.device.synchronize()
        nvtx.end_range(vision_nvtx)
        
        llm_nvtx = nvtx.start_range(message="LLM", color="red")
        
        generated_text = ""
        while inputs["not_need_stop"]:
            llm_token_nvtx = nvtx.start_range(message="token", color="blue")
            
            generated_ids = model.generate(**inputs)  # already trimmed in paddle
            inputs["input_ids"] = generated_ids
            inputs["inputs_embeds"] = None
            new_text_piece = processor.batch_decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            if new_text_piece == "<|im_end|>":
                continue
            generated_text += new_text_piece
            
            addle.device.synchronize()
            nvtx.end_range(llm_token_nvtx)
            
        paddle.device.synchronize()
        nvtx.end_range(llm_nvtx)
        
        paddle.device.synchronize()
        endtime = datetime.datetime.now()
        print("Final output_text:\n", generated_text)

        duringtime = endtime - starttime
        duringtime = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
        sumtime += duringtime
        print(f"Single {MODEL_NAME} end to end time : ", duringtime, "ms")

        paddle.device.cuda.empty_cache()
        inference_global_mem = paddle.device.cuda.memory_reserved() / (1024**3)
        print(f"Inference used CUDA memory : {inference_global_mem:.3f} GiB")

    print(f"Single {MODEL_NAME} ave end to end time : ", sumtime / repeat_times, "ms")

else:
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pd",
    )
    model_inputs = init_model_inputs(arg_config=predictor_args)


    inputs_embeds = model_vision.vision_forward(**inputs)
    inputs.update(model_inputs)
    inputs["inputs_embeds"]=inputs_embeds
    generated_text = ""
    while inputs["not_need_stop"]:
        generated_ids = model.generate(**inputs, max_new_tokens=128)  # already trimmed in paddle
        inputs["input_ids"] = generated_ids
        inputs["inputs_embeds"] = None
        new_text_piece = processor.batch_decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        if new_text_piece == "<|im_end|>":
            continue
        generated_text += new_text_piece
    print("Final output_text:\n", generated_text)

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
import paddlenlp
import PIL.Image
from paddlenlp.experimental import transformers
from paddlenlp.experimental.transformers.deepseek_v2.modeling import (
    DeepseekV2BlockInferenceModel,
    DeepseekV2ForCausalLMBlockInferenceModel,
    DeepseekV2LMHead,
)
from paddlenlp.generation import GenerationConfig
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import (
    AutoInferenceModelForCausalLM,
    DeepseekTokenizerFast,
    DeepseekV2Config,
)
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)
from paddlenlp.transformers.model_utils import (
    dy2st_nocheck_guard_context,
    register_base_model,
)
from paddlenlp.trl import llm_utils

from paddlemix.models.deepseek_vl2 import DeepseekVLV2Config, DeepseekVLV2ForCausalLM
from paddlemix.processors.deepseek_vl2_processing import DeepseekVLV2Processor

sys.path.append("PaddleNLP/llm/predict")
from predictor import ModelArgument, PredictorArgument


class MIXDeepseekV2BlockInferenceModel(DeepseekV2BlockInferenceModel):
    def __init__(self, config: DeepseekV2Config, base_model_prefix: str):
        super(MIXDeepseekV2BlockInferenceModel, self).__init__(config, base_model_prefix)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        caches=None,
        pre_caches=None,
        **kwargs,
    ):

        seq_lens_this_time = kwargs.get("seq_lens_this_time", None)
        draft_tokens = kwargs.get("draft_tokens", None)
        seq_lens_encoder = kwargs.get("seq_lens_encoder", None)

        ids_remove_padding, padding_offset, cum_offsets, cu_seqlens_q, cu_seqlens_k = self.remove_padding(
            input_ids, seq_lens_this_time, draft_tokens, seq_lens_encoder
        )

        kwargs["cu_seqlens_q"] = cu_seqlens_q
        kwargs["cu_seqlens_k"] = cu_seqlens_k
        kwargs["padding_offsets"] = padding_offset
        kwargs["max_input_length"] = self.max_seq_len
        kwargs["block_size"] = self.block_size

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(ids_remove_padding)
        else:
            assert len(inputs_embeds.shape) == 3
            # This is the case in the image-to-text model
            # In the prefill phase, the language model is first fed with inputs_embeds instead of input_ids
            # but in decoder phase, the language model is fed with input_ids just like normal text-to-text model.
            inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[2]])
        with dy2st_nocheck_guard_context():
            hidden_states, _ = self.transformer_block(
                input_ids=input_ids,
                src=inputs_embeds,
                cum_offsets=cum_offsets,
                attn_mask=attention_mask,
                caches=caches,
                pre_caches=pre_caches,
                rotary_embs=None,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cum_offsets=cum_offsets,
        )


class MIXDeepseekV2ForCausalLMBlockInferenceModel(DeepseekV2ForCausalLMBlockInferenceModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: DeepseekV2Config, base_model_prefix: str = "deepseek_v2"):
        super(DeepseekV2ForCausalLMBlockInferenceModel, self).__init__(config)
        self.base_model_prefix = base_model_prefix
        self.max_candidate_len = config.get("speculate_max_candidate_len", 5)
        self.verify_window = config.get("speculate_verify_window", 2)
        self.max_seq_len = config.max_seq_len
        self.return_full_hidden_states = config.get("return_full_hidden_states", False)
        self.deepseek_v2 = MIXDeepseekV2BlockInferenceModel(config, base_model_prefix)
        if config.tie_word_embeddings:
            self.lm_head = DeepseekV2LMHead(
                config, embedding_weights=self.deepseek_v2.embed_tokens.weight, transpose_y=True
            )
            self.tie_weights()
        else:
            self.lm_head = DeepseekV2LMHead(config)

    def forward(
        self,
        input_ids,
        inputs_embeds=None,
        src_mask=None,
        pre_caches=None,
        caches=None,
        seq_lens_this_time=None,
        seq_lens_encoder=None,
        seq_lens_decoder=None,
        rope_emb=None,
        block_tables=None,
        k_quant_scales=None,
        v_quant_scales=None,
        k_dequant_scales=None,
        v_dequant_scales=None,
        draft_tokens=None,
        output_padding_offset=None,
    ):
        outputs = self.deepseek_v2(
            input_ids,
            inputs_embeds=inputs_embeds,
            src_mask=src_mask,
            caches=caches,
            rope_emb=None,
            block_tables=block_tables,
            pre_caches=pre_caches,
            seq_lens_this_time=seq_lens_this_time,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            k_quant_scales=k_quant_scales,
            v_quant_scales=v_quant_scales,
            k_dequant_scales=k_dequant_scales,
            v_dequant_scales=v_dequant_scales,
            draft_tokens=draft_tokens,
            output_padding_offset=output_padding_offset,
        )
        if self.return_full_hidden_states:
            from paddlenlp_ops import rebuild_padding_v2

            full_hidden_states = outputs[0]
            cum_offsets = outputs[1]
            hidden_states = rebuild_padding_v2(
                full_hidden_states,
                cum_offsets,
                seq_lens_decoder,
                seq_lens_encoder,
                output_padding_offset,
                self.max_seq_len,
            )
        else:
            hidden_states = outputs[0]
        logits = self.lm_head(
            hidden_states,
            tensor_parallel_output=False,
        )
        if self.return_full_hidden_states:
            return logits, full_hidden_states
        else:
            return logits

        return logits


@register_base_model
class DeepseekVLV2ForCausalLMBlockInferenceModel(MIXDeepseekV2ForCausalLMBlockInferenceModel):
    def __init__(self, config: DeepseekV2Config):
        super().__init__(config, base_model_prefix="language.model")

    def get_input_embeddings(self):
        return self.deepseek_v2.embed_tokens

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        if "language.lm_head.weight" in state_dict:
            self.lm_head.weight.set_value(
                paddle.to_tensor(state_dict["language.lm_head.weight"]).cast(self.lm_head.weight.dtype)
            )
        self.deepseek_v2.set_state_dict({k: state_dict[k] for k in state_dict.keys()})

    def forward(
        self,
        input_ids,
        inputs_embeds=None,
        src_mask=None,
        pre_caches=None,
        caches=None,
        seq_lens_this_time=None,
        seq_lens_encoder=None,
        seq_lens_decoder=None,
        rope_emb=None,
        block_tables=None,
        k_quant_scales=None,
        v_quant_scales=None,
        k_dequant_scales=None,
        v_dequant_scales=None,
        draft_tokens=None,
        output_padding_offset=None,
    ):
        outputs = self.deepseek_v2(
            input_ids,
            inputs_embeds=inputs_embeds,
            src_mask=src_mask,
            caches=caches,
            rope_emb=None,
            block_tables=block_tables,
            pre_caches=pre_caches,
            seq_lens_this_time=seq_lens_this_time,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            k_quant_scales=k_quant_scales,
            v_quant_scales=v_quant_scales,
            k_dequant_scales=k_dequant_scales,
            v_dequant_scales=v_dequant_scales,
            draft_tokens=draft_tokens,
            output_padding_offset=output_padding_offset,
        )
        if self.return_full_hidden_states:
            from paddlenlp_ops import rebuild_padding_v2

            full_hidden_states = outputs[0]
            cum_offsets = outputs[1]
            hidden_states = rebuild_padding_v2(
                full_hidden_states,
                cum_offsets,
                seq_lens_decoder,
                seq_lens_encoder,
                output_padding_offset,
                self.max_seq_len,
            )
        else:
            hidden_states = outputs[0]
        logits = self.lm_head(
            hidden_states,
            tensor_parallel_output=False,
        )
        if self.return_full_hidden_states:
            return logits, full_hidden_states
        else:
            return logits

        return logits

    def prepare_inputs_for_generation(self, **kwargs):
        # only last token for inputs_ids if cache is defined in kwargs
        input_ids = kwargs["input_ids"]
        inputs_embeds = kwargs.get("inputs_embeds", None)
        src_mask = kwargs.get("src_mask", None)
        block_tables = kwargs.get("block_tables", None)

        pre_caches = kwargs.get("pre_caches", None)
        caches = kwargs.get("caches", None)

        seq_lens_this_time = kwargs["seq_lens_this_time"]
        seq_lens_encoder = kwargs["seq_lens_encoder"]
        seq_lens_decoder = kwargs["seq_lens_decoder"]
        k_quant_scales = kwargs.get("k_quant_scales", None)
        v_quant_scales = kwargs.get("v_quant_scales", None)
        k_dequant_scales = kwargs.get("k_dequant_scales", None)
        v_dequant_scales = kwargs.get("v_dequant_scales", None)

        # speculative decoding related parameters
        draft_tokens = kwargs.get("draft_tokens", None)
        output_padding_offset = kwargs.get("output_padding_offset", None)

        model_inputs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "src_mask": src_mask,
            "rope_emb": None,
            "pre_caches": pre_caches,
            "caches": caches,
            "seq_lens_this_time": seq_lens_this_time,
            "seq_lens_encoder": seq_lens_encoder,
            "seq_lens_decoder": seq_lens_decoder,
            "block_tables": block_tables,
            "k_quant_scales": k_quant_scales,
            "v_quant_scales": v_quant_scales,
            "k_dequant_scales": k_dequant_scales,
            "v_dequant_scales": v_dequant_scales,
            "draft_tokens": draft_tokens,
            "output_padding_offset": output_padding_offset,
        }
        return model_inputs

# set deepseek_vl2 inference class
paddlenlp.experimental.transformers.deepseek_v2.modeling.DeepseekVLV2ForCausalLMBlockInferenceModel = (
    DeepseekVLV2ForCausalLMBlockInferenceModel
)


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
    image_file_1: str = field(
        default="paddlemix/demo_images/examples_image1.jpg", metadata={"help": "The image file for the model."}
    )
    image_file_2: str = field(
        default="paddlemix/demo_images/examples_image2.jpg", metadata={"help": "The image file for the model."}
    )
    image_file_3: str = field(
        default="paddlemix/demo_images/examples_image3.jpg", metadata={"help": "The image file for the model."}
    )


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
    model_inputs["not_need_stop"] = paddle.full(
        shape=[1], fill_value=True, dtype="bool"
    ).cpu()  # must at cpu place, paddlenlp_ops bug: update_inputs_v2
    model_inputs["stop_flags"] = paddle.full(shape=[batch_size, 1], fill_value=False, dtype="bool")
    model_inputs["stop_nums"] = paddle.full(shape=[1], fill_value=batch_size, dtype="int64")
    model_inputs["pre_ids"] = paddle.full(shape=[batch_size, arg_config.max_length], fill_value=-1, dtype="int64")
    model_inputs["next_tokens"] = paddle.full(shape=[batch_size, 1], fill_value=-1, dtype="int64")

    return model_inputs


def run_model(predictor_args, llm_model_inputs):
    # conversation = [
    #     {
    #         "role": "<|User|>",
    #         "content": f"<image>\n{predictor_args.question}.",
    #         "images": [predictor_args.image_file],
    #     },
    #     {"role": "<|Assistant|>", "content": ""},
    # ]

    # pil_images = load_pil_images(conversation)
    # prepare_inputs = processor(
    #     conversations=conversation, images=pil_images, force_batchify=True, system_prompt=""
    # )
    # prepare_inputs.images = prepare_inputs.images.astype(predictor_args.dtype)

    generated_text = ""
    generated_ids = paddle.to_tensor([], dtype="int64").reshape([1, 0])
    while llm_model_inputs["not_need_stop"]:
        generated_id = vl_model.language.generate(**llm_model_inputs)  # already trimmed in paddle
        llm_model_inputs["input_ids"] = generated_id
        llm_model_inputs["inputs_embeds"] = None
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

model_path = predictor_args.model_name_or_path
tokenizer = DeepseekTokenizerFast.from_pretrained(model_path)
config = DeepseekVLV2Config.from_pretrained(model_path)

candidate_resolutions = config["candidate_resolutions"]
patch_size = config.vision_config["patch_size"]
downsample_ratio = config["downsample_ratio"]
processor = DeepseekVLV2Processor(
    tokenizer=tokenizer,
    candidate_resolutions=candidate_resolutions,
    patch_size=patch_size,
    downsample_ratio=downsample_ratio,
)
paddle.set_default_dtype(predictor_args.dtype)
vl_model = DeepseekVLV2ForCausalLM.from_pretrained(model_path, dtype=predictor_args.dtype).eval()

del vl_model.language
paddle.device.cuda.empty_cache()

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

tensor_parallel_degree = paddle.distributed.get_world_size()
tensor_parallel_rank = paddle.distributed.get_rank()
fast_llm_model = AutoInferenceModelForCausalLM.from_pretrained(
    predictor_args.model_name_or_path,
    config=llm_config,
    predictor_args=predictor_args,
    model_args=model_args,
    dtype=predictor_args.dtype,
    tensor_parallel_degree=tensor_parallel_degree,
    tensor_parallel_rank=tensor_parallel_rank,
)
fast_llm_model.eval()

vl_model.language = fast_llm_model


conversation = [
    {
        "role": "<|User|>",
        "content": "This is image_1: <image>\n"
        "This is image_2: <image>\n"
        f"This is image_3: <image>\n {predictor_args.question}",
        "images": [
            predictor_args.image_file_1,
            predictor_args.image_file_2,
            predictor_args.image_file_3,
        ],
    },
    {"role": "<|Assistant|>", "content": ""},
]

if predictor_args.benchmark:
    print(f"Benchmarking {predictor_args.model_name_or_path} ...")
    warm_up = 3
    repeat_times = 10
    sumtime = 0.0
    times = repeat_times + warm_up
    for i in range(times):
        print("run", i)
        pil_images = load_pil_images(conversation)
        prepare_inputs = processor(
            conversations=conversation, images=pil_images, force_batchify=True, system_prompt=""
        )
        prepare_inputs.images = prepare_inputs.images.astype(predictor_args.dtype)
        with paddle.no_grad():
            inputs_embeds = vl_model.prepare_inputs_embeds(**prepare_inputs)
        input_tokens_len = inputs_embeds.shape[1]
        llm_model_inputs = init_llm_model_inputs(inputs_embeds, arg_config=predictor_args)

        if i > 2:
            paddle.device.synchronize()
            starttime = datetime.datetime.now()
        generated_text = run_model(predictor_args, llm_model_inputs)
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
    pil_images = load_pil_images(conversation)
    prepare_inputs = processor(conversations=conversation, images=pil_images, force_batchify=True, system_prompt="")
    prepare_inputs.images = prepare_inputs.images.astype(predictor_args.dtype)
    with paddle.no_grad():
        inputs_embeds = vl_model.prepare_inputs_embeds(**prepare_inputs)
    input_tokens_len = inputs_embeds.shape[1]
    llm_model_inputs = init_llm_model_inputs(inputs_embeds, arg_config=predictor_args)
    generated_text = run_model(predictor_args, llm_model_inputs)
    print("Final output_text:\n", generated_text[0])

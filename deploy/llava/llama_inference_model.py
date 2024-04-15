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

import paddle
from paddlenlp.experimental.transformers import LlamaForCausalLMInferenceModel

from paddlemix.models.llava.constants import IMAGE_TOKEN_INDEX


class LlamaForClipInferenceModel(LlamaForCausalLMInferenceModel):
    """
    This class is 99% like LlamaForCausalLMInferenceModel.
    Used only for llava's second part.
    """

    # def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
    #     images = kwargs.pop("images", None)
    #     image_sizes = kwargs.pop("image_sizes", None)

    #     inputs = super().prepare_inputs_for_generation(
    #         input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
    #     )

    #     if images is not None:
    #         inputs["images"] = images
    #     if image_sizes is not None:
    #         inputs["image_sizes"] = image_sizes
    #     return inputs

    # def prepare_attention_mask_for_generation(self, input_ids, pad_token_id, eos_token_id):
    #     is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(input_ids == pad_token_id).item()
    #     is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
    #         (eos_token_id is not None) and (pad_token_id != eos_token_id)
    #     )
    #     if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
    #         attention_mask = (input_ids == pad_token_id).astype(paddle.get_default_dtype()) * get_scale_by_dtype(
    #             return_positive=False
    #         )
    #     else:
    #         attention_mask = paddle.ones_like(input_ids, dtype=paddle.get_default_dtype())
    #     return attention_mask

    @paddle.no_grad()
    def generate_text_with_image_features(
        self,
        input_ids: paddle.Tensor,
        image_features: paddle.Tensor,
        penalty_score=None,
        frequency_score=None,
        presence_score=None,
        min_length=None,
        max_length=None,
        temperature=None,
        top_p=None,
        eos_token_id=None,
        seq_len_encoder=None,
        seq_len_decoder=None,
        step_idx=None,
        stop_flags=None,
        tgt_ids=None,
        tgt_pos=None,
        tgt_generation_mask=None,
        pre_ids=None,
        stop_nums=None,
        cache_kvs=[],
        **generate_kwargs
    ) -> paddle.Tensor:

        attention_mask = paddle.ones_like(x=input_ids, dtype="bool")

        position_ids = paddle.arange(start=0, end=input_ids.shape[1], dtype="int64")

        input_ids_list = []
        for i in range(len(input_ids)):
            input_ids_list.append(input_ids[i][attention_mask[i]])

        new_input_embeds = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids_list):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.llama.embed_tokens(cur_input_ids)
                cur_input_embeds_1 = paddle.cast(cur_input_embeds_1, dtype=cur_image_features.dtype)
                cur_input_embeds = paddle.concat(x=[cur_input_embeds_1, cur_image_features[0:0]], axis=0)
                new_input_embeds.append(cur_input_embeds)
                cur_image_idx += 1
                continue

            image_index = paddle.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].squeeze(axis=1)
            image_token_indices = paddle.concat(
                x=[
                    paddle.to_tensor([-1], dtype=cur_input_ids.dtype),
                    image_index.astype(cur_input_ids.dtype),
                    paddle.to_tensor([cur_input_ids.shape[0]], dtype=cur_input_ids.dtype),
                ],
                axis=0,
            )
            cur_input_ids_noim = []

            for i in range(image_token_indices.shape[0] - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])

            split_sizes = []
            for x in cur_input_ids_noim:
                split_sizes.append(x.shape[0])

            cur_input_embeds = self.llama.embed_tokens(paddle.concat(x=cur_input_ids_noim))
            # cur_input_embeds_no_im = paddle.split(x=cur_input_embeds, num_or_sections=split_sizes, axis=0)
            cur_input_embeds_no_im = []
            split_start = 0

            for i in range(image_token_indices.shape[0] - 1):
                if i == 0:
                    cur_input_embeds_no_im.append(cur_input_embeds[: split_sizes[i], ...])
                else:
                    split_start += split_sizes[i - 1]
                    cur_input_embeds_no_im.append(cur_input_embeds[split_start : split_start + split_sizes[i], ...])

            cur_new_input_embeds = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])

                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)

            cur_new_input_embeds = paddle.concat(x=cur_new_input_embeds)
            new_input_embeds.append(cur_new_input_embeds)

        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)

        if tokenizer_model_max_length is not None:
            for i in range(len(new_input_embeds)):
                new_input_embeds[i] = new_input_embeds[i][:tokenizer_model_max_length]

        max_len = 0
        for i in range(len(new_input_embeds)):
            max_len = max(new_input_embeds[i].shape[0], max_len)

        batch_size = len(new_input_embeds)
        new_input_embeds_padded = []

        attention_mask = paddle.zeros(shape=(batch_size, max_len), dtype=attention_mask.dtype)
        position_ids = paddle.zeros(shape=(batch_size, max_len), dtype=position_ids.dtype)
        for i, cur_new_embed in enumerate(new_input_embeds):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    paddle.concat(
                        x=(
                            paddle.zeros(shape=(max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype),
                            cur_new_embed,
                        ),
                        axis=0,
                    )
                )
                if cur_len > 0:
                    attention_mask[(i), -cur_len:] = True
                    position_ids[(i), -cur_len:] = paddle.arange(start=0, end=cur_len, dtype=position_ids.dtype)
            else:
                new_input_embeds_padded.append(
                    paddle.concat(
                        x=(
                            cur_new_embed,
                            paddle.zeros(shape=(max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype),
                        ),
                        axis=0,
                    )
                )
                if cur_len > 0:
                    attention_mask[(i), :cur_len] = True
                    position_ids[(i), :cur_len] = paddle.arange(start=0, end=cur_len, dtype=position_ids.dtype)

        new_input_embeds = paddle.stack(x=new_input_embeds_padded, axis=0)
        new_input_embeds = new_input_embeds.reshape([batch_size, -1, self.config.hidden_size])

        attention_mask = paddle.ones(shape=new_input_embeds.shape[:2], dtype="int64")
        batch_size, seq_length = attention_mask.shape
        position_ids = paddle.arange(seq_length).expand((batch_size, seq_length))

        outputs = self.generate(
            inputs_embeds=new_input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            penalty_score=penalty_score,
            frequency_score=frequency_score,
            presence_score=presence_score,
            min_length=min_length,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            seq_len_encoder=seq_len_encoder,
            seq_len_decoder=seq_len_decoder,
            step_idx=step_idx,
            stop_flags=stop_flags,
            tgt_ids=tgt_ids,
            tgt_pos=tgt_pos,
            tgt_generation_mask=tgt_generation_mask,
            pre_ids=pre_ids,
            stop_nums=stop_nums,
            cache_kvs=cache_kvs,
        )
        return outputs

    def to_static(self, output_path: str, config: dict):

        cache_kvs_shapes = self.get_cache_kvs_shape(config, max_length=config.get("max_length", None))
        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int32", name="inputs_ids"),
            paddle.static.InputSpec(
                shape=[None, None, None], dtype="float32", name="image_features"
            ),  # image_features
            # paddle.static.InputSpec(shape=[None, None], dtype="bool", name="attention_mask"),  # attention_mask
            # paddle.static.InputSpec(shape=[None, None], dtype="int64", name="position_ids"),  # position_ids
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="penalty_score"),  # penalty_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="frequency_score"),  # frequency_score
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="presence_score"),  # presence_score
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="min_length"),  # min_decode_length
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="max_length"),  # max_decode_length
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="temperature"),  # temperature
            paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="top_p"),  # top_p
            paddle.static.InputSpec(shape=[None], dtype="int64", name="eos_token_id"),  # eos_token_id
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_len_encoder"),  # seq_len_encoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_len_decoder"),  # seq_len_decoder
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="step_idx"),  # step_idx
            paddle.static.InputSpec(shape=[None, 1], dtype="bool", name="stop_flags"),  # stop_flags
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="tgt_ids"),  # tgt_ids
            paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="tgt_pos"),  # tgt_pos
            paddle.static.InputSpec(
                shape=[None, 1, 1, None], dtype="bool", name="tgt_generation_mask"
            ),  # tgt_generation_mask
            paddle.static.InputSpec(shape=[None, None], dtype="int64", name="pre_ids"),  # pre_ids
            paddle.static.InputSpec(shape=[1], dtype="int64", name="stop_nums"),  # stop_nums
            [
                paddle.static.InputSpec(
                    shape=shape,
                    dtype="float32",
                    name="cache_kvs_{}".format(i),
                )
                for i, shape in enumerate(cache_kvs_shapes)
            ],  # cache_kvs
        ]

        model = paddle.jit.to_static(self.generate_text_with_image_features, input_spec=input_spec)
        paddle.jit.save(model, output_path, skip_prune_program=True)


from paddlemix.auto import AutoConfigMIX

config = AutoConfigMIX.from_pretrained("paddlemix/llava/llava-v1.5-7b")
config.tensor_parallel_degree = 1
config.tensor_parallel_rank = 0
config.weight_only_quant_bits = -1
config.quant_type = None


model = LlamaForClipInferenceModel.from_pretrained("paddlemix/llava/llava-v1.5-7b", config=config)
model.eval()

model.to_static("./checkpoints/encode_text/llama", config)

# image_features = paddle.rand(shape=[1, 576, 4096], dtype="float32")
# input_ids = paddle.to_tensor([[1    , 319  , 13563, 1546 , 263  , 12758, 5199 , 322  , 385  , 23116,
#          21082, 20255, 29889, 450  , 20255, 4076 , 8444 , 29892, 13173, 29892,
#          322  , 1248 , 568  , 6089 , 304  , 278  , 5199 , 29915, 29879, 5155 ,
#          29889, 3148 , 1001 , 29901, 29871, -200 , 29871, 13   , 233  , 146  ,
#          146  , 235  , 194  , 179  , 30810, 232  , 188  , 136  , 31046, 319  ,
#          1799 , 9047 , 13566, 29901]], dtype="int64")

# max_len = config.max_length

# batch, seq, _ = image_features.shape
# seq += input_ids.shape[1]
# tgt_generation_mask = paddle.full([batch, 1, 1, max_len], 1)
# inputs = [
#     input_ids,  # input_ids
#     image_features,  # image_features
#     paddle.full([batch, 1], 1.0, dtype="float32"),  # penalty_score
#     paddle.full([batch, 1], 0.0, dtype="float32"),  # frequency_score,
#     paddle.full([batch, 1], 0.0, dtype="float32"),  # presence_score,
#     paddle.full([batch, 1], 1, dtype="int64"),  # min_length,
#     paddle.full([batch, 1], max_len, dtype="int64"),  # max_length,
#     paddle.full([batch, 1], 0.2, dtype="float32"),  # temperature,
#     paddle.full([batch, 1], 0.0, dtype="float32"),  # top_p,
#     paddle.full([1], config.eos_token_id, dtype="int64"),  # eos_token_id,
#     paddle.full([batch, 1], seq, dtype="int32"),  # seq_len_encoder,
#     paddle.full([batch, 1], seq, dtype="int32"),  # seq_len_decoder,
#     paddle.full([batch, 1], 0, dtype="int64"),  # step_idx,
#     paddle.full([batch, 1], False, dtype="bool"),  # stop_flags,
#     paddle.full([batch, 1], -123, dtype="int64"),  # tgt_ids can be be initialized arbitrarily
#     paddle.full([batch, 1], seq - 1, dtype="int64"),  # tgt_pos,
#     tgt_generation_mask,  # tgt_generation_mask,
#     paddle.full([batch, max_len], -1, dtype="int64"),  # pre_ids, can be initialized arbitrarily
#     paddle.full([1], batch, dtype="int64"),  # stop_nums, be batch
# ]

# for i in range(config.num_hidden_layers):
#     tmp = paddle.zeros(
#             shape=[ 2,
#                     batch,
#                     config.num_attention_heads ,
#                     max_len,
#                     config.hidden_size // config.num_attention_heads,
#                     ])

#     inputs.append(tmp)

# model.generate_text_with_image_features(
#     input_ids=inputs[0],
#     image_features=inputs[1],
#     penalty_score=inputs[2],
#     frequency_score=inputs[3],
#     presence_score=inputs[4],
#     min_length=inputs[5],
#     max_length=inputs[6],
#     temperature=inputs[7],
#     top_p=inputs[8],
#     eos_token_id=inputs[9],
#     seq_len_encoder=inputs[10],
#     seq_len_decoder=inputs[11],
#     step_idx=inputs[12],
#     stop_flags=inputs[13],
#     tgt_ids=inputs[14],
#     tgt_pos=inputs[15],
#     tgt_generation_mask=inputs[16],
#     pre_ids=inputs[17],
#     stop_nums=inputs[18],
#     cache_kvs=inputs[19:],
# )

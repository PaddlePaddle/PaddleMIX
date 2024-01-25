# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The Salesforce Team Authors and The HuggingFace Team. All rights reserved.
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

""" Paddle BLIP2 model."""

from typing import Optional, Tuple, Union
from contextlib import contextmanager

import paddle
import paddle.distributed as dist
import paddle.nn as nn
from paddlenlp.transformers.bloom.modeling import BloomForCausalLM
from paddlenlp.transformers.llama.modeling import LlamaForCausalLM
from paddlenlp.transformers.opt.modeling import OPTForCausalLM
from paddlenlp.transformers.t5.modeling import T5ForConditionalGeneration

from paddlemix.models.blip2.base_model import (
    Blip2ForConditionalGenerationModelOutput,
    Blip2ForStage1ModelOutput,
    Blip2PretrainedModel,
)
from paddlemix.models.blip2.modeling_utils import (
    all_gather_with_grad,
    concat_all_gather,
    disabled_train,
    masked_fill,
)
from paddlemix.models.blip2.Qformer import BertLMHeadModel
from paddlemix.utils.log import logger

from .configuration import Blip2Config

__all__ = [
    "Blip2ForConditionalGeneration",
]

BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/blip2-flan-t5-xl",
    "Salesforce/blip2-opt-2.7b",
]


@contextmanager
def dtype_guard(dtype="float32"):
    origin_dtype = paddle.get_default_dtype()
    paddle.set_default_dtype(dtype)
    try:
        yield
    finally:
        paddle.set_default_dtype(origin_dtype)


def Parameter(tensor):
    return paddle.create_parameter(
        tensor.shape,
        dtype=tensor.dtype,
        default_initializer=nn.initializer.Assign(tensor),
    )


class Blip2ForConditionalGeneration(Blip2PretrainedModel):
    config_class = Blip2Config
    main_input_name = "pixel_values"
    """
    Initialize the function to create an instance of Blip2.

    Args:
        config (`Blip2Config`): Configuration information for Blip2.

    Returns:
        Blip2(`Blip2PretrainedModel`): Returns an instance of Blip2PretrainedModel.

    """

    def __init__(
        self,
        config: Blip2Config,
    ):
        super().__init__(config)
        from paddlemix.models.blip2.eva_vit import VisionTransformer

        config.vision_config.update({"mp_degree": config.mp_degree})
        self.visual_encoder = VisionTransformer(config=config.vision_config)
        self.freeze_vit = config.freeze_vit
        self.train_stage1 = False
        if self.freeze_vit:
            # freeze vit except the post layer norm layer.
            for name, param in self.visual_encoder.named_parameters():
                param.stop_gradient = True
            self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logger.info("freeze vision encoder")
        if config.get("train_mode", None) == "stage1":
            self.train_stage1 = True
            tokenizer_name = config.qformer_config.tokenizer_name or "bert-base-uncased"
            self.tokenizer = self.init_tokenizer(tokenizer_name)
            self.Qformer = BertLMHeadModel(
                config=config.qformer_config,
                encoder_width=self.visual_encoder.num_features,
                train_in_satge1=True,
                tokenizer_length=len(self.tokenizer),
            )

            state_dict = self.Qformer.state_dict()
            for name, param in self.Qformer.named_parameters():
                if "_query" in name:
                    key_orig = name.replace("_query", "")
                    param.copy_(state_dict[key_orig], False)

            self.temp = self.create_parameter(
                shape=(1,), default_initializer=paddle.nn.initializer.Constant(value=0.07)
            )
            self.max_txt_len = config.get("max_txt_len")
        else:
            if config.use_decoder_only_language_model:
                name_or_path = (
                    config.text_config["_name_or_path"] if isinstance(config.text_config, dict) else config.text_config
                )
                if "opt" in name_or_path:
                    language_model = OPTForCausalLM.from_pretrained(
                        name_or_path,
                        load_state_as_np=True,
                        ignore_mismatched_sizes=True,
                    )


                elif "llama" in name_or_path:

                    from paddlenlp.transformers.llama.configuration import LlamaConfig

                    if config.mp_degree > 1:
                        import paddle.distributed.fleet as fleet

                        hcg = fleet.get_hybrid_communicate_group()
                        language_model = LlamaForCausalLM.from_pretrained(
                            name_or_path,
                            tensor_parallel_degree=config.mp_degree,
                            tensor_parallel_rank=hcg.get_model_parallel_rank(),
                            tensor_parallel_output=False,
                        )
                    else:
                        language_model = LlamaForCausalLM.from_pretrained(
                            name_or_path,
                            tensor_parallel_output=False,
                        )
                    language_model.hidden_size = LlamaConfig.from_pretrained(config.text_config).hidden_size
                    language_model.pad_token_id = LlamaConfig.from_pretrained(config.text_config).pad_token_id


                elif "bloom" in name_or_path:

                    import paddle.distributed.fleet as fleet
                    from paddlenlp.transformers.bloom.configuration import BloomConfig

                    hcg = fleet.get_hybrid_communicate_group()
                    llm_config = BloomConfig.from_pretrained("bigscience/bloom")
                    llm_config.tensor_parallel_degree = config.mp_degree
                    llm_config.dtype = "float16"
                    language_model = BloomForCausalLM(llm_config)
                    language_model.pad_token_id = llm_config.pad_token_id
                    language_model.hidden_size = llm_config.hidden_size

                elif "glm2" in name_or_path:
                    import paddle.distributed.fleet as fleet
                    from paddlenlp.transformers.chatglm_v2.configuration import ChatGLMv2Config
                    from paddlenlp.transformers.chatglm_v2.modeling import ChatGLMv2ForCausalLM

                    llm_config = ChatGLMv2Config.from_pretrained(config.text_config)

                    if config.mp_degree > 1:
                        hcg = fleet.get_hybrid_communicate_group()
                        language_model = ChatGLMv2ForCausalLM.from_pretrained(
                            config.text_config, 
                            tensor_parallel_degree=config.mp_degree,
                            tensor_parallel_rank=hcg.get_model_parallel_rank(),
                            tensor_parallel_output=False,
                        )
                    else:
                        language_model = ChatGLMv2ForCausalLM.from_pretrained(
                            config.text_config,
                            tensor_parallel_output=False,
                        )

                    language_model.pad_token_id = llm_config.pad_token_id
                    language_model.hidden_size = llm_config.hidden_size
                    
            else:
                from paddlenlp.transformers import T5Config

                t5_config = T5Config(config.text_config)
                for key, value in config.text_config.items():
                    t5_config[key] = config.text_config[key]

                language_model = T5ForConditionalGeneration(t5_config)
                language_model.hidden_size = t5_config["d_model"]

            self.language_model = language_model
            for name, param in self.language_model.named_parameters():
                param.stop_gradient = True
            self.pad_token_id = self.language_model.pad_token_id

            self.Qformer = BertLMHeadModel(
                config=config.qformer_config,
                encoder_width=self.visual_encoder.num_features,
                train_in_satge1=False,
                text_hidden_size=self.language_model.hidden_size,
                model_config=config,  # in order to pass some parameters that are not available in config.qformer_config
            )
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

    def get_input_embeddings(self) -> nn.Layer:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: paddle.Tensor,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
        text_input_stage1: Optional[paddle.Tensor] = None,
        **kwargs
    ):
        """
        pixel_values (Tensor): image pixels of shape `(batch_size, 3, H, W)`):
        input_ids (Tensor): Indices of input sequence tokens in the vocabulary.
        attention_mask Tensor: Tensor of integers valued 0 or 1, where 0 specifies paddings and should not be attended to by the model.
        return_dict (dict{Tensor}): whether to return dict
        text_input_stage1: text input for stage1
        """

        if self.train_stage1:
            return self.forward_stage1(pixel_values, text_input_stage1)
        else:
            return self.forward_stage2(
                pixel_values,
                input_ids,
                attention_mask,
                return_dict,
                decoder_input_ids=kwargs.get("decoder_input_ids", None),
                decoder_attention_mask=kwargs.get("decoder_attention_mask", None),
            )

    def forward_stage2(
        self,
        pixel_values: paddle.Tensor,
        input_ids: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
        decoder_input_ids: Optional[paddle.Tensor] = None,
        decoder_attention_mask: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        """
        pixel_values (Tensor): image pixels of shape `(batch_size, 3, H, W)`):
        input_ids (Tensor): Indices of input sequence tokens in the vocabulary.
        attention_mask Tensor: Tensor of integers valued 0 or 1, where 0 specifies paddings and should not be attended to by the model.
        return_dict (dict{Tensor}): whether to return dict
        text_input_stage1: text input for stage1
        decoder_input_ids: Indices of input sequence tokens in the vocabulary for encoder-decoder generation
        decoder_attention_mask: Tensor of integers valued 0 or 1 for encoder-decoder generation
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        with paddle.amp.auto_cast(level="O2"):
            image_embeds = self.Qformer.ln_vision(self.visual_encoder(pixel_values))
        image_embeds = image_embeds.astype("float32")

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = paddle.ones(image_embeds.shape[:-1], dtype="int64")
        query_tokens = self.Qformer.query_tokens.expand([image_embeds.shape[0], -1, -1])
        query_outputs = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.Qformer.language_projection(query_output)
        language_model_attention_mask = paddle.ones(language_model_inputs.shape[:-1], dtype="int64")
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = paddle.concat([language_model_inputs, inputs_embeds], axis=1)
        if attention_mask is None:
            attention_mask = paddle.ones_like(input_ids)

        attention_mask = paddle.concat([language_model_attention_mask, attention_mask], axis=1)

        if self.config.use_decoder_only_language_model:
            targets = input_ids * (1 - (input_ids == self.pad_token_id).astype(input_ids.dtype)) + (
                input_ids == self.pad_token_id
            ).astype(input_ids.dtype) * (-100)

            empty_targets = paddle.ones(language_model_attention_mask.shape, dtype="int64").fill_(-100)
            labels = paddle.concat([empty_targets, targets], axis=1)
            labels.stop_gradient = True
            with paddle.amp.auto_cast(level="O2"):
                outputs = self.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=labels,
                )
                loss = outputs.loss
        else:
            targets = decoder_input_ids * (
                1 - (decoder_input_ids == self.pad_token_id).astype(decoder_input_ids.dtype)
            ) + (decoder_input_ids == self.pad_token_id).astype(input_ids.dtype) * (-100)
            targets.stop_gradient = True
            with paddle.amp.auto_cast(level="O2"):
                outputs = self.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    decoder_attention_mask=decoder_attention_mask,
                    return_dict=return_dict,
                    labels=targets,
                )
                loss = outputs[0]
        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
        )

    def forward_stage1(self, pixel_values, text_input):
        text = text_input

        image = pixel_values
        image_embeds = self.Qformer.ln_vision(self.visual_encoder(image))

        image_atts = paddle.ones(image_embeds.shape[:-1], dtype="int64")
        query_tokens = self.Qformer.query_tokens.expand(shape=[image_embeds.shape[0], -1, -1])
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )
        image_feats = paddle.nn.functional.normalize(
            x=self.Qformer.vision_proj(query_output.last_hidden_state), axis=-1
        )

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_attention_mask=True,
            return_tensors="pd",
        )
        text_output = self.Qformer.bert(
            text_tokens.input_ids, attention_mask=text_tokens.attention_mask, return_dict=True
        )
        text_feat = paddle.nn.functional.normalize(
            self.Qformer.text_proj(text_output.last_hidden_state[:, 0, :]), axis=-1
        )

        # Image-text Contrastive
        # image_feats_all = image_feats
        # text_feat_all = text_feat
        image_feats_all = concat_all_gather(image_feats)
        text_feat_all = concat_all_gather(text_feat)
        sim_q2t = paddle.matmul(image_feats.unsqueeze(axis=1), text_feat_all.unsqueeze(axis=-1)).squeeze()
        sim_i2t = sim_q2t.max(axis=-1)
        sim_i2t = sim_i2t / self.temp
        sim_t2q = paddle.matmul(
            x=text_feat.unsqueeze(axis=1).unsqueeze(axis=1), y=image_feats_all.transpose(perm=[0, 2, 1])
        ).squeeze()
        sim_t2i = sim_t2q.max(axis=-1)
        sim_t2i = sim_t2i / self.temp

        rank = dist.get_rank()
        bs = image.shape[0]

        targets = paddle.linspace(start=rank * bs, stop=rank * bs + bs - 1, num=bs).astype(int)
        one_hot_label = paddle.nn.functional.one_hot(targets, num_classes=sim_i2t.shape[1])
        smooth_label = paddle.nn.functional.label_smooth(label=one_hot_label, epsilon=0.1)
        loss_itc = (
            paddle.nn.functional.cross_entropy(input=sim_i2t, label=smooth_label, soft_label=True)
            + paddle.nn.functional.cross_entropy(input=sim_t2i, label=smooth_label, soft_label=True)
        ) / 2
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        image_embeds_world = all_gather_with_grad(image_embeds)
        with paddle.no_grad():
            weights_t2i = paddle.nn.functional.softmax(x=sim_t2i, axis=1) + 0.0001
            weights_t2i_list = paddle.chunk(weights_t2i, chunks=paddle.distributed.get_world_size(), axis=-1)
            weights_t2i_list[rank].fill_diagonal_(value=0)
            weights_t2i = paddle.concat(weights_t2i_list, axis=-1)
            weights_i2t = paddle.nn.functional.softmax(x=sim_i2t, axis=1) + 0.0001
            weights_i2t_list = paddle.chunk(weights_i2t, chunks=paddle.distributed.get_world_size(), axis=-1)
            weights_i2t_list[rank].fill_diagonal_(value=0)
            weights_i2t = paddle.concat(weights_i2t_list, axis=-1)
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = paddle.multinomial(x=weights_t2i[b], num_samples=1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = paddle.stack(x=image_embeds_neg, axis=0)
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = paddle.multinomial(x=weights_i2t[b], num_samples=1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])
        text_ids_neg = paddle.stack(x=text_ids_neg, axis=0)
        text_atts_neg = paddle.stack(x=text_atts_neg, axis=0)
        text_ids_all = paddle.concat(x=[text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], axis=0)
        text_atts_all = paddle.concat(
            x=[text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg], axis=0
        )
        query_tokens_itm = self.Qformer.query_tokens.expand(shape=[text_ids_all.shape[0], -1, -1])
        query_atts_itm = paddle.ones(shape=query_tokens_itm.shape[:-1], dtype="int64")
        attention_mask_all = paddle.concat(x=[query_atts_itm, text_atts_all], axis=1)
        image_embeds_all = paddle.concat(x=[image_embeds, image_embeds_neg, image_embeds], axis=0)
        image_atts_all = paddle.ones(shape=image_embeds_all.shape[:-1], dtype="int64")
        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.shape[1], :]
        vl_output = self.Qformer.itm_head(vl_embeddings)
        logits = vl_output.mean(axis=1)

        itm_labels = paddle.concat([paddle.ones([bs], dtype="int64"), paddle.zeros([2 * bs], dtype="int64")], axis=0)
        loss_itm = paddle.nn.functional.cross_entropy(input=logits, label=itm_labels)
        # Image Captioning

        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, (0)] = self.tokenizer.bos_token_id
        labels = masked_fill(decoder_input_ids, decoder_input_ids == self.tokenizer.pad_token_id, -100)
        query_atts = paddle.ones(shape=query_tokens.shape[:-1], dtype="int64")
        attention_mask = paddle.concat(x=[query_atts, text_tokens.attention_mask], axis=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )
        loss_lm = lm_output.loss
        return Blip2ForStage1ModelOutput(
            loss=loss_itc + loss_itm + loss_lm, loss_itc=loss_itc, loss_itm=loss_itm, loss_lm=loss_lm
        )

    @paddle.no_grad()
    def generate_stage1(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
        Returns:
            captions (list): A list of ids of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))
        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, axis=0)
        else:
            num_beams = 1
        image_atts = paddle.ones(shape=image_embeds.shape[:-1], dtype="int64")
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
        input_ids = paddle.empty(shape=[image.shape[0], 1], dtype="int64").fill_(value=self.tokenizer.bos_token_id)
        query_tokens = self.query_tokens.expand(shape=[image_embeds.shape[0], -1, -1])
        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs,
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    @paddle.no_grad()
    def generate(
        self,
        pixel_values: paddle.Tensor,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        **generate_kwargs,
    ) -> paddle.Tensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.
        Args:
            pixel_values (`paddle.Tensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`paddle.Tensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`paddle.Tensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices
        Returns:
            captions (list): A list of ids of length batch_size * num_captions.
        """
        batch_size = pixel_values.shape[0]
        image_embeds = self.Qformer.ln_vision(
            self.visual_encoder(pixel_values.cast(self.visual_encoder.pos_embed.dtype))
        )
        image_attention_mask = paddle.ones(image_embeds.shape[:-1], dtype="int64")

        query_tokens = self.Qformer.query_tokens.expand([image_embeds.shape[0], -1, -1])
        query_outputs = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds.cast(query_tokens.dtype),
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs = self.Qformer.language_projection(query_output)
        language_attention_mask = paddle.ones(language_model_inputs.shape[:-1], dtype="int64")
        if input_ids is None:
            input_ids = paddle.to_tensor([[self.config.text_config.bos_token_id]]).tile([batch_size, 1])
        if attention_mask is None:
            attention_mask = paddle.ones_like(input_ids)
        attention_mask = paddle.concat([language_attention_mask, attention_mask], axis=1)
        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = paddle.concat([language_model_inputs.cast(inputs_embeds.dtype), inputs_embeds], axis=1)

        with dtype_guard(self.language_model._dtype):
            outputs = self.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                top_p=0.9,
                decode_strategy="greedy_search",
                temperature=1,
                num_beams=5,
                max_length=30,
                min_length=8,
                eos_token_id=50118,
                repetition_penalty=1,
                length_penalty=1,
                num_return_sequences=1,
            )

        return outputs

    @paddle.no_grad()
    def encode_image(
        self,
        pixel_values: paddle.Tensor,
        **kwargs,
    ):
        image_embeds = self.Qformer.ln_vision(self.visual_encoder(pixel_values.astype("float16")))
        image_embeds = image_embeds.astype("float32")

        image_attention_mask = paddle.ones(image_embeds.shape[:-1], dtype="int64")

        query_tokens = self.Qformer.query_tokens.expand([image_embeds.shape[0], -1, -1])
        query_outputs = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs[0]
        language_model_inputs = self.Qformer.language_projection(query_output)
        return language_model_inputs

    @paddle.no_grad()
    def predict_answers(
        self,
        pixel_values: paddle.Tensor,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        max_len=10,
        min_len=1,
        **kwargs
    ):
        """
        Args:
            pixel_values (`paddle.Tensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`paddle.Tensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`paddle.Tensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
        Returns:
            captions (list): A list of ids of length batch_size * num_captions.
        """
        # batch_size = pixel_values.shape[0]
        image_embeds = self.Qformer.ln_vision(self.visual_encoder(pixel_values))
        image_attention_mask = paddle.ones(image_embeds.shape[:-1], dtype="int64")

        query_tokens = self.Qformer.query_tokens.expand([image_embeds.shape[0], -1, -1])
        query_outputs = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs = self.Qformer.language_projection(query_output)
        language_attention_mask = paddle.ones(language_model_inputs.shape[:-1], dtype="int64")

        attention_mask = paddle.concat([language_attention_mask, attention_mask], axis=1)
        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = paddle.concat([language_model_inputs, inputs_embeds], axis=1)

        with dtype_guard(self.language_model._dtype):
            outputs = self.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                decode_strategy="greedy_search",
                temperature=1,
                num_beams=5,
                max_length=max_len,
                min_length=min_len,
                eos_token_id=50118,
                repetition_penalty=1,
                length_penalty=0,
            )

        return outputs

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
from paddlevlp.utils.log import logger
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute
import paddle.distributed as dist

from paddlenlp.transformers.model_outputs import ModelOutput
from paddlenlp.transformers.model_utils import PretrainedModel

from paddlevlp.models.blip2.modeling_opt import OPTForCausalLM
from paddlenlp.transformers.t5.configuration import T5Config
from paddlenlp.transformers.t5.modeling import T5ForConditionalGeneration
from paddlenlp.utils.initializer import normal_, ones_, zeros_
from paddlenlp.utils.log import logger
from .configuration import Blip2Config
from paddlevlp.models.blip2.Qformer import BertLMHeadModel
from paddlenlp.transformers import AutoTokenizer
from paddlevlp.models.blip2.modeling_utils import disabled_train, all_gather_with_grad, concat_all_gather, masked_fill

BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/blip2-flan-t5-xl",
    "Salesforce/blip2-opt-2.7b",
]

__all__ = ["Blip2ForConditionalGeneration", ]


def Parameter(tensor):
    return paddle.create_parameter(
        tensor.shape,
        dtype=tensor.dtype,
        default_initializer=nn.initializer.Assign(tensor), )


@dataclass
class Blip2ForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`Blip2ForConditionalGeneration`].
    Args:
        loss (`paddle.Tensor`, *optional*, returned when `labels` is provided, `paddle.Tensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`paddle.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[paddle.Tensor]] = None
    logits: Optional[Tuple[paddle.Tensor]] = None
    vision_outputs: Optional[paddle.Tensor] = None
    qformer_outputs: Optional[Tuple[paddle.Tensor]] = None
    language_model_outputs: Optional[Tuple[paddle.Tensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in
            ["vision_outputs", "qformer_outputs", "language_model_outputs"] else
            getattr(self, k).to_tuple() for k in self.keys())


@dataclass
class Blip2ForStage1ModelOutput(Blip2ForConditionalGenerationModelOutput):
    """
    Class defining the outputs of [`Blip2ForStage1ModelOutput`].
    """
    loss: Optional[Tuple[paddle.Tensor]] = None
    loss_itc: Optional[Tuple[paddle.Tensor]] = None
    loss_itm: Optional[paddle.Tensor] = None
    loss_lm: Optional[Tuple[paddle.Tensor]] = None


class Blip2PretrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Blip2Config
    base_model_prefix = "blip"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"language_model.encoder.embed_tokens.weight",
        r"language_model.decoder.embed_tokens.weight",
    ]
    _no_split_modules = ["Blip2Attention", "T5Block", "OPTDecoderLayer"]
    _keep_in_fp32_modules = ["wo"]

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_range
        if (isinstance(module, nn.Conv2D) or isinstance(module, nn.Embedding) or
                isinstance(module, nn.Linear)):
            normal_(module.weight, mean=0.0, std=factor)
            if hasattr(module, "bias") and module.bias is not None:
                zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            zeros_(module.bias)
            ones_(module.weight)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            zeros_(module.bias)

    @classmethod
    def init_tokenizer(cls, tokenizer_name="bert-base-uncased"):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        from_hf_hub: bool=False,
                        subfolder: str=None,
                        *args,
                        **kwargs):
        """
        Creates an instance of `PretrainedModel`. Model weights are loaded
        by specifying name of a built-in pretrained model, a pretrained model from HF Hub, a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model or dir path
                to load from. The string can be:

                - Name of a built-in pretrained model
                - Name of a pretrained model from HF Hub
                - Name of a community-contributed pretrained model.
                - Local directory path which contains model weights file("model_state.pdparams")
                  and model config file ("model_config.json").
            from_hf_hub (bool): load model from huggingface hub. Default to `False`.
            subfolder (str, optional) An optional value corresponding to a folder inside the repo.
                Only works when loading from Huggingface Hub.
            *args (tuple): Position arguments for model `__init__`. If provided,
                use these as position argument values for model initialization.
            **kwargs (dict): Keyword arguments for model `__init__`. If provided,
                use these to update pre-defined keyword argument values for model
                initialization. If the keyword is in `__init__` argument names of
                base model, update argument values of the base model; else update
                argument values of derived model.
            load_state_as_np (bool, optional): The weights read in can be choosed
                to place on CPU or GPU though the model is on the default device.
                If `True`, load the model weights as `numpy.ndarray` on CPU.
                Otherwise, weights would be loaded as tensors on the default
                device. Note that if on GPU, the latter would creates extra
                temporary tensors in addition to the model weights, which
                doubles the memory usage . Thus it is suggested to use `True`
                for big models on GPU. Default to `False`.

        Returns:
            PretrainedModel: An instance of `PretrainedModel`.

        Example:
            .. code-block::

                from paddlenlp.transformers import BertForSequenceClassification

                # Name of built-in pretrained model
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

                # Name of pretrained model from PaddleHub
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

                # Name of community-contributed pretrained model
                model = BertForSequenceClassification.from_pretrained('yingyibiao/bert-base-uncased-sst-2-finetuned', num_labels=3)

                # Load from local directory path
                model = BertForSequenceClassification.from_pretrained('./my_bert/'
        """
        import os
        from paddlenlp.transformers.utils import (
            ContextManagers, InitTrackerMeta, adapt_stale_fwd_patch,
            fn_args_to_dict, is_paddle_support_lazy_init, resolve_cache_dir,
            weight_name_suffix)
        from paddlenlp.transformers.configuration_utils import PretrainedConfig
        from paddlenlp.utils.env import (
            CONFIG_NAME,
            ENABLE_TORCH_CHECKPOINT,
            LEGACY_CONFIG_NAME,
            PADDLE_WEIGHT_FILE_NAME,
            PYTORCH_WEIGHT_FILE_NAME, )
        from paddlenlp.transformers.model_utils import no_init_weights
        load_state_as_np = kwargs.pop("load_state_as_np", False)
        config = kwargs.pop("config", None)
        force_download = kwargs.pop("force_download", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", None)
        cache_dir = kwargs.pop("cache_dir", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
        dtype = kwargs.pop("dtype", None)

        cache_dir = resolve_cache_dir(pretrained_model_name_or_path,
                                      from_hf_hub, cache_dir)

        model_kwargs = kwargs
        # 1. get the PretrainedConfig to init model
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                from_hf_hub=from_hf_hub,
                subfolder=subfolder,
                **kwargs, )

        if dtype is None:
            dtype = config.dtype
        else:
            config.dtype = dtype

        if not os.path.exists(os.path.join(cache_dir, CONFIG_NAME)):
            config.save_pretrained(cache_dir)

        init_contexts = []
        if low_cpu_mem_usage:
            load_state_as_np = True
            # Instantiate model.
            init_contexts.append(no_init_weights(_enable=True))
            if is_paddle_support_lazy_init():
                init_contexts.append(paddle.LazyGuard())

        if dtype:
            init_contexts.append(dtype_guard(dtype))

        # 2. resolve model_weight file
        support_conversion = cls.support_conversion(
            config) and ENABLE_TORCH_CHECKPOINT

        model_weight_file = cls._resolve_model_file_path(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            subfolder=subfolder,
            from_hf_hub=from_hf_hub,
            config=config,
            support_conversion=support_conversion, )

        if model_weight_file.endswith(PYTORCH_WEIGHT_FILE_NAME):
            if support_conversion:
                # try to get the name-mapping info
                logger.info(
                    f"start to convert pytorch weight file<{model_weight_file}> to "
                    f"paddle weight file<{os.path.join(cache_dir, PADDLE_WEIGHT_FILE_NAME)}> ..."
                )
                model_state_dict = cls.convert(model_weight_file, config,
                                               cache_dir)
            else:
                raise ValueError(
                    f"download the {PYTORCH_WEIGHT_FILE_NAME} weight file, but model<{cls}> "
                    "don't support conversion from pytorch weight file to paddle weight file "
                    "or conversion is been disabled by `ENABLE_TORCH_CHECKPOINT` environment variable"
                )

        else:
            # 4. loading the state dict
            if config.tensor_parallel_degree > 1 and model_weight_file.endswith(
                    "model_state.pdparams"):
                model_state_dict = cls.convert_tensor_parallel(
                    model_weight_file, config)
            else:
                model_state_dict = paddle.load(
                    model_weight_file, return_numpy=load_state_as_np)

        # 3. init the model
        init_args = config["init_args"] or ()
        with ContextManagers(init_contexts):
            model = cls(config, *init_args, **model_kwargs)

        loaded_state_dict_keys = list(model_state_dict.keys())
        from paddlevlp.models.blip2.eva_vit import interpolate_pos_embed
        interpolate_pos_embed(model, model_state_dict)
        # TODO(wj-Mcat): load shard checkpoint weight file, refer to: https://github.com/huggingface/transformers/pull/16343
        model, missing_keys, unexpected_keys, mismatched_keys = cls._load_pretrained_model(
            model=model,
            state_dict=model_state_dict,
            loaded_keys=loaded_state_dict_keys,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            dtype=dtype, )

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(
                f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n"
            )

        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training.")
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join([
                f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                for key, shape1, shape2 in mismatched_keys
            ])
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference.")
        if paddle.in_dynamic_mode():
            return model

        return model, model_state_dict


class Blip2ForConditionalGeneration(Blip2PretrainedModel):
    config_class = Blip2Config
    main_input_name = "pixel_values"
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"language_model.encoder.embed_tokens.weight",
        r"language_model.decoder.embed_tokens.weight",
    ]

    def __init__(
            self,
            config: Blip2Config, ):
        super().__init__(config)
        from paddlevlp.models.blip2.eva_vit import VisionTransformer
        self.visual_encoder = VisionTransformer.from_pretrained(
            pretrained_model_name_or_path=config.vision_config,
            mp_degree=config.mp_degree)
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
            self.tokenizer = self.init_tokenizer()
            self.Qformer = BertLMHeadModel.from_pretrained(
                pretrained_model_name_or_path=config.qformer_config,
                encoder_width=self.visual_encoder.num_features,
                train_in_satge1=True,
                tokenizer_length=len(self.tokenizer),
                mp_degree=config.mp_degree)

            state_dict = self.Qformer.state_dict()
            for name, param in self.Qformer.named_parameters():
                if '_query' in name:
                    key_orig = name.replace('_query', '')
                    param.copy_(state_dict[key_orig], False)  ### problem

            self.temp = self.create_parameter(
                shape=(1, ),
                default_initializer=paddle.nn.initializer.Constant(value=0.07))
            self.max_txt_len = config.get("max_txt_len")
        else:
            if config.use_decoder_only_language_model:
                if "opt" in config.text_config:
                    language_model = OPTForCausalLM.from_pretrained(
                        config.text_config,
                        load_state_as_np=True,
                        mp_degree=config.mp_degree)
                else:
                    raise NotImplementedError
            else:
                if "t5" in config.text_config:
                    language_model = T5ForConditionalGeneration(
                        config.text_config)
                else:
                    raise NotImplementedError

            self.language_model = language_model
            for name, param in self.language_model.named_parameters():
                param.stop_gradient = True
            self.pad_token_id = self.language_model.pad_token_id

            self.Qformer = BertLMHeadModel.from_pretrained(
                pretrained_model_name_or_path=config.qformer_config,
                encoder_width=self.visual_encoder.num_features,
                train_in_satge1=False,
                text_hidden_size=self.language_model.hidden_size,
                mp_degree=config.mp_degree)
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

    def get_input_embeddings(self) -> nn.Layer:
        return self.vision_model.embeddings.patch_embedding

    def forward(self,
                pixel_values: paddle.Tensor,
                input_ids: paddle.Tensor=None,
                attention_mask: Optional[paddle.Tensor]=None,
                return_dict: Optional[bool]=None,
                text_input_stage1: Optional[paddle.Tensor]=None,
                **kwargs):

        if self.train_stage1:
            return self.forward_stage1(pixel_values, text_input_stage1)
        else:
            return self.forward_stage2(
                pixel_values,
                input_ids,
                attention_mask,
                return_dict, )

    def forward_stage2(
            self,
            pixel_values: paddle.Tensor,
            input_ids: paddle.Tensor,
            attention_mask: Optional[paddle.Tensor]=None,
            return_dict: Optional[bool]=None,
            **kwargs) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        r"""
        Returns:
        Examples:
        Image captioning (without providing a text prompt):
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from paddlenlp.transformers import Blip2Processor, Blip2ForConditionalGeneration
        >>> import paddle
        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        >>> model = Blip2ForConditionalGeneration.from_pretrained(
        ...     "Salesforce/blip2-flan-t5-xl"
        ... )
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pd")
        >>> generated_ids, scores = model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        two cats laying on a couch
        ```
        Visual question answering (prompt = question):
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from paddlenlp.transformers import Blip2Processor, Blip2ForConditionalGeneration
        >>> import paddle
        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        >>> model = Blip2ForConditionalGeneration.from_pretrained(
        ...     "Salesforce/blip2-flan-t5-xl"
        ... )
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> prompt = "Question: how many cats are there? Answer:"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pd")
        >>> generated_ids, scores= model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        two
        ```"""
        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)
        with paddle.amp.auto_cast(level='O2'):
            image_embeds = self.Qformer.ln_vision(
                self.visual_encoder(pixel_values))
        image_embeds = image_embeds.astype("float32")

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = paddle.ones(
            image_embeds.shape[:-1], dtype="int64")
        query_tokens = self.Qformer.query_tokens.expand(
            [image_embeds.shape[0], -1, -1])
        query_outputs = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True, )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.Qformer.language_projection(query_output)
        language_model_attention_mask = paddle.ones(
            language_model_inputs.shape[:-1], dtype="int64")
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = paddle.concat(
            [language_model_inputs, inputs_embeds], axis=1)
        if attention_mask is None:
            attention_mask = paddle.ones_like(input_ids)

        attention_mask = paddle.concat(
            [language_model_attention_mask, attention_mask], axis=1)

        targets = input_ids * (1 - (
            input_ids == self.pad_token_id).astype(input_ids.dtype)) + (
                input_ids == self.pad_token_id).astype(input_ids.dtype) * (-100)

        empty_targets = paddle.ones(
            language_model_attention_mask.shape, dtype="int64").fill_(-100)
        labels = paddle.concat([empty_targets, targets], axis=1)
        labels.stop_gradient = True
        with paddle.amp.auto_cast(level='O2'):
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=labels, )
            loss = outputs.loss
        return Blip2ForConditionalGenerationModelOutput(loss=loss, )

    def forward_stage1(self, pixel_values, text_input):
        text = text_input

        image = pixel_values
        image_embeds = self.Qformer.ln_vision(self.visual_encoder(image))

        image_atts = paddle.ones(image_embeds.shape[:-1], dtype="int64")
        query_tokens = self.Qformer.query_tokens.expand(
            shape=[image_embeds.shape[0], -1, -1])
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True)
        image_feats = paddle.nn.functional.normalize(
            x=self.Qformer.vision_proj(query_output.last_hidden_state), axis=-1)

        text_tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_txt_len,
            return_attention_mask=True,
            return_tensors="pd")
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True)
        text_feat = paddle.nn.functional.normalize(
            self.Qformer.text_proj(text_output.last_hidden_state[:, 0, :]),
            axis=-1)

        ###============== Image-text Contrastive ===================###
        # image_feats_all = image_feats
        # text_feat_all = text_feat
        image_feats_all = concat_all_gather(image_feats)
        text_feat_all = concat_all_gather(text_feat)
        sim_q2t = paddle.matmul(
            image_feats.unsqueeze(axis=1),
            text_feat_all.unsqueeze(axis=-1)).squeeze()
        sim_i2t = sim_q2t.max(axis=-1)
        sim_i2t = sim_i2t / self.temp
        sim_t2q = paddle.matmul(
            x=text_feat.unsqueeze(axis=1).unsqueeze(axis=1),
            y=image_feats_all.transpose(perm=[0, 2, 1])).squeeze()
        sim_t2i = sim_t2q.max(axis=-1)
        sim_t2i = sim_t2i / self.temp

        rank = dist.get_rank()
        bs = image.shape[0]

        targets = paddle.linspace(
            start=rank * bs, stop=rank * bs + bs - 1, num=bs).astype(int)
        one_hot_label = paddle.nn.functional.one_hot(
            targets, num_classes=sim_i2t.shape[1])
        smooth_label = paddle.nn.functional.label_smooth(
            label=one_hot_label, epsilon=0.1)
        loss_itc = (paddle.nn.functional.cross_entropy(
            input=sim_i2t, label=smooth_label, soft_label=True) +
                    paddle.nn.functional.cross_entropy(
                        input=sim_t2i, label=smooth_label, soft_label=True)) / 2
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(
            text_tokens.attention_mask)
        image_embeds_world = all_gather_with_grad(image_embeds)
        with paddle.no_grad():
            weights_t2i = paddle.nn.functional.softmax(
                x=sim_t2i, axis=1) + 0.0001
            weights_t2i_list = paddle.chunk(
                weights_t2i,
                chunks=paddle.distributed.get_world_size(),
                axis=-1)
            weights_t2i_list[rank].fill_diagonal_(value=0)
            weights_t2i = paddle.concat(weights_t2i_list, axis=-1)
            weights_i2t = paddle.nn.functional.softmax(
                x=sim_i2t, axis=1) + 0.0001
            weights_i2t_list = paddle.chunk(
                weights_i2t,
                chunks=paddle.distributed.get_world_size(),
                axis=-1)
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
        text_ids_all = paddle.concat(
            x=[text_tokens.input_ids, text_tokens.input_ids, text_ids_neg],
            axis=0)
        text_atts_all = paddle.concat(
            x=[
                text_tokens.attention_mask, text_tokens.attention_mask,
                text_atts_neg
            ],
            axis=0)
        query_tokens_itm = self.Qformer.query_tokens.expand(
            shape=[text_ids_all.shape[0], -1, -1])
        query_atts_itm = paddle.ones(
            shape=query_tokens_itm.shape[:-1], dtype='int64')
        attention_mask_all = paddle.concat(
            x=[query_atts_itm, text_atts_all], axis=1)
        image_embeds_all = paddle.concat(
            x=[image_embeds, image_embeds_neg, image_embeds], axis=0)
        image_atts_all = paddle.ones(
            shape=image_embeds_all.shape[:-1], dtype='int64')
        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True)
        vl_embeddings = output_itm.last_hidden_state[:, :query_tokens_itm.shape[
            1], :]
        vl_output = self.Qformer.itm_head(vl_embeddings)
        logits = vl_output.mean(axis=1)

        itm_labels = paddle.concat(
            [
                paddle.ones(
                    [bs], dtype='int64'), paddle.zeros(
                        [2 * bs], dtype='int64')
            ],
            axis=0)
        loss_itm = paddle.nn.functional.cross_entropy(
            input=logits, label=itm_labels)
        ##================= Image Captioning ========================##

        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, (0)] = self.tokenizer.bos_token_id
        labels = masked_fill(decoder_input_ids,
                             decoder_input_ids == self.tokenizer.pad_token_id,
                             -100)
        query_atts = paddle.ones(shape=query_tokens.shape[:-1], dtype='int64')
        attention_mask = paddle.concat(
            x=[query_atts, text_tokens.attention_mask], axis=1)
        #import pdb;pdb.set_trace()
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels)
        loss_lm = lm_output.loss
        return Blip2ForStage1ModelOutput(
            loss=loss_itc + loss_itm + loss_lm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm)

    @paddle.no_grad()
    def generate_stage1(self,
                        samples,
                        use_nucleus_sampling=False,
                        num_beams=3,
                        max_length=30,
                        min_length=10,
                        top_p=0.9,
                        repetition_penalty=1.0):
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
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples['image']
        image_embeds = self.ln_vision(self.visual_encoder(image))
        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, axis=0)
        else:
            num_beams = 1
        image_atts = paddle.ones(shape=image_embeds.shape[:-1], dtype='int64')
        model_kwargs = {
            'encoder_hidden_states': image_embeds,
            'encoder_attention_mask': image_atts
        }
        input_ids = paddle.empty(
            shape=[image.shape[0], 1],
            dtype='int64').fill_(value=self.tokenizer.bos_token_id)
        query_tokens = self.query_tokens.expand(
            shape=[image_embeds.shape[0], -1, -1])
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
            **model_kwargs)
        captions = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        return captions

    @paddle.no_grad()
    def generate(
            self,
            pixel_values: paddle.Tensor,
            input_ids: Optional[paddle.Tensor]=None,
            attention_mask: Optional[paddle.Tensor]=None,
            **generate_kwargs, ) -> paddle.Tensor:
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
            captions (list): A list of strings of length batch_size * num_captions.
        """
        batch_size = pixel_values.shape[0]
        image_embeds = self.Qformer.ln_vision(self.visual_encoder(pixel_values))
        image_attention_mask = paddle.ones(
            image_embeds.shape[:-1], dtype="int64")

        query_tokens = self.Qformer.query_tokens.expand(
            [image_embeds.shape[0], -1, -1])
        query_outputs = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True, )
        query_output = query_outputs.last_hidden_state

        language_model_inputs = self.Qformer.language_projection(query_output)
        language_attention_mask = paddle.ones(
            language_model_inputs.shape[:-1], dtype="int64")
        if input_ids is None:
            input_ids = paddle.to_tensor(
                [[self.config.text_config.bos_token_id]]).tile([batch_size, 1])
        if attention_mask is None:
            attention_mask = paddle.ones_like(input_ids)
        attention_mask = paddle.concat(
            [language_attention_mask, attention_mask], axis=1)
        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = paddle.concat(
            [language_model_inputs, inputs_embeds], axis=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=False,
            top_p=0.9,
            decode_strategy="greedy_search",  # align to torch
            temperature=1,
            num_beams=5,
            max_length=30,
            min_length=8,
            eos_token_id=50118,
            repetition_penalty=1,
            length_penalty=1,
            num_return_sequences=1, )

        return outputs

    @paddle.no_grad()
    def encode_image(
            self,
            pixel_values: paddle.Tensor,
            **kwargs, ):
        image_embeds = self.ln_vision(
            self.visual_encoder(pixel_values.astype("float16")))
        image_embeds = image_embeds.astype("float32")

        image_attention_mask = paddle.ones(
            image_embeds.shape[:-1], dtype="int64")

        query_tokens = self.query_tokens.expand([image_embeds.shape[0], -1, -1])
        query_outputs = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True, )
        query_output = query_outputs[0]
        return query_output


from contextlib import contextmanager


@contextmanager
def dtype_guard(dtype="float32"):
    origin_dtype = paddle.get_default_dtype()
    paddle.set_default_dtype(dtype)
    try:
        yield
    finally:
        paddle.set_default_dtype(origin_dtype)

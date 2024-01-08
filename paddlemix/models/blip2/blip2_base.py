"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib

# import datetime
import logging
from functools import partial

import paddle

# import paddle.distributed as dist
import paddle.nn as nn

# from .clip_vit import create_clip_vit_L
# from paddle.amp import autocast as autocast
from paddlenlp.transformers import BertTokenizer
from paddlenlp.transformers.bert.configuration import BertConfig

# import paddlemix
from paddlemix.models.blip2.modeling import Blip2PretrainedModel

from .configuration import Blip2VisionConfig
from .eva_vit import VisionTransformer, convert_weights_to_fp16
from .Qformer import BertLMHeadModel


class Blip2Base(Blip2PretrainedModel):
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side, return_attention_mask=True)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=paddle.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use paddle.float16
        enable_autocast = paddle.device.get_device() != "cpu"
        enable_autocast = False

        if enable_autocast:
            return paddle.amp.auto_cast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        # Add ignore_mismatched_sizes:
        #   RuntimeError: Error(s) in loading state_dict for BertLMHeadModel:
        #   Skip loading for cls.predictions.bias. cls.predictions.bias receives a shape [30522], but the expected shape is [30523].
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config, ignore_mismatched_sizes=True
        )
        tmp = paddle.zeros([1, num_query_token, encoder_config.hidden_size])
        query_tokens = paddle.create_parameter(
            shape=tmp.shape, dtype=tmp.dtype, default_initializer=paddle.nn.initializer.Assign(tmp)
        )
        # query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        # normal_(query_tokens, mean=0.0, std=encoder_config.initializer_range)
        normal_ = paddle.nn.initializer.Normal(mean=0.0, std=encoder_config.initializer_range)
        normal_(query_tokens)
        return Qformer, query_tokens

    def init_vision_encoder(self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision):
        print(model_name)
        assert model_name in [
            "eva_clip_g",
            "eva2_clip_L",
            "clip_L",
        ], "vit model must be eva_clip_g, eva2_clip_L or clip_L"
        print('jjjjjjj')
        if model_name == "eva_clip_g":
            visual_encoder = self.create_eva_vit_g(img_size, drop_path_rate, use_grad_checkpoint, precision)

        ln_vision = LayerNorm(visual_encoder.num_features)
        self.vit_name = model_name
        return visual_encoder, ln_vision

    def create_eva_vit_g(self, img_size=224, drop_path_rate=0.4, use_checkpoint=False, precision="fp16"):
        vision_config = Blip2VisionConfig(
            img_size=img_size,
            patch_size=14,
            use_mean_pooling=False,
            embed_dim=1408,
            depth=39,
            num_heads=1408 // 88,
            mlp_ratio=4.3637,
            qkv_bias=True,
            drop_path_rate=drop_path_rate,
            norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
            use_checkpoint=use_checkpoint,
        )
        model = VisionTransformer(vision_config)

        if precision == "fp16":
            convert_weights_to_fp16(model)
        return model

    def get_optimizer_params(self, weight_decay, lr_scale=1):

        vit_num_layers = self.visual_encoder.get_num_layer()
        lr_scales = list(lr_scale ** (vit_num_layers + 1 - i) for i in range(vit_num_layers + 2))

        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.named_parameters():
            if param.stop_gradient:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.0
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            if "visual_encoder" in name:
                layer_id = self.visual_encoder.get_num_layer(name.replace("visual_encoder.", ""))
                group_name = "vit_layer_%d_%s" % (layer_id, group_name)
            else:
                layer_id = None

            if group_name not in parameter_group_names:
                if layer_id is not None:
                    scale = lr_scales[layer_id]
                else:
                    scale = 1
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale,
                }
                parameter_group_vars[group_name] = {"weight_decay": this_weight_decay, "params": [], "lr_scale": scale}
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
        # import json
        # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        optim_params = list(parameter_group_vars.values())
        return optim_params

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: paddle.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.astype(paddle.float32))
        return ret.astype(orig_type)

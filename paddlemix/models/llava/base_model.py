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

from .clip_encoder import build_vision_tower
from .constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from .mm_projector import build_vision_projector

__all__ = ["LlavaMetaModel", "LlavaMetaForCausalLM"]


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.mm_vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower

            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            for p in self.mm_projector.parameters():
                p.stop_gradient = not True
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = paddle.load(path=pretrain_mm_mlp_adapter)

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.set_state_dict(state_dict=get_w(mm_projector_weights, "mm_projector"))


class LlavaMetaForCausalLM:
    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):

                target_shape = past_key_values[-1][-1].shape[1] + 1
                attention_mask = paddle.concat(
                    x=(
                        attention_mask,
                        paddle.ones(
                            shape=(attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                            dtype=attention_mask.dtype,
                        ),
                    ),
                    axis=1,
                )
                position_ids = paddle.sum(x=attention_mask, axis=1, dtype="int64").unsqueeze(axis=-1) - 1

            return (input_ids, position_ids, attention_mask, past_key_values, None, labels)
        if type(images) is list or images.ndim == 5:
            concat_images = paddle.concat(x=[image for image in images], axis=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = paddle.split(x=image_features, num_or_sections=split_sizes, axis=0)
            image_features = [x.flatten(start_axis=0, stop_axis=1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = paddle.ones_like(x=input_ids, dtype="bool")
        else:
            attention_mask = attention_mask.astype(dtype="bool")
        if position_ids is None:
            position_ids = paddle.arange(start=0, end=input_ids.shape[1], dtype="int64")
        if labels is None:
            labels = paddle.full_like(x=input_ids, fill_value=IGNORE_INDEX)

        input_ids = [
            cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = paddle.concat(x=[cur_input_embeds_1, cur_image_features[0:0]], axis=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = (
                [-1]
                + paddle.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].squeeze(axis=1).tolist()
                + [cur_input_ids.shape[0]]
            )

            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(paddle.concat(x=cur_input_ids_noim))
            cur_input_embeds_no_im = paddle.split(x=cur_input_embeds, num_or_sections=split_sizes, axis=0)

            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        paddle.full(
                            shape=(cur_image_features.shape[0],), fill_value=IGNORE_INDEX, dtype=cur_labels.dtype
                        )
                    )

            cur_new_input_embeds = paddle.concat(x=cur_new_input_embeds)
            cur_new_labels = paddle.concat(x=cur_new_labels)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)

        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        new_input_embeds_padded = []
        new_labels_padded = paddle.full(
            shape=(batch_size, max_len), fill_value=IGNORE_INDEX, dtype=new_labels[0].dtype
        )
        attention_mask = paddle.zeros(shape=(batch_size, max_len), dtype=attention_mask.dtype)
        position_ids = paddle.zeros(shape=(batch_size, max_len), dtype=position_ids.dtype)
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
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
                    new_labels_padded[(i), -cur_len:] = cur_new_labels
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
                    new_labels_padded[(i), :cur_len] = cur_new_labels
                    attention_mask[(i), :cur_len] = True
                    position_ids[(i), :cur_len] = paddle.arange(start=0, end=cur_len, dtype=position_ids.dtype)
        new_input_embeds = paddle.stack(x=new_input_embeds_padded, axis=0)
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded
        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = paddle.cast(attention_mask, dtype=_attention_mask.dtype)
        if _position_ids is None:
            position_ids = None
        return (None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels)

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data
                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(axis=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(axis=0, keepdim=True)
                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.stop_gradient = not True
                for p in self.get_output_embeddings().parameters():
                    p.stop_gradient = not False
            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = paddle.load(path=model_args.pretrain_mm_mlp_adapter)
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.stop_gradient = not False
                for p in self.get_output_embeddings().parameters():
                    p.stop_gradient = not False

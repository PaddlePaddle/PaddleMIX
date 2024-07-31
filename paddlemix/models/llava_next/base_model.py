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
from .mm_utils import get_anyres_image_grid_shape

__all__ = ["LlavaMetaModel", "LlavaMetaForCausalLM"]


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = paddle.create_parameter(
                    shape=paddle.empty(shape=[config.hidden_size], dtype=self._dtype).shape,
                    dtype=self._dtype,
                    default_initializer=paddle.nn.initializer.Assign(
                        paddle.empty(shape=[config.hidden_size], dtype=self._dtype)
                    ),
                )
                self.image_newline.stop_gradient = False

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
        mm_patch_merge_type = model_args.mm_patch_merge_type
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
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            for p in self.mm_projector.parameters():
                p.stop_gradient = not True
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = paddle.load(path=pretrain_mm_mlp_adapter)
            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / paddle.sqrt(x=paddle.to_tensor(data=[self.config.hidden_size], dtype=self.dtype))
                self.image_newline = paddle.create_parameter(
                    shape=(paddle.randn(shape=[self.config.hidden_size], dtype=self.dtype) * embed_std).shape,
                    dtype=self.dtype,
                    default_initializer=paddle.nn.initializer.Assign(
                        paddle.randn(shape=[self.config.hidden_size], dtype=self.dtype) * embed_std
                    ),
                )
                self.image_newline.stop_gradient = False

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.set_state_dict(state_dict=get_w(mm_projector_weights, "mm_projector"))


def unpad_image(tensor, original_size):
    """
    Unpads a Paddle tensor of a padded and resized image.
    Args:
    tensor (paddle.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).
    Returns:
    paddle.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height
    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]
    return unpadded_tensor


class LlavaMetaForCausalLM:
    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes=None,
    ):
        vision_tower = self.get_vision_tower()

        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return (input_ids, position_ids, attention_mask, past_key_values, None, labels)
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [(x.unsqueeze(axis=0) if x.ndim == 3 else x) for x in images]
            concat_images = paddle.concat(x=[image for image in images], axis=0)
            image_features = self.encode_images(concat_images)

            split_sizes = [image.shape[0] for image in images]
            image_features = paddle.split(image_features, split_sizes, axis=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(start_axis=0, stop_axis=1) for x in image_features]
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == "anyres":
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                image_sizes[image_idx],
                                self.config.image_grid_pinpoints,
                                self.get_vision_tower().config.image_size,
                            )

                            image_feature = paddle.reshape(
                                image_feature, (num_patch_height, num_patch_width, height, width, -1)
                            )
                        else:
                            raise NotImplementedError
                        if "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.transpose(perm=[4, 0, 2, 1, 3])
                            image_feature = image_feature.flatten(start_axis=1, stop_axis=2).flatten(
                                start_axis=2, stop_axis=3
                            )
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = paddle.concat(
                                x=(
                                    image_feature,
                                    self.llama.image_newline[:, (None), (None)].expand(
                                        shape=[*image_feature.shape[:-1], 1]
                                    ).astype(image_feature.dtype),
                                ),
                                axis=-1,
                            )
                            x = image_feature.flatten(start_axis=1, stop_axis=2)
                            perm_12 = list(range(x.ndim))
                            perm_12[0] = 1
                            perm_12[1] = 0
                            image_feature = x.transpose(perm=perm_12)
                        else:
                            image_feature = image_feature.transpose(perm=[0, 2, 1, 3, 4])
                            image_feature = image_feature.flatten(start_axis=0, stop_axis=3)
                        image_feature = paddle.concat(x=(base_image_feature, image_feature), axis=0)
                    else:
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = paddle.concat(
                                x=(image_feature, self.llama.image_newline[None].to(image_feature.place)), axis=0
                            )
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
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

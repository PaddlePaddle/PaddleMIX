# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import gc
from typing import List, Optional

import paddle
from einops import rearrange, repeat
from paddlenlp.transformers import PretrainedModel

from paddlemix.models.janus.siglip_vit import SigLIPVisionTransformer
from .configuration_deepseek import DeepseekVLV2Config
from .modeling_deepseek import DeepseekV2ForCausalLM
# from paddlenlp.transformers.deepseek_v2.modeling import DeepseekV2ForCausalLM # diff


class DeepseekVLMlpProjector(paddle.nn.Layer):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.projector_type == "identity":
            modules = paddle.nn.Identity()
        elif cfg.projector_type == "linear":
            modules = paddle.nn.Linear(in_features=cfg.input_dim, out_features=cfg.n_embed)
        elif cfg.projector_type == "mlp_gelu":
            mlp_depth = cfg.depth
            modules = [paddle.nn.Linear(in_features=cfg.input_dim, out_features=cfg.n_embed)]
            for _ in range(1, mlp_depth):
                modules.extend([paddle.nn.GELU(), paddle.nn.Linear(in_features=cfg.n_embed, out_features=cfg.n_embed)])
            modules = paddle.nn.Sequential(*modules)
        elif cfg.projector_type == "downsample_mlp_gelu":
            mlp_depth = cfg.depth
            mlp_ratio = cfg.mlp_ratio
            modules = [
                paddle.nn.Linear(
                    in_features=cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio,
                    out_features=cfg.n_embed * mlp_ratio,
                )
            ]
            for _ in range(1, mlp_depth - 1):
                modules.extend(
                    [
                        paddle.nn.GELU(),
                        paddle.nn.Linear(in_features=cfg.n_embed * mlp_ratio, out_features=cfg.n_embed * mlp_ratio),
                    ]
                )
            modules.extend(
                [paddle.nn.GELU(), paddle.nn.Linear(in_features=cfg.n_embed * mlp_ratio, out_features=cfg.n_embed)]
            )
            modules = paddle.nn.Sequential(*modules)
        else:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")
        if cfg.token_pooling:
            self.token_pooling_layer = paddle.nn.Linear(in_features=cfg.input_dim * 4, out_features=cfg.input_dim)
        self.layers = modules

    def forward(self, x):
        if self.cfg.token_pooling:
            batch_size, wxh, channels = tuple(x.shape)
            w = h = int(wxh**0.5)
            x = x.reshape([batch_size, w, h, channels])
            x = x.transpose(perm=[0, 3, 1, 2])
            patches = x.unfold(axis=2, size=2, step=2).unfold(axis=3, size=2, step=2)
            batch_size, channels, h_patches, w_patches, _, _ = tuple(patches.shape)
            patches = patches.reshape([batch_size, channels, h_patches * w_patches, -1])
            patches = patches.transpose(perm=[0, 2, 1, 3])
            patches = patches.reshape([batch_size, h_patches * w_patches, channels * 4])
            x = self.token_pooling_layer(patches)

        elif self.cfg.projector_type == "downsample_mlp_gelu":
            bs, hw, input_dim = tuple(x.shape)
            h = w = int(hw**0.5)
            """compute padding"""
            if h % self.cfg.downsample_ratio:
                pad = self.cfg.downsample_ratio - h % self.cfg.downsample_ratio
            else:
                pad = 0
            x = x.reshape([bs, h, w, input_dim])
            if pad > 0:
                x = paddle.nn.functional.pad(
                    x=x, pad=(0, 0, 0, pad, 0, pad), mode="constant", value=0, pad_from_left_axis=False
                )
            """4 to 1 concat"""
            x = x.transpose(perm=[0, 3, 1, 2])
            x = paddle.nn.functional.unfold(
                x=x, kernel_sizes=self.cfg.downsample_ratio, strides=self.cfg.downsample_ratio, paddings=0
            )
            x = x.transpose(perm=[0, 2, 1])
        return self.layers(x)


class DeepseekVLV2PreTrainedModel(PretrainedModel):
    config_class = DeepseekVLV2Config
    base_model_prefix = "deepseek_vl_v2"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class DeepseekVLV2ForCausalLM(DeepseekVLV2PreTrainedModel):
    def __init__(self, config: DeepseekVLV2Config):
        super().__init__(config)
        vision_config = config.vision_config
        self.vision = SigLIPVisionTransformer(
            img_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
            embed_dim=vision_config.width,
            depth=vision_config.layers,
            num_heads=vision_config.heads,
            mlp_ratio=vision_config.mlp_ratio,
            class_token=vision_config.class_token,
            global_pool=vision_config.global_pool,
            ignore_head=vision_config.ignore_head,
            weight_init=vision_config.weight_init,
            num_classes=0,
            # approximate=True,
        )
        # delete params deterministic and num_recomputing_layers
        projector_config = config.projector_config
        self.projector = DeepseekVLMlpProjector(projector_config)
        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        embed_std = 1 / paddle.sqrt(x=paddle.to_tensor(data=projector_config.n_embed, dtype="float32"))

        if self.tile_tag == "2D":
            self.image_newline = paddle.create_parameter(
                shape=[projector_config.n_embed],
                dtype=paddle.get_default_dtype(),
                default_initializer=paddle.nn.initializer.Normal(std=embed_std),
            )
            self.view_seperator = paddle.create_parameter(
                shape=[projector_config.n_embed],
                dtype=paddle.get_default_dtype(),
                default_initializer=paddle.nn.initializer.Normal(std=embed_std),
            )
            self.image_newline.stop_gradient = True
            self.view_seperator.stop_gradient = True
        elif self.tile_tag == "1D":
            candidate_resolutions = config.candidate_resolutions
            if len(candidate_resolutions) == 0:
                raise ValueError(
                    f"len(candidate_resolutions) should be larger than 0, but got {len(candidate_resolutions)}"
                )
            tile_variants_num = len(candidate_resolutions)
            self.tile_indicators = paddle.create_parameter(
                shape=(tile_variants_num + 1, config.aligner.params.n_embed),
                dtype=paddle.get_default_dtype(),
                default_initializer=paddle.nn.initializer.Normal(std=embed_std),
            )
        else:
            raise ValueError(f"tile tag should be either 1D or 2D, but got {self.tile_tag}")
        language_config = config.language_config
        self.language = DeepseekV2ForCausalLM(language_config)

    def prepare_inputs_embeds(
        self,
        input_ids: paddle.Tensor,
        images: Optional[paddle.Tensor] = None,
        images_seq_mask: Optional[paddle.Tensor] = None,
        images_spatial_crop: Optional[paddle.Tensor] = None,
        **ignore_kwargs
    ):
        """

        Args:
            input_ids (paddle.Tensor): [b, T]
            images (paddle.Tensor): [b, max_n_images, 3, height, width]
            images_seq_mask (paddle.Tensor): [b, T]
            images_spatial_crop (paddle.Tensor): [b, max_n_images, 2]

        Returns:
            input_embeds (paddle.Tensor): [b, T, D]
        """
        if images is None or images_spatial_crop.sum() == 0:
            return self.language.get_input_embeddings()(input_ids)

        bs, max_n_images, _ = tuple(images_spatial_crop.shape)
        batch_num_tiles = [(0) for _ in range(bs)]
        total_tiles = []
        for idx in range(bs):
            for jdx in range(max_n_images):
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break
                batch_num_tiles[idx] += 1 + num_width_tiles * num_height_tiles
            total_tiles.append(images[idx, : batch_num_tiles[idx]])
        total_tiles = paddle.concat(x=total_tiles, axis=0)

        assert tuple(total_tiles.shape)[0] == sum(batch_num_tiles)
        if tuple(total_tiles.shape)[0] == 0:
            return self.language.get_input_embeddings()(input_ids)
        images_feature = self.vision(total_tiles)
        images_embeds = self.projector(images_feature)
        _, hw, n_dim = tuple(images_embeds.shape)
        h = w = int(hw**0.5)

        input_embeds = self.language.get_input_embeddings()(input_ids)
        tile_index = 0
        for idx in range(tuple(images_spatial_crop.shape)[0]):
            images_in_this_batch = []
            for jdx in range(tuple(images_spatial_crop.shape)[1]):
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break
                num_tiles_in_image = num_width_tiles * num_height_tiles
                global_features = images_embeds[tile_index]
                local_features = images_embeds[tile_index + 1 : tile_index + 1 + num_tiles_in_image]
                tile_index += num_tiles_in_image + 1

                if self.tile_tag == "2D":
                    global_features = global_features.reshape([h, w, n_dim])
                    new_lines_in_global = repeat(self.image_newline, "d -> h 1 d", h=h)
                    global_features = paddle.concat(x=[global_features, new_lines_in_global], axis=1)
                    global_features = global_features.reshape([-1, n_dim])
                    local_features = rearrange(
                        local_features,
                        "(th tw) (h w) d -> (th h) (tw w) d",
                        th=num_height_tiles,
                        tw=num_width_tiles,
                        h=h,
                        w=w,
                    )

                    new_lines_in_local = repeat(self.image_newline, "d -> (th h) 1 d", th=num_height_tiles, h=h)

                    local_features = paddle.concat(x=[local_features, new_lines_in_local], axis=1)
                    local_features = local_features.reshape([-1, n_dim])
                    if self.global_view_pos == "head":
                        global_local_features = paddle.concat(
                            x=[global_features, self.view_seperator[None, :], local_features], axis=0
                        )
                    else:
                        global_local_features = paddle.concat(
                            x=[local_features, self.view_seperator[None, :], global_features], axis=0
                        )
                else:
                    global_features = paddle.concat(x=[self.tile_indicators[0:1], global_features], axis=0)
                    local_features = paddle.concat(
                        x=[self.tile_indicators[1 : num_tiles_in_image + 1].unsqueeze(axis=1), local_features], axis=1
                    )
                    local_features = rearrange(local_features, "crop_num hw d -> (crop_num hw) d")

                    if self.global_view_pos == "head":
                        global_local_features = paddle.concat(x=[global_features, local_features], axis=0)
                    else:
                        global_local_features = paddle.concat(x=[local_features, global_features], axis=0)
                images_in_this_batch.append(global_local_features)
            if len(images_in_this_batch) > 0:
                images_in_this_batch = paddle.concat(x=images_in_this_batch, axis=0)
                input_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(axis=-1), images_in_this_batch)
        return input_embeds

    @paddle.no_grad()
    def incremental_prefilling(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        images: Optional[paddle.Tensor] = None,
        images_seq_mask: Optional[paddle.Tensor] = None,
        images_spatial_crop: Optional[paddle.Tensor] = None,
        chunk_size: int = 1024,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_embeds(
                input_ids=input_ids,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
            )
            del images
            del images_seq_mask
            del images_spatial_crop
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.place)
            self._clear_cuda_cache()
        bzs, seq_len, _ = tuple(inputs_embeds.shape)
        past_key_values = None
        prefilling_len = seq_len - 1
        for i in range(0, prefilling_len, chunk_size):
            chunk_start = i
            chunk_end = min(i + chunk_size, prefilling_len)
            chunk_inputs_embeds = inputs_embeds[:, chunk_start:chunk_end]
            chunk_attention_mask = attention_mask[:, 0:chunk_end]

            if past_key_values is not None:
                position_ids = paddle.arange(start=chunk_start, end=chunk_end, dtype="int64").unsqueeze(axis=0)

                past_key_values = self._move_past_key_values_to_gpu(past_key_values, inputs_embeds.place)
            else:
                position_ids = None

            with paddle.no_grad():
                outputs = self.forward(
                    inputs_embeds=chunk_inputs_embeds,
                    attention_mask=chunk_attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    use_cache=True,
                )
                # past_key_values = outputs.past_key_values
                past_key_values = outputs[1]
                past_key_values = self._move_past_key_values_to_cpu(past_key_values)
                del outputs, position_ids
                self._clear_cuda_cache()

        prefilling_key_values = []
        for layer_past in past_key_values:
            prefilling_key_values.append(
                (
                    layer_past[0][:, :, 0:prefilling_len, ...].to(inputs_embeds.place),
                    layer_past[1][:, :, 0:prefilling_len, ...].to(inputs_embeds.place),
                )
            )
        return inputs_embeds, prefilling_key_values

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        images: Optional[paddle.Tensor] = None,
        images_seq_mask: Optional[paddle.Tensor] = None,
        images_spatial_crop: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # cache_position: Optional[paddle.Tensor] = None,
    ):
        # print('input_ids', input_ids.shape, input_ids.sum().item()) # [1, 472] 54625351 # [1, 780] 62258750
        # print('attention_mask', attention_mask) # [1, 472]
        # print('position_ids', position_ids) # None
        # # print('inputs_embeds', inputs_embeds.shape, inputs_embeds.sum().item())
        # print('images', images.shape, images.sum().item()) # [1, 3, 3, 384, 384] 348481.75 ？？
        # print('images_seq_mask', images_seq_mask) # [1, 780]
        # print('images_spatial_crop', images_spatial_crop) # shape=[1, 1, 2]     [[[2, 1]]]
        # print('labels', labels) # [1, 780], sum 195540
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # print('forward input_ids', input_ids.shape, input_ids.sum())
            inputs_embeds = self.prepare_inputs_embeds(
                input_ids=input_ids,  # [1, 1509]
                images=images,  # [1, 7, 3, 384, 384]
                images_seq_mask=images_seq_mask,  # [1, 1509]
                images_spatial_crop=images_spatial_crop,  # [1, 1, 2]
            )

        outputs = self.language.forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs

    def _clear_cuda_cache(self):
        """clear CUDA memory cache"""
        gc.collect()
        if paddle.device.cuda.device_count() >= 1:
            paddle.device.cuda.empty_cache()
            # paddle.device.cuda.synchronize()

    def _move_past_key_values_to_cpu(self, past_key_values):
        if past_key_values is None:
            return None
        return tuple(tuple(t.cpu() for t in layer) for layer in past_key_values)

    def _move_past_key_values_to_gpu(self, past_key_values, device="gpu:0"):
        if past_key_values is None:
            return None
        return tuple(tuple(t.to(device) for t in layer) for layer in past_key_values)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        images: Optional[paddle.Tensor] = None,
        images_seq_mask: Optional[paddle.Tensor] = None,
        images_spatial_crop: Optional[paddle.Tensor] = None,
        attention_mask=None,
        pixel_values=None,
        image_sizes=None,
        num_logits_to_keep=None,
        **kwargs
    ):
        # input_ids is the full sequence for the first token generation
        generated_input_ids = (
            kwargs["full_input_ids"] if kwargs["full_input_ids"].shape[1] == attention_mask.shape[1] else input_ids
        )
        model_inputs = self.language.prepare_inputs_for_generation(
            generated_input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,  # already prepared input_embeds
            attention_mask=attention_mask,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )
        model_inputs["images"] = images
        model_inputs["images_seq_mask"] = images_seq_mask
        model_inputs["images_spatial_crop"] = images_spatial_crop
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += tuple(
                past_state.index_select(axis=0, index=beam_idx.to(past_state.place)) for past_state in layer_past
            )
        return reordered_past

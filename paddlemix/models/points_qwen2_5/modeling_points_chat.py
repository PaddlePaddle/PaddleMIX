# -*- coding: utf-8 -*-

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

# @Time    : 2025/4/19 下午8:37
# @Author  : zhaop-l(zhaopuzxjc@126.com)

from typing import List, Tuple

import paddle
from paddle import Tensor, nn
from paddlenlp.transformers import CLIPVisionModel
from paddlenlp.transformers.model_utils import GenerationMixin, PretrainedModel
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from PIL import Image

from .catty import split_image_with_catty
from .configuration_points_chat import POINTSChatConfig
from .dynamic_high_resolution import split_image
from .modeling_llama import CustomLlamaForCausalLM


class POINTSChatModel(PretrainedModel, GenerationMixin):
    config_class = POINTSChatConfig
    _no_split_modules = ["CLIPVisionModel", "LLamaDecoderLayer"]
    """Chat model for POINTS.
    Official implementation of the paper "POINTS: Improving Your Vision-language Model with Affordable Strategies"  # noqa: E501
    paper: https://huggingface.co/papers/2409.04828

    Args:
        config (PretrainedConfig): The model config.
    """

    def __init__(self, config: POINTSChatConfig) -> None:
        super().__init__(config)
        self.general_vit = CLIPVisionModel(config.vision_config)
        self.ocr_vit = CLIPVisionModel(config.vision_config)
        self.llm = CustomLlamaForCausalLM(config.llm_config)
        self.vision_projector = nn.Sequential(
            nn.Linear(
                in_features=config.vision_config.hidden_size * 4,
                out_features=config.llm_config.hidden_size,
            ),
            nn.GELU(),
            nn.Linear(
                in_features=config.llm_config.hidden_size,
                out_features=config.llm_config.hidden_size,
            ),
        )

    def apply_chat_template(self, prompt: str, image_num: int) -> str:
        """Apply the Yi-1.5-Chat template to the prompt.

        Args:
            prompt (str): The prompt to apply the template to.
            image_num (int): The number of the image in the prompt.
        Returns:
            str: The prompt with the template applied.
        """
        image_tokens = "<|image_pad|>" * 144 * image_num
        prompt = f"<|im_start|>user\n{image_tokens}{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    def pixel_shuffle(self, feature_map: Tensor, scale_factor: float = 0.5) -> Tensor:
        """Implementation of pixel shuffle.

        Merge several patches into a single patch by concatenating
        them across the channel dimension. Therefore, we can reduce
        the image sequence length. In POINTS, we merge 2x2 adjacent
        patches into a single patch.

        Args:
            feature_map (torch.Tensor): The feature map to be pixel
                shuffled.
            scale_factor (float, optional): The scale factor for the
        """
        # taken from https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5/blob/main/modeling_internvl_chat.py#L187 # noqa
        n, w, h, c = tuple(feature_map.shape)
        # N, W, H, C --> N, W, H * scale, C // scale
        feature_map = feature_map.reshape((n, w, int(h * scale_factor), int(c / scale_factor)))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        feature_map = feature_map.transpose(perm=[0, 2, 1, 3]).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        feature_map = feature_map.reshape(
            (
                n,
                int(h * scale_factor),
                int(w * scale_factor),
                int(c / (scale_factor * scale_factor)),
            )
        )
        feature_map = feature_map.transpose(perm=[0, 2, 1, 3]).contiguous()
        return feature_map

    def extract_image_features(self, images: Tensor, vision_encoder: str = "general_vit") -> Tensor:
        """Extract the image features from the vision encoder.

        Args:
            images (torch.Tensor): The images to extract the features from.
            vision_encoder (str, optional): The vision encoder to use.
                Defaults to 'general_vit'.

        Returns:
            torch.Tensor: The extracted image features.
        """
        if vision_encoder == "general_vit":
            image_features = self.general_vit(images, output_hidden_states=True)
        else:
            image_features = self.ocr_vit(images, output_hidden_states=True)
        image_features = image_features.hidden_states[-2]
        image_features = image_features[:, 1:]
        image_features = image_features.reshape((-1, 24, 24, 1024))
        image_features = self.pixel_shuffle(image_features, 0.5)
        image_features = image_features.reshape((-1, 144, 4096))
        image_features = self.vision_projector(image_features)
        return image_features

    def get_pos_mapping(self, pos: List[list]) -> Tuple[dict, int]:
        """Get the position mapping for the images.

        Args:
            pos (List[list]): The position of the images in the prompt.

        Returns:
            Tuple[dict, int]: The position mapping and the
            total number of images.
        """
        mapping = {}
        total_images = 0
        for i, (start, end) in enumerate(pos):
            num_image = int((end - start) / 144)
            mapping[i] = num_image
            total_images += num_image
        return mapping, total_images

    @paddle.no_grad()
    def chat(
        self,
        pixel_values: Image,
        prompt: str,
        tokenizer: PretrainedTokenizer,
        image_processor: object,
        catty: bool = True,
        generation_config: dict = None,
        max_splits: int = 8,
    ):
        """Generate a response to the input prompt.

        Args:
            pixel_values (Image): The input image.
            prompt (str): The input prompt.
            tokenizer (PreTrainedTokenizer): The tokenizer to use.
            image_processor (object): The image processor to use.
            catty (bool, optional): Whether to use catty. Defaults to True.
            generation_config (dict, optional): The generation config.
                Defaults to None.
            max_splits (int, optional): The maximum number of splits.
                Defaults to 8.
        Returns:
            str: The generated response.
        """
        if catty:
            cropped_images = split_image_with_catty(pixel_values, do_resize=True, max_crop_slices=max_splits)
        else:
            cropped_images = split_image(pixel_values, max_splits=max_splits)

        prompt = self.apply_chat_template(prompt, len(cropped_images))
        cropped_images = image_processor.preprocess(cropped_images, return_tensors="pd")["pixel_values"]
        # extract features with general_vit
        general_vit_features = self.extract_image_features(cropped_images, vision_encoder="general_vit")

        # extract features with ocr_vit
        ocr_vit_features = self.extract_image_features(cropped_images, vision_encoder="ocr_vit")
        image_features = 0.5 * general_vit_features + 0.5 * ocr_vit_features

        model_inputs = tokenizer(prompt, return_tensors="pd")
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        # stop token
        eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        # image token
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        generation_config.update({"eos_token_id": eos_token_id})
        outputs = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_features=[image_features],
            image_token_id=image_token_id,
            **generation_config,
        )
        response = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]
        return response

    def generate(
        self,
        input_ids,
        attention_mask,
        image_features,
        image_token_id,
        generation_config=None,
        output_hidden_states=None,
        return_dict=None,
        **generate_kwargs,
    ):
        input_embeddings = self.llm.lm.embed_in(input_ids)
        batch_size = tuple(input_ids.shape)[0]
        assert len(image_features) == batch_size
        for i in range(batch_size):
            special_pos = input_ids[i] == image_token_id
            pos = (special_pos[:-1] != special_pos[1:]).nonzero() + 1
            if tuple(pos.shape)[0] % 2 != 0:
                pos = paddle.concat(x=[paddle.to_tensor(data=[[0]]).to(pos.place), pos])
            pos = pos.reshape((-1, 2)).tolist()
            pos_mapping, total_images = self.get_pos_mapping(pos)
            assert total_images == len(image_features[i])
            img_offset = 0
            for j, (start, end) in enumerate(pos):
                num_images = pos_mapping[j]
                input_embeddings[i, start:end] = paddle.concat(
                    x=[image_features[i][img_offset + k] for k in range(num_images)],
                    axis=0,
                )
                img_offset += num_images

        outputs = self.llm.generate(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **generate_kwargs,
        )
        return outputs

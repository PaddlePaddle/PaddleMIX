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

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import paddle
from paddlenlp.transformers import DeepseekTokenizerFast
from PIL import Image, ImageOps

from ..models.deepseek_vl2.conversation import get_conv_template
from .base_processing import ProcessorMixin

__all__ = ["DeepseekVLV2Processor"]


def pad_sequence(sequences, padding_value=0, fix_len=None):
    """Pad a list of variable length paddle.Tensor with size L x *
    where L is length of the sequence and * is any number of dimensions (including 0)
    """
    max_size = sequences[0].shape
    trailing_dims = max_size[1:]
    max_len = max([s.shape[0] for s in sequences])
    if fix_len is not None:
        assert fix_len >= max_len, "fix_len is too small."
        max_len = fix_len
    out_dims = (len(sequences), max_len) + trailing_dims
    out_tensor = paddle.full(out_dims, padding_value, dtype=sequences[0].dtype)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        out_tensor[i, :length, ...] = tensor
    return out_tensor


def select_best_resolution(image_size, candidate_resolutions):
    original_width, original_height = image_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in candidate_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = width * height - effective_resolution

        if (
            effective_resolution > max_effective_resolution
            or effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = width, height

    return best_fit


class DictOutput(object):
    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


@dataclass
class DeepseekVLChatProcessorOutput(DictOutput):
    sft_format: str
    input_ids: paddle.int64
    target_ids: paddle.int64
    images: paddle.Tensor
    images_seq_mask: paddle.bool
    images_spatial_crop: paddle.int64
    num_image_tokens: List[int]

    def __len__(self):
        return len(self.input_ids)


@dataclass
class DeepseekBatchCollateOutput(DictOutput):
    sft_format: List[str]
    input_ids: paddle.int64
    labels: paddle.int64
    images: paddle.Tensor
    attention_mask: paddle.Tensor
    images_seq_mask: paddle.bool
    images_spatial_crop: paddle.int64
    seq_lens: List[int]

    def to(self, device, dtype="bfloat16"):
        self.input_ids = self.input_ids.to(device)
        self.labels = self.labels.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.images_seq_mask = self.images_seq_mask.to(device)
        self.images_spatial_crop = self.images_spatial_crop.to(device)
        self.images = self.images.to(device=device, dtype=dtype)
        return self


class ImageTransform(object):
    def __init__(
        self,
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True,
    ):
        self.mean = mean
        self.std = std
        self.normalize = normalize
        transform_pipelines = [paddle.vision.transforms.ToTensor()]

        if normalize:
            transform_pipelines.append(paddle.vision.transforms.Normalize(mean, std))
        self.transform = paddle.vision.transforms.Compose(transform_pipelines)

    def __call__(self, pil_img: Image.Image):
        x = self.transform(pil_img)
        return x


class DeepseekVLV2Processor(ProcessorMixin):
    tokenizer_class = "DeepseekTokenizerFast"
    attributes = ["tokenizer"]

    def __init__(
        self,
        tokenizer: DeepseekTokenizerFast,
        candidate_resolutions: Tuple[Tuple[int, int]],
        patch_size: int,
        downsample_ratio: int,
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize: bool = True,
        image_token: str = "<image>",
        pad_token: str = "<｜▁pad▁｜>",
        add_special_token: bool = False,
        sft_format: str = "deepseek",
        mask_prompt: bool = True,
        ignore_id: int = -100,
        **kwargs
    ):
        self.candidate_resolutions = candidate_resolutions
        self.image_size = candidate_resolutions[0][0]
        self.patch_size = patch_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.normalize = normalize
        self.downsample_ratio = downsample_ratio
        self.image_transform = ImageTransform(mean=image_mean, std=image_std, normalize=normalize)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"

        if tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": pad_token})
        print(
            f"""Add pad token = ['{pad_token}'] to the tokenizer{pad_token}:{tokenizer.encode(pad_token, add_special_tokens=False).input_ids[0]}"""
        )
        image_token_id = self.tokenizer.get_vocab()[image_token]

        if image_token_id is None:
            special_tokens = [image_token]
            special_tokens_dict = {"additional_special_tokens": special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
        self.image_token_id = self.tokenizer.get_vocab()[image_token]
        print(
            f"""Add image token = ['{image_token}'] to the tokenizer {image_token}:{tokenizer.encode(image_token, add_special_tokens=False).input_ids[0]}"""
        )

        special_tokens = ["<|ref|>", "<|/ref|>", "<|det|>", "<|/det|>", "<|grounding|>"]
        special_tokens_dict = {"additional_special_tokens": special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        print(
            f"""Add grounding-related tokens = {
                special_tokens
            } to the tokenizer with input_ids <|ref|>:{
                tokenizer.encode('<|ref|>', add_special_tokens=False).input_ids[0]
            } <|/ref|>:{
                tokenizer.encode('<|/ref|>', add_special_tokens=False).input_ids[0]
            } <|det|>:{
                tokenizer.encode('<|det|>', add_special_tokens=False).input_ids[0]
            } <|/det|>:{
                tokenizer.encode('<|/det|>', add_special_tokens=False,).input_ids[0]
            } <|grounding|>:{
                tokenizer.encode('<|grounding|>', add_special_tokens=False).input_ids[0]
            }
            """
        )
        special_tokens = ["<|User|>", "<|Assistant|>"]
        special_tokens_dict = {"additional_special_tokens": special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        print(
            f"""Add chat tokens = {special_tokens} to the tokenizer with input_ids
                <|User|>:{tokenizer.encode('<|User|>', add_special_tokens=False).input_ids[0]}
                <|Assistant|>:{tokenizer.encode('<|Assistant|>', add_special_tokens=False).input_ids[0]}
            """
        )

        self.image_token = image_token
        self.pad_token = pad_token
        self.add_special_token = add_special_token
        self.sft_format = sft_format
        self.mask_prompt = mask_prompt
        self.ignore_id = ignore_id
        super().__init__(tokenizer, **kwargs)

    def new_chat_template(self):
        conv = get_conv_template(self.sft_format)
        return conv

    def format_messages(
        self, conversations: List[Dict[str, str]], sft_format: str = "deepseek", system_prompt: str = ""
    ):
        """
        Applies the SFT template to conversation.

        Args:
            conversations (List[Dict]): A List of messages.
            sft_format (str, optional): The format of the SFT template to use. Defaults to "deepseek".
            system_prompt (str, optional): The system prompt to use in the SFT template. Defaults to "".

        Returns:
            sft_prompt (str): The formatted text.
        """
        conv = get_conv_template(sft_format)
        conv.set_system_message(system_prompt)
        for message in conversations:
            conv.append_message(message["role"], message["content"].strip())
        sft_prompt = conv.get_prompt().strip()
        return sft_prompt

    def format_messages_v2(self, messages, pil_images, systems=None):
        """play the role of format_messages_v2 and get_images_info in the last version"""
        tokenized_data = []
        masked_tokenized_data = []
        images_list = []
        images_seq_mask = []
        images_spatial_crop = []
        num_image_tokens = []
        image_index = 0
        conv = get_conv_template(self.sft_format)
        conv_system_message = conv.system_message
        for idx, message in enumerate(messages):
            if idx == 0:
                tokenized_data += [self.bos_id]
                masked_tokenized_data += [self.bos_id]
                images_seq_mask += [False]
                conv.system_message = conv_system_message
            else:
                conv.system_message = ""
            if message["role"] == conv.roles[0] or message["role"] == "user":
                conv.reset_message()
                conv.append_message(conv.roles[0], str(message["content"]).strip())
                conv.append_message(conv.roles[1], "")
                formatted_question = conv.get_prompt()
                (tokenized_str, images, seq_mask, spatial_crop, n_image_tokens) = self.tokenize_with_images(
                    formatted_question,
                    pil_images[image_index : image_index + formatted_question.count(self.image_token)],
                    bos=False,
                    eos=False,
                    cropping=len(pil_images) <= 2,
                )
                image_index += formatted_question.count(self.image_token)
                tokenized_data += tokenized_str
                if self.mask_prompt:
                    masked_tokenized_data += [self.ignore_id] * len(tokenized_str)
                else:
                    masked_tokenized_data += tokenized_str
                images_list += images
                images_seq_mask += seq_mask
                images_spatial_crop += spatial_crop
                num_image_tokens += n_image_tokens
            elif message["role"] == conv.roles[1] or message["role"] == "assistant":
                formatted_answer = message["content"].strip()
                assert (
                    formatted_answer.count(self.image_token) == 0
                ), f"there should be no {self.image_token} in the assistant's reply, but got {messages}"
                (tokenized_str, images, seq_mask, spatial_crop, n_image_tokens) = self.tokenize_with_images(
                    formatted_answer, [], bos=False, eos=True, cropping=len(pil_images) <= 2
                )
                tokenized_data += tokenized_str
                masked_tokenized_data += tokenized_str
                images_seq_mask += seq_mask
            elif message["role"] == "system" or message["role"] == "deepseekapi-sys":
                assert idx == 0, "system information should only exist in the beginning of the conversation"
                formatted_system = message["content"].strip()
                tokenized_str = self.encode(formatted_system, bos=False, eos=False)
                tokenized_data += tokenized_str
                if self.mask_prompt:
                    masked_tokenized_data += [self.ignore_id] * len(tokenized_str)
                else:
                    masked_tokenized_data += tokenized_str
                seq_mask = [False] * len(tokenized_str)
                images_seq_mask += seq_mask
            else:
                assert False, f"Unknown role: {message['role']}"
        assert len(tokenized_data) == len(
            images_seq_mask
        ), f"format_messages_v2: tokenized_str's length {len(tokenized_str)} is not equal to imags_seq_mask's length {len(images_seq_mask)}"
        assert len(images_spatial_crop) == len(num_image_tokens), "image number should be compatible"
        return (
            tokenized_data,
            masked_tokenized_data,
            images_list,
            images_seq_mask,
            images_spatial_crop,
            num_image_tokens,
        )

    def format_prompts(self, prompts: str, sft_format: str = "deepseek", system_prompt: str = ""):
        """
        Applies the SFT template to prompts.

        Args:
            prompts (str): the non-sft formatted prompt;
            sft_format (str, optional): The format of the SFT template to use. Defaults to "deepseek".
            system_prompt (str, optional): The system prompt to use in the SFT template. Defaults to "".

        Returns:
            sft_prompt (str): The formatted text.
        """
        conv = get_conv_template(sft_format)
        conv.set_system_message(system_prompt)
        conv.append_message(conv.roles[0], prompts.strip())
        conv.append_message(conv.roles[1], "")
        sft_prompt = conv.get_prompt().strip()
        return sft_prompt

    @property
    def bos_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_id(self):
        return self.tokenizer.pad_token_id

    def encode(self, text: str, bos: bool = True, eos: bool = False):
        t = self.tokenizer.encode(text, add_special_tokens=False).input_ids
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int], **kwargs) -> str:
        return self.tokenizer.decode(t, **kwargs)

    def process_one(
        self,
        prompt: str = None,
        conversations: List[Dict[str, str]] = None,
        images: List[Image.Image] = None,
        apply_sft_format: bool = False,
        inference_mode: bool = True,
        system_prompt: str = "",
        **kwargs
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            apply_sft_format (bool): if prompt is not None, then apply the SFT format to prompt;
                if conversations is not None, then it will always apply the SFT format to conversations;
            inference_mode (bool): if True, then remove the last eos token;
            system_prompt (str): the system prompt;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (paddle.Tensor): [N + image tokens]
                - target_ids (paddle.Tensor): [N + image tokens]
                - images (paddle.Tensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """
        assert prompt is None or conversations is None, "prompt and conversations cannot be used at the same time."
        if prompt is None:
            sft_format = self.format_messages(
                conversations=conversations, sft_format=self.sft_format, system_prompt=system_prompt
            )
            (
                tokenized_str,
                masked_tokenized_str,
                images_list,
                images_seq_mask,
                images_spatial_crop,
                num_image_tokens,
            ) = self.format_messages_v2(conversations, images)
        else:
            if apply_sft_format:
                sft_format = self.format_prompts(
                    prompts=prompt, sft_format=self.sft_format, system_prompt=system_prompt
                )
            else:
                sft_format = prompt

            (
                tokenized_str,
                images_list,
                images_seq_mask,
                images_spatial_crop,
                num_image_tokens,
            ) = self.tokenize_with_images(sft_format, images, bos=True, eos=True, cropping=len(images) <= 2)
            masked_tokenized_str = []
            for token_index in tokenized_str:
                if token_index != self.image_token_id:
                    masked_tokenized_str.append(token_index)
                else:
                    masked_tokenized_str.append(self.ignore_id)
        assert (
            len(tokenized_str) == len(images_seq_mask) == len(masked_tokenized_str)
        ), f"tokenized_str's length {len(tokenized_str)},\
                input_ids' length {len(masked_tokenized_str)},\
                imags_seq_mask's length {len(images_seq_mask)}, are not equal"

        input_ids = paddle.to_tensor(data=tokenized_str, dtype=paddle.int64)
        target_ids = paddle.to_tensor(data=masked_tokenized_str, dtype=paddle.int64)
        images_seq_mask = paddle.to_tensor(data=images_seq_mask, dtype=paddle.bool)
        target_ids[(input_ids < 0) | (input_ids == self.image_token_id)] = self.ignore_id

        input_ids[input_ids < 0] = self.pad_id
        if inference_mode:
            assert input_ids[-1] == self.eos_id
            input_ids = input_ids[:-1]
            target_ids = target_ids[:-1]
            images_seq_mask = images_seq_mask[:-1]
        if len(images_list) == 0:
            images = paddle.zeros(shape=(1, 3, self.image_size, self.image_size))
            images_spatial_crop = paddle.zeros(shape=(1, 2), dtype="int64")
        else:
            images = paddle.stack(x=images_list, axis=0)
            images_spatial_crop = paddle.to_tensor(data=images_spatial_crop, dtype="int64")
        prepare = DeepseekVLChatProcessorOutput(
            sft_format=sft_format,
            input_ids=input_ids,
            target_ids=target_ids,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            num_image_tokens=num_image_tokens,
        )
        return prepare

    def __call__(
        self,
        *,
        prompt: str = None,
        conversations: List[Dict[str, str]] = None,
        images: List[Image.Image] = None,
        apply_sft_format: bool = False,
        force_batchify: bool = True,
        inference_mode: bool = True,
        system_prompt: str = "",
        **kwargs
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            apply_sft_format (bool): if prompt is not None, then apply the SFT format to prompt;
                if conversations is not None, then it will always apply the SFT format to conversations;
            force_batchify (bool): force batchify the inputs;
            inference_mode (bool): if True, then remove the last eos token;
            system_prompt (str): the system prompt;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """
        prepare = self.process_one(
            prompt=prompt,
            conversations=conversations,
            images=images,
            apply_sft_format=apply_sft_format,
            inference_mode=inference_mode,
            system_prompt=system_prompt,
        )
        if force_batchify:
            prepare = self.batchify([prepare])
        return prepare

    def tokenize_with_images(
        self, conversation: str, images: List[Image.Image], bos: bool = True, eos: bool = True, cropping: bool = True
    ):
        """Tokenize text with <image> tags."""
        assert conversation.count(self.image_token) == len(images)
        text_splits = conversation.split(self.image_token)
        images_list, images_seq_mask, images_spatial_crop = [], [], []
        num_image_tokens = []
        tokenized_str = []
        for text_sep, image in zip(text_splits, images):
            """encode text_sep"""
            tokenized_sep = self.encode(text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)
            """select best resolution for anyres"""
            if cropping:
                best_width, best_height = select_best_resolution(image.size, self.candidate_resolutions)
            else:
                best_width, best_height = self.image_size, self.image_size

            """process the global view"""
            global_view = ImageOps.pad(
                image, (self.image_size, self.image_size), color=tuple(int(x * 255) for x in self.image_transform.mean)
            )
            images_list.append(self.image_transform(global_view))

            """process the local views"""
            local_view = ImageOps.pad(
                image, (best_width, best_height), color=tuple(int(x * 255) for x in self.image_transform.mean)
            )

            for i in range(0, best_height, self.image_size):
                for j in range(0, best_width, self.image_size):
                    images_list.append(
                        self.image_transform(local_view.crop((j, i, j + self.image_size, i + self.image_size)))
                    )

            """record height / width crop num"""
            num_width_tiles, num_height_tiles = (best_width // self.image_size, best_height // self.image_size)
            images_spatial_crop.append([num_width_tiles, num_height_tiles])

            """add image tokens"""
            h = w = math.ceil(self.image_size // self.patch_size / self.downsample_ratio)
            tokenized_image = [self.image_token_id] * h * (w + 1)
            tokenized_image += [self.image_token_id]
            tokenized_image += [self.image_token_id] * (num_height_tiles * h) * (num_width_tiles * w + 1)
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)
            num_image_tokens.append(len(tokenized_image))

        """process the last text split"""
        tokenized_sep = self.encode(text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        """add the bos and eos tokens"""
        if bos:
            tokenized_str = [self.bos_id] + tokenized_str
            images_seq_mask = [False] + images_seq_mask
        if eos:
            tokenized_str = tokenized_str + [self.eos_id]
            images_seq_mask = images_seq_mask + [False]
        assert len(tokenized_str) == len(
            images_seq_mask
        ), f"tokenize_with_images func: tokenized_str's length {len(tokenized_str)} is not equal to imags_seq_mask's length {len(images_seq_mask)}"
        return (tokenized_str, images_list, images_seq_mask, images_spatial_crop, num_image_tokens)

    def batchify(
        self, sample_list: List[DeepseekVLChatProcessorOutput], padding: Literal["left", "right"] = "left"
    ) -> DeepseekBatchCollateOutput:
        """
        Preprocesses the inputs for multimodal inference.

        Args:
            sample_list (List[VLChatProcessorOutput]): A list of VLChatProcessorOutput.
            padding (str): The padding method. Defaults to "left".

        Returns:
            BatchCollateOutput: A dictionary of the inputs to use for multimodal inference.
        """
        batched_sft_format = [sample.sft_format for sample in sample_list]
        batched_input_ids = [sample.input_ids for sample in sample_list]
        batched_labels = [sample.target_ids for sample in sample_list]
        batched_images_seq_mask = [sample["images_seq_mask"] for sample in sample_list]
        seq_lens = [len(sample) for sample in sample_list]
        """padding input_ids and images_seq_mask"""
        if padding == "left":
            padded_input_ids = self.tokenizer.pad({"input_ids": batched_input_ids})
            batched_input_ids, batched_attention_mask = (
                padded_input_ids["input_ids"],
                padded_input_ids["attention_mask"].astype(dtype="bool"),
            )
            batched_labels = self.tokenizer.pad({"input_ids": batched_labels})["input_ids"]
            batched_labels[batched_labels == self.pad_id] = self.ignore_id
            batched_images_seq_mask = self.tokenizer.pad({"input_ids": batched_images_seq_mask})["input_ids"]
        else:
            batched_input_ids = pad_sequence(batched_input_ids, padding_value=self.pad_id)
            batched_labels = pad_sequence(batched_labels, padding_value=self.ignore_id)
            batched_images_seq_mask = pad_sequence(batched_images_seq_mask, padding_value=0)
            batched_attention_mask = batched_input_ids != self.pad_id
        """padding images to max_patch_num"""
        max_n_patches = max(tuple(sample["images"].shape)[0] for sample in sample_list)
        batched_images = []
        for sample in sample_list:
            images = sample["images"]
            n_pads = max_n_patches - tuple(images.shape)[0]
            if n_pads > 0:
                pad_images = paddle.zeros(shape=(n_pads, *tuple(images.shape)[1:]), dtype=images.dtype)
                images = paddle.concat(x=[images, pad_images], axis=0)
            batched_images.append(images)
        batched_images = paddle.stack(x=batched_images, axis=0)
        """padding images_spatial_crop to max_n_images"""
        max_n_images = max(tuple(sample["images_spatial_crop"].shape)[0] for sample in sample_list)
        batched_images_spatial_crop = []
        for sample in sample_list:
            images_spatial_crop = sample["images_spatial_crop"]
            n_pads = max_n_images - tuple(sample["images_spatial_crop"].shape)[0]
            if n_pads > 0:
                pad_images_spatial_crop = paddle.full(shape=(n_pads, 2), fill_value=0, dtype=images_spatial_crop.dtype)
                images_spatial_crop = paddle.concat(x=[images_spatial_crop, pad_images_spatial_crop], axis=0)
            batched_images_spatial_crop.append(images_spatial_crop)

        batched_images_spatial_crop = paddle.stack(x=batched_images_spatial_crop, axis=0)
        batched_samples = DeepseekBatchCollateOutput(
            input_ids=batched_input_ids,
            attention_mask=batched_attention_mask,
            labels=batched_labels,
            images=batched_images,
            images_seq_mask=batched_images_seq_mask,
            images_spatial_crop=batched_images_spatial_crop,
            sft_format=batched_sft_format,
            seq_lens=seq_lens,
        )
        return batched_samples

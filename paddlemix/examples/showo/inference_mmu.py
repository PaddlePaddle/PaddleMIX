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

import os
from functools import partial
from typing import List, Union

import paddle
from omegaconf import OmegaConf
from paddle.vision import transforms
from paddlenlp.transformers import CLIPImageProcessor, CodeGenTokenizer
from PIL import Image
from tqdm import tqdm

from paddlemix.models.llava.conversation import conv_templates
from paddlemix.models.showo import CLIPVisionTower, MAGVITv2, Showo
from paddlemix.models.showo.prompting_utils import (
    UniversalPrompting,
    create_attention_mask_for_mmu,
    create_attention_mask_for_mmu_vit,
)

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
SYSTEM_PROMPT_LEN = 28


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


def image_transform(image, resolution=256, normalize=True):
    image = transforms.Resize(resolution, interpolation="bicubic")(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)
    return image


# helpers
def pad_sequence_paddle(sequences, padding_value=0):
    """
    Implement a function similar to PyTorch's pad_sequence in PaddlePaddle.

    Args:
    - sequences (list of Tensor): The list of sequences to be padded.
    - padding_value (float, optional): The value used for padding, default is 0.

    Returns:
    - Tensor: The result of padding all sequences to the same length.
    """
    # Calculate the maximum length
    max_len = max([seq.shape[0] for seq in sequences])

    # Pad sequences
    padded_sequences = []
    for seq in sequences:
        # Calculate the length to pad
        padding_len = max_len - seq.shape[0]

        # Create a padding tensor
        if padding_len > 0:
            padding_tensor = paddle.full([padding_len] + list(seq.shape[1:]), padding_value, dtype=seq.dtype)
            # Concatenate the original sequence and the padding tensor
            padded_seq = paddle.concat([seq, padding_tensor], axis=0)
        else:
            padded_seq = seq

        padded_sequences.append(padded_seq)

    # Stack the padded sequences to form a batch
    padded_batch = paddle.stack(padded_sequences, axis=0)
    return padded_batch


def orig_pad_sequence(
    sequences: Union[paddle.Tensor, List[paddle.Tensor]],
    batch_first: bool = False,
    padding_value: float = 0.0,
) -> paddle.Tensor:
    if batch_first:
        return pad_sequence_paddle(sequences, padding_value)
    else:
        assert False, "Not implemented"


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")


if __name__ == "__main__":

    config = get_config()

    tokenizer = CodeGenTokenizer.from_pretrained(config.model.showo.llm_model_path)

    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=(
            "<|soi|>",
            "<|eoi|>",
            "<|sov|>",
            "<|eov|>",
            "<|t2i|>",
            "<|mmu|>",
            "<|t2v|>",
            "<|v2v|>",
            "<|lvg|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
    )

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name)
    vq_model.eval()

    vision_tower_name = "openai/clip-vit-large-patch14-336"
    vision_tower = CLIPVisionTower(vision_tower_name)
    clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)

    model = Showo.from_pretrained(config.model.showo.pretrained_model_path, dtype=config.dtype)
    model.eval()

    temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 1  # retain only the top_k most likely tokens, clip others to have 0 probability

    file_list = os.listdir(config.mmu_image_root)
    responses = ["" for i in range(len(file_list))]
    images = []
    config.question = config.question.split(" *** ")
    for i, file_name in enumerate(tqdm(file_list)):
        image_path = os.path.join(config.mmu_image_root, file_name)
        image_ori = Image.open(image_path).convert("RGB")
        image = image_transform(image_ori, resolution=config.dataset.params.resolution)
        image = image.unsqueeze(0)
        images.append(image)

        pixel_values = clip_image_processor.preprocess(image_ori, return_tensors="pd")["pixel_values"][0]

        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
        batch_size = 1

        for question in config.question:
            if config.model.showo.w_clip_vit:
                pad_sequence = partial(orig_pad_sequence, batch_first=True)
                conv = conv_templates["phi1.5"].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                question_input = []
                question_input.append(prompt_question.strip())

                input_ids_system = [
                    uni_prompting.text_tokenizer(SYSTEM_PROMPT, return_tensors="pd", padding="longest").input_ids
                    for _ in range(batch_size)
                ]
                input_ids_system = paddle.stack(input_ids_system, axis=0)
                assert input_ids_system.shape[-1] == 28

                input_ids_system = input_ids_system[0]

                input_ids = [
                    uni_prompting.text_tokenizer(prompt, return_tensors="pd", padding="longest").input_ids
                    for prompt in question_input
                ]

                input_ids = paddle.stack(input_ids)
                input_ids = pad_sequence(
                    input_ids, batch_first=True, padding_value=uni_prompting.text_tokenizer.pad_token_id
                )
                input_ids = paddle.to_tensor(input_ids).squeeze(0)

                # input_ids.shape [1, 13]
                input_ids_llava = paddle.concat(
                    [
                        (paddle.ones([input_ids.shape[0], 1]).astype("int64") * uni_prompting.sptids_dict["<|mmu|>"]),
                        input_ids_system,
                        (paddle.ones([input_ids.shape[0], 1]).astype("int64") * uni_prompting.sptids_dict["<|soi|>"]),
                        # place your img embedding here
                        (paddle.ones([input_ids.shape[0], 1]).astype("int64") * uni_prompting.sptids_dict["<|eoi|>"]),
                        input_ids,
                    ],
                    axis=1,
                )

                images_embeddings = vision_tower(pixel_values[None])
                # [1, 576, 1024] 36324.25390625
                images_embeddings = model.mm_projector(images_embeddings).astype(config.dtype)
                # [1, 576, 2048] -19129.31445312
                text_embeddings = model.showo.model.embed_tokens(input_ids_llava)

                # Full input seq
                part1 = text_embeddings[:, : 2 + SYSTEM_PROMPT_LEN, :]
                part2 = text_embeddings[:, 2 + SYSTEM_PROMPT_LEN :, :]
                input_embeddings = paddle.concat((part1, images_embeddings, part2), axis=1)
                # [1, 620, 2048] -19089.80859375

                attention_mask_llava = create_attention_mask_for_mmu_vit(
                    input_embeddings, system_prompt_len=SYSTEM_PROMPT_LEN
                )

                # [1, 1, 620, 620] sum -83102582052061530030080.
                cont_toks_list = model.mmu_generate(
                    input_embeddings=input_embeddings,
                    attention_mask=attention_mask_llava[0].unsqueeze(0),
                    max_new_tokens=config.max_new_tokens,
                    top_k=top_k,
                    eot_token=tokenizer.eos_token_id,
                )
            else:
                input_ids = uni_prompting.text_tokenizer(["USER: \n" + question + " ASSISTANT:"])["input_ids"]
                input_ids = paddle.to_tensor(input_ids).astype("int64")
                input_ids = paddle.concat(
                    [
                        (paddle.ones([input_ids.shape[0], 1]).astype("int64") * uni_prompting.sptids_dict["<|mmu|>"]),
                        (paddle.ones([input_ids.shape[0], 1]).astype("int64") * uni_prompting.sptids_dict["<|soi|>"]),
                        image_tokens,
                        (paddle.ones([input_ids.shape[0], 1]).astype("int64") * uni_prompting.sptids_dict["<|eoi|>"]),
                        (paddle.ones([input_ids.shape[0], 1]).astype("int64") * uni_prompting.sptids_dict["<|sot|>"]),
                        input_ids,
                    ],
                    axis=1,
                )

                attention_mask = create_attention_mask_for_mmu(
                    input_ids, eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"])
                )

                cont_toks_list = model.mmu_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config.max_new_tokens,
                    top_k=top_k,
                    eot_token=uni_prompting.sptids_dict["<|eot|>"],
                )

            cont_toks_list_stack = paddle.stack(cont_toks_list).squeeze()[None]

            text = uni_prompting.text_tokenizer.batch_decode(cont_toks_list_stack, skip_special_tokens=True)
            print("User: " + question + "\n Answer : " + text[0] + "\n")

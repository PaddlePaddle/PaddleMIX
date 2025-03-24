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

import os

import paddle
from ImageReward.models.BLIP.blip_pretrain import BLIP_Pretrain
from ImageReward.utils import ImageReward_download
from PIL import Image

try:
    BICUBIC = "bicubic"
except ImportError:
    BICUBIC = Image.BICUBIC

_MODELS = {"ImageReward-v1.0": "ImageReward.pdparams"}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return paddle.vision.transforms.Compose(
        transforms=[
            paddle.vision.transforms.Resize(size=n_px, interpolation=BICUBIC),
            paddle.vision.transforms.CenterCrop(size=n_px),
            _convert_image_to_rgb,
            paddle.vision.transforms.ToTensor(),
            paddle.vision.transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class MLP(paddle.nn.Layer):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=self.input_size, out_features=1024),
            paddle.nn.Dropout(p=0.2),
            paddle.nn.Linear(in_features=1024, out_features=128),
            paddle.nn.Dropout(p=0.2),
            paddle.nn.Linear(in_features=128, out_features=64),
            paddle.nn.Dropout(p=0.1),
            paddle.nn.Linear(in_features=64, out_features=16),
            paddle.nn.Linear(in_features=16, out_features=1),
        )
        for name, param in self.layers.named_parameters():
            if "weight" in name:
                init_Normal = paddle.nn.initializer.Normal(mean=0.0, std=1.0 / (self.input_size + 1))
                init_Normal(param)
            if "bias" in name:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(param)

    def forward(self, input):
        return self.layers(input)


class IRSMC(paddle.nn.Layer):
    def __init__(self, med_config):
        super().__init__()
        self.blip = BLIP_Pretrain(image_size=224, vit="large", med_config=med_config)
        self.preprocess = _transform(224)
        self.mlp = MLP(768)
        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072

    def score_batched_old(self, prompts, images):
        results = []
        for i, prompt in enumerate(prompts):
            results.append(self.score(prompt, images[i]))
        return results

    def score_gard(self, prompt_ids, prompt_attention_mask, image):
        image_embeds = self.blip.visual_encoder(image)
        image_atts = paddle.ones(shape=tuple(image_embeds.shape)[:-1], dtype="int64")
        text_output = self.blip.text_encoder(
            prompt_ids,
            attention_mask=prompt_attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        txt_features = text_output.last_hidden_state[:, 0, :]
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std
        return rewards

    def score(self, prompt, image):
        if type(image).__name__ == "list":
            _, rewards = self.inference_rank(prompt, image)
            return rewards
        text_input = self.blip.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pd",
            return_attention_mask=True,
        )
        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str) and os.path.isfile(image):
            pil_image = Image.open(image)
        else:
            raise TypeError(
                "This image parameter type has not been supportted yet. Please pass PIL.Image or file path str."
            )
        image = self.preprocess(pil_image).unsqueeze(axis=0)
        image_embeds = self.blip.visual_encoder(image)
        image_atts = paddle.ones(shape=tuple(image_embeds.shape)[:-1], dtype="int64")
        text_output = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        txt_features = text_output.last_hidden_state[:, 0, :].astype(dtype="float32")
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std
        return rewards.detach().cpu().numpy().item()

    def score_batched(self, prompts, images):
        assert isinstance(prompts, list)
        assert isinstance(images, list)

        text_input = self.blip.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pd",
            return_attention_mask=True,
        )

        images = [self.preprocess(image).unsqueeze(axis=0) for image in images]
        images = paddle.concat(x=images, axis=0)

        image_embeds = self.blip.visual_encoder(images)
        image_atts = paddle.ones(shape=tuple(image_embeds.shape)[:-1], dtype="int64")

        text_output = self.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        txt_features = text_output.last_hidden_state[:, 0, :].astype(dtype="float32")

        rewards = self.mlp(txt_features)

        rewards = (rewards - self.mean) / self.std

        return rewards.reshape([tuple(txt_features.shape)[0]]).detach().cpu().numpy().tolist()

    def inference_rank(self, prompt, generations_list):
        text_input = self.blip.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pd",
            return_attention_mask=True,
        )
        txt_set = []
        for generation in generations_list:
            if isinstance(generation, Image.Image):
                pil_image = generation
            elif isinstance(generation, str):
                if os.path.isfile(generation):
                    pil_image = Image.open(generation)
            else:
                raise TypeError(
                    "This image parameter type has not been supportted yet. Please pass PIL.Image or file path str."
                )
            image = self.preprocess(pil_image).unsqueeze(axis=0)
            image_embeds = self.blip.visual_encoder(image)
            image_atts = paddle.ones(shape=tuple(image_embeds.shape)[:-1], dtype="int64")
            text_output = self.blip.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            txt_set.append(text_output.last_hidden_state[:, 0, :])
        txt_features = paddle.concat(x=txt_set, axis=0).astype(dtype="float32")
        rewards = self.mlp(txt_features)
        rewards = (rewards - self.mean) / self.std
        rewards = paddle.squeeze(x=rewards)
        _, rank = paddle.sort(descending=True, x=rewards, axis=0), paddle.argsort(descending=True, x=rewards, axis=0)
        _, indices = paddle.sort(x=rank, axis=0), paddle.argsort(x=rank, axis=0)
        indices = indices + 1
        return (
            indices.detach().cpu().numpy().tolist(),
            rewards.detach().cpu().numpy().tolist(),
        )


def rm_load(
    name: str = "ImageReward-v1.0",
    download_root: str = None,
    med_config: str = None,
):
    """Load a ImageReward model

    Parameters
    ----------
    name : str
        A model name listed by `ImageReward.available_models()`, or the path to a model checkpoint containing the state_dict


    download_root: str
        path to download the model files; by default, it uses "./"

    Returns
    -------
    model : The ImageReward model
    """
    if name in _MODELS:
        model_path = _MODELS[name]
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found;")
    print("load checkpoint from %s" % model_path)
    state_dict = paddle.load(path=str(model_path))

    if med_config is None:
        med_config = ImageReward_download(
            "https://huggingface.co/THUDM/ImageReward/blob/main/med_config.json",
            download_root or os.path.expanduser("./"),
        )

    model = IRSMC(med_config=med_config)
    model.set_state_dict(state_dict=state_dict)
    print("checkpoint loaded")
    model.eval()
    return model

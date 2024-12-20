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

import dataclasses
from enum import Enum, auto
from io import BytesIO
from typing import List, Optional

import paddle
import paddle.nn as nn
import requests
from paddle.vision import transforms
from paddlenlp.generation.stopping_criteria import StoppingCriteriaList
from paddlenlp.transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Model
from paddlenlp.transformers.model_outputs import CausalLMOutputWithPast
from PIL import Image

from ...processors.got_process import BlipImageEvalProcessor
from .got_vision_b import build_GOT_vit_b

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<imgpad>"
DEFAULT_IM_START_TOKEN = "<img>"
DEFAULT_IM_END_TOKEN = "</img>"


class Qwen2LMHead(nn.Layer):
    def __init__(self, config, embedding_weights=None, transpose_y=False, tensor_parallel_output=1):
        super(Qwen2LMHead, self).__init__()
        self.config = config
        vocab_size = config.vocab_size

        self.transpose_y = transpose_y
        if transpose_y:
            # only for weight from embedding_weights
            if embedding_weights is not None:
                self.weight = embedding_weights
            else:
                self.weight = self.create_parameter(
                    shape=[vocab_size, config.hidden_size],
                    dtype=paddle.get_default_dtype(),
                )
        else:
            # for weight from model init
            self.weight = self.create_parameter(
                shape=[config.hidden_size, vocab_size],
                dtype=paddle.get_default_dtype(),
            )

    def forward(self, hidden_states, tensor_parallel_output=1):
        logits = paddle.matmul(hidden_states, self.weight, transpose_y=self.transpose_y)
        return logits


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "<|im_end|>"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep + "\n"
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        if self.sep_style == SeparatorStyle.MPT:
            if self.system:
                ret = self.system + self.sep
            else:
                ret = ""
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
        )


class KeywordsStoppingCriteria(StoppingCriteriaList):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
        self.keyword_ids = [
            keyword_id[0] for keyword_id in self.keyword_ids if type(keyword_id) is list and len(keyword_id) == 1
        ]
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: paddle.Tensor, scores: paddle.Tensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            for keyword_id in self.keyword_ids:
                if output_ids[0, -1] == keyword_id:
                    return True
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len :], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


class GOTImageEvalProcessor:
    def __init__(self, image_size=384, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation="bicubic"),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)


class GOTConfig(Qwen2Config):
    model_type = "GOT"


class GOTQwenModel(Qwen2Model):
    config_class = GOTConfig

    def __init__(self, config: Qwen2Config):
        super(GOTQwenModel, self).__init__(config)
        self.vision_tower_high = build_GOT_vit_b()
        self.mm_projector_vary = nn.Linear(1024, 1024)

    def initialize_vision_modules(
        self,
        vision_tower,
        pretrained_stage1_model=None,
        freeze_vision_tower=False,
        use_im_start_end=False,
        vision_select_layer=-1,
        dtype=paddle.float16,
    ):
        # Vary old codes, not use in GOT
        image_processor = BlipImageEvalProcessor(image_size=1024)
        # 1024*1024

        image_processor_high = BlipImageEvalProcessor(image_size=1024)

        self.vision_tower_high = self.vision_tower_high.to(dtype=dtype)

        self.mm_projector_vary = self.mm_projector_vary.to(dtype=dtype)

        image_token_len = 256

        self.config.vision_tower = vision_tower
        self.config.image_token_len = image_token_len

        self.config.use_im_start_end = True

        self.config.vision_select_layer = vision_select_layer
        self.config.freeze_vision_tower = freeze_vision_tower

        return dict(
            image_processor=image_processor,
            image_processor_high=image_processor_high,
            image_token_len=image_token_len,
        )

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, "orig_embeds_params", None)
        if orig_embeds_params is not None:
            with paddle.no_grad():
                self.get_input_embeddings().weight[: -self.num_new_tokens] = orig_embeds_params[
                    : -self.num_new_tokens
                ].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        vision_tower_high = getattr(self, "vision_tower_high", None)

        if vision_tower_high is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:
            use_im_start_end = getattr(self.config, "use_im_start_end", -1)

            im_patch_token = getattr(self.config, "im_patch_token", -1)
            im_start_token = getattr(self.config, "im_start_token", -1)
            im_end_token = getattr(self.config, "im_end_token", -1)

            im_patch_token = 151859
            im_start_token = 151857
            im_end_token = 151858

            image_features = []

            for image in images:
                if self.training:
                    image = image[1]
                P, C, H, W = image.shape
                if P == 1:
                    with paddle.set_grad_enabled(False):
                        cnn_feature = vision_tower_high(image)
                        cnn_feature = cnn_feature.flatten(2).transpose([0, 2, 1])  # 256*1024
                    image_feature = self.mm_projector_vary(cnn_feature)
                    image_features.append(image_feature)

                else:
                    image_patches = paddle.unbind(image)
                    image_patches_features = []
                    for image_patch in image_patches:
                        image_p = paddle.stack([image_patch])
                        with paddle.set_grad_enabled(False):
                            cnn_feature_p = vision_tower_high(image_p)
                            cnn_feature_p = cnn_feature_p.flatten(2).transpose([0, 2, 1])
                        image_feature_p = self.mm_projector_vary(cnn_feature_p)
                        image_patches_features.append(image_feature_p)
                    image_feature = paddle.concat(image_patches_features, axis=1)
                    image_features.append(image_feature)

            dummy_image_features_2 = paddle.zeros([256, 1024], dtype=inputs_embeds.dtype)
            dummy_image_features = dummy_image_features_2
            use_im_start_end = True
            new_input_embeds = []
            for cur_input_ids, cur_input_embeds, cur_image_features in zip(input_ids, inputs_embeds, image_features):
                if (cur_input_ids == im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0.0 * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue

                if use_im_start_end:
                    if (cur_input_ids == im_start_token).sum() != (cur_input_ids == im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")

                    image_start_tokens = paddle.where(cur_input_ids == im_start_token)[0]
                    for image_start_token_pos, per_cur_image_features in zip(image_start_tokens, cur_image_features):
                        num_patches = per_cur_image_features.shape[0]

                        if cur_input_ids[image_start_token_pos + num_patches + 1] != im_end_token:
                            raise ValueError("The image end token should follow the image start token.")

                        cur_input_embeds = paddle.concat(
                            (
                                cur_input_embeds[: image_start_token_pos + 1],
                                per_cur_image_features,
                                cur_input_embeds[image_start_token_pos + num_patches + 1 :],
                            ),
                            axis=0,
                        )

                    new_input_embeds.append(cur_input_embeds)
                else:
                    raise NotImplementedError

            inputs_embeds = paddle.stack(new_input_embeds, axis=0)

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,  #
            past_key_values=past_key_values,  # None
            inputs_embeds=inputs_embeds,  # [1, 800, 1024]
            use_cache=use_cache,  # True
            position_ids=position_ids,  #
            output_attentions=output_attentions,  # False
            output_hidden_states=output_hidden_states,  # False
            return_dict=return_dict,  # False
        )


class GOTQwenForCausalLM(Qwen2ForCausalLM):
    config_class = GOTConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.qwen2 = GOTQwenModel(config)

        self.vocab_size = config.vocab_size

        if config.tie_word_embeddings:
            self.lm_head = Qwen2LMHead(config, embedding_weights=self.qwen2.embed_tokens.weight, transpose_y=True)
            self.tie_weights()
        else:
            self.lm_head = Qwen2LMHead(config)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_model(self):
        return self.qwen2

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.qwen2(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.astype(dtype="float32")

        # logits
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            # loss_fct = nn.CrossEntropyLoss()
            loss_fct = nn.CrossEntropyLoss(reduction="sum")
            shift_logits = shift_logits.reshape([-1, self.config.vocab_size])
            shift_labels = shift_labels.reshape([-1])
            # Enable model parallelism

            loss = loss_fct(shift_logits, shift_labels)
            label_sum = paddle.sum(shift_labels != -100)  # .cast("float32")
            loss = loss / label_sum

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        batch_size, seq_length = input_ids.shape
        attention_mask = paddle.ones((batch_size, seq_length), dtype=paddle.bool)

        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[1]  # [1, 800, 16, 64]
            if past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.astype(dtype="int64").cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(
        self,
        tokenizer,
        freeze_lm_model=False,
        pretrained_stage1_model=None,
    ):
        config = self.get_model().config

        self.resize_token_embeddings(len(tokenizer))

        config.im_patch_token = 151859

        config.use_im_start_end = True

        if config.use_im_start_end:
            self.resize_token_embeddings(len(tokenizer))
            config.im_start_token, config.im_end_token = 151857, 151858

    def load_image(self, image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image

    def chat(
        self,
        tokenizer,
        image_file,
        ocr_type,
        ocr_box="",
        ocr_color="",
        render=False,
        save_render_file=None,
        print_prompt=False,
        gradio_input=False,
        stream_flag=False,
        dtype="bfloat16",
    ):

        image_processor_high = GOTImageEvalProcessor(image_size=1024)

        use_im_start_end = True

        image_token_len = 256

        if gradio_input:
            image = image_file.copy()
        else:
            image = self.load_image(image_file)

        w, h = image.size

        if ocr_type == "format":
            qs = "OCR with format: "
        else:
            qs = "OCR: "

        if ocr_box:
            bbox = eval(ocr_box)
            if len(bbox) == 2:
                bbox[0] = int(bbox[0] / w * 1000)
                bbox[1] = int(bbox[1] / h * 1000)
            if len(bbox) == 4:
                bbox[0] = int(bbox[0] / w * 1000)
                bbox[1] = int(bbox[1] / h * 1000)
                bbox[2] = int(bbox[2] / w * 1000)
                bbox[3] = int(bbox[3] / h * 1000)
            if ocr_type == "format":
                qs = str(bbox) + " " + "OCR with format: "
            else:
                qs = str(bbox) + " " + "OCR: "

        if ocr_color:
            if ocr_type == "format":
                qs = "[" + ocr_color + "]" + " " + "OCR with format: "
            else:
                qs = "[" + ocr_color + "]" + " " + "OCR: "

        if use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN + "\n" + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv_mpt = Conversation(
            system="""<|im_start|>system
        You should follow the instructions carefully and explain your answers in detail.""",
            # system = None,
            roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
            version="mpt",
            messages=(),
            offset=0,
            sep_style=SeparatorStyle.MPT,
            sep="<|im_end|>",
        )

        conv = conv_mpt.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        inputs = tokenizer([prompt])

        image_tensor_1 = image_processor_high(image)
        input_ids = paddle.to_tensor(inputs.input_ids)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        output_ids = self.generate(
            input_ids,
            images=[image_tensor_1.unsqueeze(0).to(dtype)],
            do_sample=False,
            num_beams=1,
            no_repeat_ngram_size=20,
            max_new_tokens=4096,
            stopping_criteria=stopping_criteria,  # list of stopping criteria
        )[0]

        outputs = tokenizer.decode(output_ids[0]).strip()

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        response_str = outputs
        return response_str

    def dynamic_preprocess(self, image, min_num=1, max_num=6, image_size=1024, use_thumbnail=True):
        def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
            best_ratio_diff = float("inf")
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            return best_ratio

        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def chat_crop(
        self,
        tokenizer,
        image_file,
        ocr_type,
        render=False,
        save_render_file=None,
        print_prompt=False,
        gradio_input=False,
        stream_flag=False,
        dtype="bfloat16",
    ):
        # Model
        multi_page = False

        image_processor_high = GOTImageEvalProcessor(image_size=1024)

        use_im_start_end = True

        image_token_len = 256

        image_list = []

        if multi_page:
            qs = "OCR with format across multi pages: "
            patches = image_file
            sub_images = []
            for sub_image in patches:
                sub_images.append(self.load_image(sub_image))
            ll = len(patches)

        else:
            if ocr_type == "format":
                qs = "OCR with format upon the patch reference: "
            else:
                qs = "OCR upon the patch reference: "
            if gradio_input:
                img = image_file.copy()
            else:
                img = self.load_image(image_file)
            sub_images = self.dynamic_preprocess(img)
            ll = len(sub_images)

        for image in sub_images:
            image_tensor_1 = image_processor_high(image)
            image_list.append(image_tensor_1)

        image_list = paddle.stack(image_list)

        print("====new images batch size======:  \n", image_list.shape)

        if use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len * ll
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv_mpt = Conversation(
            system="""<|im_start|>system
        You should follow the instructions carefully and explain your answers in detail.""",
            # system = None,
            roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
            version="mpt",
            messages=(),
            offset=0,
            sep_style=SeparatorStyle.MPT,
            sep="<|im_end|>",
        )

        conv = conv_mpt.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        inputs = tokenizer([prompt])

        input_ids = paddle.to_tensor(inputs.input_ids)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        output_ids = self.generate(
            input_ids,
            images=[image_list.to(dtype)],
            do_sample=False,
            num_beams=1,
            # no_repeat_ngram_size = 20,
            max_new_tokens=4096,
            stopping_criteria=stopping_criteria,
        )[0]

        outputs = tokenizer.decode(output_ids[0]).strip()

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        response_str = outputs
        return response_str

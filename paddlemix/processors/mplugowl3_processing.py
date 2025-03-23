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

import random
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import paddle
import paddle.vision.transforms as transforms
from einops import rearrange, repeat
from paddle.vision.transforms import Resize
from paddlenlp.transformers.image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
)
from paddlenlp.transformers.processing_utils import ProcessorMixin
from PIL import Image

OWL_MEDIA_TOKEN = ["<|image|>"]


def recursive_converter(converter, value):
    if isinstance(value, list):
        new_value = []
        for v in value:
            new_value += [recursive_converter(converter, v)]
        return new_value
    else:
        return converter(value)


def box_area(boxes):
    # 获取边界框的宽度和高度
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    # 计算面积
    area = width * height
    return area


def custom_max(a, b):
    return paddle.where(a > b, a, b)


def custom_min(a, b):
    return paddle.where(a < b, a, b)


def box_iou(boxes1, area1, boxes2, eps=1e-05):
    # >>>>>>    area2 = torchvision.ops.boxes.box_area(boxes2)
    area1 = area1.astype("float32")
    boxes1 = boxes1.astype("float32")
    boxes2 = boxes2.astype("float32")

    area2 = box_area(boxes2).astype("float32")
    lt = custom_max(boxes1[:, None, :2], boxes2[:, :2])
    rb = custom_min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / (union + eps)
    return iou, union


# def box_iou(boxes1, area1, boxes2, eps=1e-5):
#     area2 = box_area(boxes2)

#     lt = paddle.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
#     rb = paddle.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

#     wh = (rb - lt).clip(min=0)  # [N,M,2]
#     inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

#     union = area1[:, None] + area2 - inter

#     iou = inter / (union + eps)
#     return iou, union


available_anchor_strategy = ["docowl", "random", "highest", "last", "llava"]

grid_dict = {
    "grid_33": [
        (1, 1),
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
        (2, 2),
        (1, 4),
        (4, 1),
        (1, 5),
        (5, 1),
        (1, 6),
        (6, 1),
        (2, 3),
        (3, 2),
        (1, 7),
        (7, 1),
        (4, 2),
        (2, 4),
        (1, 8),
        (8, 1),
        (3, 3),
        (1, 9),
        (9, 1),
    ],
    "grid_squ_3x3": [(1, 1), (2, 2), (3, 3)],
    "grid_squ_4": [(2, 2), (1, 3), (1, 4), (3, 1), (4, 1)],
    "grid_squ_6": [(2, 2), (1, 3), (1, 4), (3, 1), (4, 1), (2, 3), (3, 2)],
    "grid_squ_2": [(2, 1)],
    "grid_squ_9": [
        (1, 1),
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
        (2, 2),
        (1, 4),
        (4, 1),
        (1, 5),
        (5, 1),
        (1, 6),
        (6, 1),
        (2, 3),
        (3, 2),
        (1, 7),
        (7, 1),
        (4, 2),
        (2, 4),
        (1, 8),
        (8, 1),
        (3, 3),
        (1, 9),
        (9, 1),
    ],
}


cut_prompt_template_dict = {
    'v0': lambda img_token, h, w: f''.join([f"{img_token}" for i in range(h) for j in range(w)]),
    'v1': lambda img_token, h, w: f'Cut to {h} rows {w} columns, '+ ' '.join([f"subimg({i},{j}){img_token}"for i in range(h) for j in range(w)]),
    'v1_global': lambda img_token, h, w: f'Cut to {h} rows {w} columns with a global view, '+ ' '.join([f"subimg({i},{j}){img_token}"for i in range(h) for j in range(w)]+[f"global_view{img_token}"]),
    'v2_global': lambda img_token, h, w: f'Cut to {h} rows {w} columns with a global view\n'+ '\n'.join([' '.join([f"subimg({i},{j}){img_token}" for j in range(w)]) for i in range(h)])+f"\nglobal_view{img_token}",
    'v3': lambda img_token, h, w: f'<|start_cut|>{h}*{w}'+ ' '.join([f"{img_token}"for i in range(h) for j in range(w)])+'<|end_cut|>',
    'v3_global': lambda img_token, h, w: f'<|start_cut|>{h}*{w}\n'+ '\n'.join([' '.join([f"{img_token}" for j in range(w)]) for i in range(h)])+f'\n{img_token}<|end_cut|>',
}


def anchor_rank(anchors, anchors_areas, input_image_size, eps=1e-5):
    # anchors x1 y1 x2 y2

    # image_size: (h, w)
    # xyxy
    input_image_bbox = paddle.to_tensor([0, 0, input_image_size[1], input_image_size[0]]).unsqueeze(0)

    boxes1 = anchors
    boxes2 = input_image_bbox
    boxes3 = anchors.clone()
    # y2
    boxes3[:, 3] = input_image_size[0] / input_image_size[1] * anchors[:, 2]  # 用于算分辨率无关的iou

    area1 = anchors_areas

    iou, _ = box_iou(boxes1, area1, boxes2)
    iou = iou.squeeze(1)
    shape_iou, _ = box_iou(boxes1, area1, boxes3)
    shape_iou = shape_iou.diag()
    # 优先匹配形状接近 再匹配分辨率接近
    index = paddle.argmax(shape_iou * 100 + iou, axis=0)
    return index


def select_best_resolution(anchors, anchors_areas, input_image_size):  # TODO For a further check
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_size = (input_image_size[1], input_image_size[0])
    possible_resolutions = [(_[2], _[3]) for _ in anchors]  # xyxy -> w,h

    original_width, original_height = original_size
    # best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    index = 0
    for i, (width, height) in enumerate(possible_resolutions):
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            # best_fit = (width, height)
            index = i

    return index


def build_cut_shape_indices(cut_shape):
    # cut_shape: a list of (nh,nw)
    cut_shape_indices = []
    for shape in cut_shape:
        n = shape[0] * shape[1]
        indices = paddle.concat(
            [repeat(paddle.to_tensor(shape), "l -> n l", n=n), paddle.arange(n).unsqueeze(1)], axis=1
        )
        assert indices.shape[0] == n
        assert indices.shape[1] == 3  # nh,nw,idx

        cut_shape_indices.append(indices)
    cut_shape_indices = paddle.concat(cut_shape_indices, axis=0).astype("int64")
    return cut_shape_indices


class AnchorResize(paddle.nn.Layer):
    def __init__(self, image_size, anchors, interpolation="bilinear", antialias=None, anchor_strategy="docowl"):
        super().__init__()
        self.image_size = image_size
        # xyxy
        self.anchors = paddle.to_tensor(
            [[0, 0, _[1] * image_size[1], _[0] * image_size[0]] for _ in anchors],
        )

        self.anchor_areas = box_area(self.anchors)

        self.interpolation = interpolation
        self.antialias = antialias
        self.anchor_strategy = anchor_strategy
        assert self.anchor_strategy in available_anchor_strategy

    def resize_global(self, img):
        transform = Resize(size=self.image_size, interpolation=self.interpolation)
        return transform(img)

    def forward(self, img, skip_resize=False):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        if self.anchor_strategy == "docowl":
            selected_anchor = anchor_rank(self.anchors, self.anchor_areas, (img.size[1], img.size[0]))
        elif self.anchor_strategy == "random":
            selected_anchor = random.randint(0, len(self.anchors) - 1)
        elif self.anchor_strategy == "highest":
            # 选面积最大的 在这个基础上 尽可能选最方正的
            selected_anchor = paddle.argmax(
                self.anchors[:, 2] * self.anchors[:, 3] * 100 - paddle.abs(self.anchors[:, 2] - self.anchors[:, 3])
            )
        elif self.anchor_strategy == "last":
            selected_anchor = len(self.anchors) - 1
        elif self.anchor_strategy == "llava":
            selected_anchor = select_best_resolution(self.anchors, self.anchor_areas, (img.size[1], img.size[0]))
        else:
            selected_anchor = None
        assert selected_anchor is not None

        target_size = self.anchors[selected_anchor][2:].tolist()  # w,h
        if skip_resize:
            # for debug
            return selected_anchor
        # return F.resize(img, [target_size[1],target_size[0]], self.interpolation, max_size=None, antialias=self.antialias), selected_anchor
        # image_np = np.array(img)
        # image_tensor = paddle.to_tensor(image_np, dtype="float32")
        # image_tensor = image_tensor.transpose([2, 0, 1])  # 变成 (3, 500, 500)
        # if self.interpolation == "bilinear" or "bicubic":
        #     image_tensor = image_tensor.unsqueeze(0)  # 变成 (1, 3, 500, 500)
        transform = Resize(size=[target_size[1], target_size[0]], interpolation=self.interpolation)
        return (transform(img), selected_anchor)
        # return (
        #     F.interpolate(
        #         image_tensor, size=[target_size[1], target_size[0]], mode=self.interpolation, align_corners=False
        #     )[0],
        #     selected_anchor,
        # )

    def __repr__(self) -> str:
        detail = f"(size={self.image_size}, anchor={self.anchors}, interpolation={self.interpolation.value}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"


class CutMixin:
    def __init__(
        self,
        cut_cfg={
            "anchors": "grid_squ_6",
            "anchor_strategy": "docowl",
            "cut_prompt": "v3",
            "add_global": True,
            "cut_prob": 1.0,
        },
    ) -> None:
        if cut_cfg is None:
            self.cut_enable = False
            return
        else:
            self.cut_enable = True
        image_size = self.image_size
        anchors = cut_cfg.get("anchors", "grid_33")
        anchor_strategy = cut_cfg.get("anchor_strategy", "docowl")
        cut_prompt = cut_cfg.get("cut_prompt", "v0")
        self.cut_prob = cut_cfg.get("cut_prob", 1.0)

        self.force_shape_cut = cut_cfg.get("force_shape_cut", False)
        force_shape_cut_anchors = cut_cfg.get("force_shape_cut_anchors", "force_shape_cut_anchors")

        self.add_global = cut_cfg.get("add_global", False)

        # h,w
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size

        if anchors in grid_dict:
            anchors = grid_dict[anchors]
        else:
            anchors = eval(anchors)
        self.anchors = [tuple(_) for _ in anchors]
        self.anchor_max = max([max(_) for _ in self.anchors])
        self.resizer = AnchorResize(
            image_size=image_size, anchors=anchors, interpolation="bicubic", anchor_strategy=anchor_strategy
        )

        if force_shape_cut_anchors in grid_dict:
            force_shape_cut_anchors = grid_dict[force_shape_cut_anchors]
        else:
            force_shape_cut_anchors = eval(force_shape_cut_anchors)
        self.force_shape_cut_anchors = [tuple(_) for _ in force_shape_cut_anchors]
        self.force_shape_cut_anchors_max = max([max(_) for _ in self.force_shape_cut_anchors])

        self.old_resizer = transforms.Resize(image_size, interpolation="bicubic")

        # 把image processor的缩放去掉 只保留后面的变换
        self.image_transform = transforms.Compose(self.image_transform.transforms[1:])
        if self.add_global:
            self.cut_prompt_template = cut_prompt_template_dict[cut_prompt + "_global"]
        else:
            self.cut_prompt_template = cut_prompt_template_dict[cut_prompt]

        self.media_tokens = ["<|image|>", "<|video|>"]

    def _process_image(self, images):
        new_images = []
        cut_shape = []
        for image in images:
            raw_image = image
            image, selected_anchor = self.resizer(image)
            image_input = self.image_transform(image)  # h,w,3 -> 3,h,w
            cut_shape.append(
                (image_input.shape[1] // self.image_size[0], image_input.shape[2] // self.image_size[1])
            )  # cut_h, cut_w
            image_input = rearrange(
                image_input, "C (num_h h) (num_w w) -> (num_h num_w) C h w", h=self.image_size[0], w=self.image_size[1]
            )

            new_images.append(image_input)

            if self.add_global:
                new_images.append(self.image_transform(self.resizer.resize_global(raw_image)).unsqueeze(0))
                cut_shape.append((1, 1))

        new_images = paddle.concat(new_images, axis=0)
        cut_shape_indices = build_cut_shape_indices(cut_shape)
        return new_images, cut_shape, cut_shape_indices


class TensorType(Enum):
    PADDLE = "paddle"


class mPLUGOwl3BatchFeature(BatchFeature):
    r"""
    Extend from BatchFeature for supporting various image size
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None, tensor_type: Union[None, str, TensorType] = None):
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type)

    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None):
        if tensor_type is None:
            return self

        # is_tensor, as_tensor = self._get_is_as_tensor_fns(tensor_type)
        is_tensor = lambda x: isinstance(x, paddle.Tensor)
        as_tensor = paddle.to_tensor

        def converter(value):
            try:
                if not is_tensor(value):
                    tensor = as_tensor(value)
                    return tensor
            except:  # noqa E722
                if key == "overflowing_values":
                    raise ValueError("Unable to create tensor returning overflowing values of different lengths. ")
                raise ValueError(
                    "Unable to create tensor, you should probably activate padding "
                    "with 'padding=True' to have batched tensors with the same length."
                )

        for key, value in self.items():
            self[key] = recursive_converter(converter, value)
        return self


class mPLUGOwl3ImageProcessor(BaseImageProcessor, CutMixin):
    model_input_names = ["pixel_values"]

    def __init__(self, image_size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation="bicubic"),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        CutMixin.__init__(self)

    def preprocess(
        self, images: Union[Image.Image, List[Image.Image]], cut_enable=True, **kwargs
    ) -> mPLUGOwl3BatchFeature:
        if isinstance(images, Image.Image):
            images_list = [images]
        else:
            images_list = images

        if self.cut_enable and cut_enable:
            image_data, cut_shape, cut_shape_indices = self._process_image(images_list)
        else:
            image_data = [self.image_transform(self.resizer.resize_global(image)) for image in images_list]
            image_data = paddle.stack(image_data, axis=0)
            cut_shape = cut_shape_indices = None

        return mPLUGOwl3BatchFeature(
            data={"pixel_values": image_data, "cut_shape": cut_shape, "cut_shape_indices": cut_shape_indices}
        )

    def to_dict(self):
        encoder_dict = super().to_dict()
        pop_keys = ["image_transform", "resizer", "old_resizer", "cut_prompt_template"]
        for pk in pop_keys:
            encoder_dict.pop(pk, None)
        return encoder_dict


class MediaIndicesHelper:
    def __init__(self, tokenizer) -> None:
        self.media_position = []
        self.tokenizer = tokenizer

    def has_media(self, text, media_tokens=None):
        if media_tokens is None:
            media_tokens = OWL_MEDIA_TOKEN
        has_media_flag = any([media_token == text for media_token in media_tokens])
        if any([media_token in text for media_token in media_tokens]):
            # 不允许出现text中包含media token但是不仅仅是media token。 media token必须单独为一个chunk
            assert has_media_flag, text
        return has_media_flag

    def add_media(self, text_chunk, text=None, tokenize_fn=None):
        # cross
        assert tokenize_fn is not None
        assert text is not None
        assert text in OWL_MEDIA_TOKEN
        media_token_ids = tokenize_fn(text)
        start = len(text_chunk)
        end = start + len(media_token_ids)
        self.media_position.append([start, end])
        text_chunk.extend(media_token_ids)
        return len(media_token_ids)

    def cal_media_offset(self, input_ids):
        if len(self.media_position) == 0:
            return paddle.ones_like(input_ids) * (-1000000)

        media_starts = paddle.to_tensor([_[0] for _ in self.media_position]).reshape([1, -1])
        rng = paddle.arange(input_ids.shape[0]).reshape([-1, 1])
        matrix = (rng > media_starts).sum(axis=1)

        return matrix

    def len_images(
        self,
    ):
        return len(self.media_position)


class mPLUGOwl3Processor(ProcessorMixin):
    r"""
    Args:
        image_processor ([`mPLUGOwl3ImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerWrapper`], *optional*):
            The tokenizer is a required input.
    """
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "mPLUGOwl3ImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor: mPLUGOwl3ImageProcessor = None,
        tokenizer=None,
        prompt_style="chatml",
        inference_mode=True,
        addition_eod="<|endoftext|>",
    ):
        super().__init__(image_processor, tokenizer)
        self.image_processor: mPLUGOwl3ImageProcessor
        self.prompt_style = prompt_style
        self.inference_mode = inference_mode
        self.media_tokens = ["<|image|>"]
        self.addition_eod = addition_eod

    def build_text_qwen(self, messages):
        # role should be within ['system', 'user', 'assistant']
        im_start, im_end = "<|im_start|>", "<|im_end|>"

        text = []
        for num_turn, message in enumerate(messages):
            if num_turn == 0 and message["role"] != "system":
                if self.prompt_style != "plain":
                    text.append({"text": f"{im_start}system\n{im_end}", "label": 0})
            if message["role"] == "system":
                if self.prompt_style != "plain":
                    text.append({"text": f"{im_start}system\n{message['content']}{im_end}", "label": 0})
            elif message["role"] == "user":
                if self.prompt_style != "plain":
                    content = f"\n{im_start}user\n{message['content']}{im_end}"
                else:
                    content = message["content"]
                pattern = "|".join(map(re.escape, self.media_tokens))
                chunk_strs = re.split(f"({pattern})", content)
                for chunk_str in chunk_strs:
                    text.append({"text": chunk_str, "label": 0})

            elif message["role"] == "assistant":
                if self.prompt_style != "plain":
                    text.append({"text": f"\n{im_start}assistant\n", "label": 0})
                    text.append({"text": f"{message['content']}{im_end}", "label": 1})
                else:
                    text.append({"text": f"{message['content']}", "label": 1})
                text.append({"text": self.addition_eod, "label": 1})
            else:
                raise NotImplementedError
        if self.inference_mode:
            while text and text[-1]["label"] == 1:  # 只要列表非空且最后一个元素满足条件
                text.pop()  # 就移除最后一个元素
        return text

    def wrapped_tokenize(self, text):
        return self.tokenizer(text).input_ids

    def encode_text_sft(self, texts):
        # output enc_chunk

        enc_chunk = []
        label_chunk = []
        enc_length = 0

        num_images = 0

        media_helper = MediaIndicesHelper(tokenizer=self.tokenizer)
        for current_ti, text_chunk in enumerate(texts):

            text = text_chunk["text"]
            label = text_chunk["label"]

            if not media_helper.has_media(text):
                curr_chunk = self.wrapped_tokenize(text)
                if label == 1:
                    enc_length += len(curr_chunk)
                    enc_chunk += curr_chunk
                    label_chunk += [label] * len(curr_chunk)
                else:

                    enc_length += len(curr_chunk)
                    enc_chunk += curr_chunk
                    label_chunk += [label] * len(curr_chunk)
            # For media tokens
            else:

                add_length = media_helper.add_media(enc_chunk, text=text, tokenize_fn=self.wrapped_tokenize)
                enc_length += add_length
                label_chunk += [label] * add_length
                # enc_chunk.extend([self.media_tokens[text]] * self.media_lengths[text])
                # enc_length += self.media_lengths[text]
                # label_chunk += [label] * self.media_lengths[text]
                num_images += 1

        enc_chunk = paddle.to_tensor(enc_chunk).astype(dtype="int64")
        media_offset = [paddle.to_tensor([_[0] for _ in media_helper.media_position]).astype(dtype="int64")]
        return {
            "input_ids": enc_chunk.unsqueeze(0),
            "media_offset": media_offset,
        }

    def __call__(
        self,
        messages,
        images=None,
        videos=None,
        max_length: Optional[int] = None,
        cut_enable=True,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PADDLE,
        **kwargs
    ) -> mPLUGOwl3BatchFeature:
        medias = []
        if videos is not None:
            medias.extend([{"type": "video", "content": video, "use_video_span": True} for video in videos])
        if images is not None:
            medias.extend([{"type": "image", "content": image} for image in images])

        if len(medias):
            image_tensor_list = []
            pattern = r"(<\|image\|>|<\|video\|>)"
            # 存在媒体
            image_token_ptr = 0
            # media_layout = []
            for message in messages:
                text_list = re.split(pattern, message["content"])
                text = ""
                for text_content in text_list:
                    if text_content in ["<|image|>", "<|video|>"]:
                        media_item = medias[image_token_ptr]
                        image_token_ptr += 1
                        if text_content == "<|image|>":
                            assert media_item["type"] == "image"
                            image = media_item["content"]

                            image_inputs = self.image_processor(
                                [image], cut_enable=cut_enable, return_tensors=return_tensors
                            )
                            if image_inputs.get("cut_shape", None) is not None:
                                cut_shape = image_inputs["cut_shape"]
                                cut_text = self.image_processor.cut_prompt_template(
                                    img_token="<|image|>", h=cut_shape[0][0], w=cut_shape[0][1]
                                )
                                text += cut_text
                                image_tensor_list.append(image_inputs["pixel_values"])
                            else:
                                text += text_content
                                image_tensor_list.append(image_inputs["pixel_values"])
                        elif text_content == "<|video|>":
                            assert media_item["type"] == "video"
                            video = media_item["content"]
                            use_video_span = media_item["use_video_span"]
                            image_tensor = self.image_processor(video, cut_enable=False)["pixel_values"]
                            image_tensor_list.append(image_tensor)
                            num_video_frame = image_tensor.shape[0]
                            if use_video_span:
                                text_content = (
                                    "<|start_video_frame|>" + "<|image|>" * num_video_frame + "<|end_video_frame|>"
                                )
                            else:
                                text_content = "<|image|>" * num_video_frame
                            text += text_content
                    else:
                        text += text_content
                message["content"] = text
            assert image_token_ptr == len(medias), (image_token_ptr, len(medias))  # 保证图和token数目一致
            assert all(len(_.shape) == 4 for _ in image_tensor_list), [_.shape for _ in image_tensor_list]
            num_image_tokens = sum([_["content"].count("<|image|>") for _ in messages])
            num_image_shapes = sum([_.shape[0] for _ in image_tensor_list])
            assert num_image_tokens == num_image_shapes, (messages, [_.shape for _ in image_tensor_list])

        image_tensor_list = paddle.concat(image_tensor_list, axis=0)

        text = self.build_text_qwen(messages)
        model_inputs = self.encode_text_sft(text)

        if len(medias) is not None:
            model_inputs.update({"pixel_values": image_tensor_list})
            # if 'cut_shape' in model_inputs:
            #     model_inputs.pop('cut_shape')
            # if 'cut_shape_indices' in model_inputs:
            #     model_inputs.pop('cut_shape_indices')
        return mPLUGOwl3BatchFeature(model_inputs)

    def check_media(self, images, messages):
        media_num = 0 if images is None else len(images)
        media_count = sum([message["content"].count("<|image|>") for message in messages])
        assert media_num == media_count

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        output_ids = args[0]
        result_text = []
        for result in output_ids:
            result = result[result != 0]
            if result[0] == self.tokenizer.bos_id:
                result = result[1:]
            if result[-1] == self.tokenizer.eos_id:
                result = result[:-1]
            result_text.append(self.tokenizer.decode(result, *args[1:], **kwargs).strip())
        return result_text
        # return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        result = args[0]
        result = result[result != 0]
        if result[0] == self.tokenizer.bos_id:
            result = result[1:]
        if result[-1] == self.tokenizer.eos_id or (
            hasattr(self.tokenizer, "eot_id") and result[-1] == self.tokenizer.eot_id
        ):
            result = result[:-1]
        return self.tokenizer.decode(result, *args[1:], **kwargs).strip()

    def _convert(self, input_str, max_inp_length: Optional[int] = None):
        if self.version > 2.5 or not getattr(self.tokenizer, "add_bos_token", False):
            input_ids = self.tokenizer.encode(input_str)
        else:
            input_ids = [self.tokenizer.bos_id] + self.tokenizer.encode(input_str)
        if max_inp_length is not None:
            input_ids = input_ids[:max_inp_length]
        input_ids = paddle.to_tensor(data=input_ids, dtype="int32")

        start_cond = (input_ids == self.tokenizer.im_start_id) | (input_ids == self.tokenizer.slice_start_id)
        end_cond = (input_ids == self.tokenizer.im_end_id) | (input_ids == self.tokenizer.slice_end_id)

        image_start_tokens = paddle.where(start_cond)[0]  # or paddle.nonzero(start_cond)[:, 0]
        image_start_tokens += 1
        image_end_tokens = paddle.where(end_cond)[0]

        valid_image_nums = max(len(image_start_tokens), len(image_end_tokens))

        image_bounds = paddle.hstack(
            [
                image_start_tokens[:valid_image_nums].unsqueeze(-1),
                image_end_tokens[:valid_image_nums].unsqueeze(-1),
            ]
        )
        return input_ids, image_bounds

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def pad(self, inputs, max_length=None, padding_value=0, padding_side="left"):
        items = []
        if isinstance(inputs[0], list):
            assert isinstance(inputs[0][0], paddle.Tensor)
            for it in inputs:
                for tr in it:
                    items.append(tr)
        else:
            assert isinstance(inputs[0], paddle.Tensor)
            items = inputs

        batch_size = len(items)
        shape = items[0].shape
        dim = len(shape)
        assert dim <= 2
        if max_length is None:
            max_length = 0
        max_length = max(max_length, max(item.shape[-1] for item in items))
        min_length = min(item.shape[-1] for item in items)
        dtype = items[0].dtype

        if dim == 0:
            return paddle.stack([item for item in items], axis=0), [0]
        elif dim == 1:
            if max_length == min_length:
                return paddle.stack([item for item in items], axis=0), [0] * batch_size
            tensor = paddle.zeros((batch_size, max_length), dtype=dtype) + padding_value
        else:
            tensor = paddle.zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value

        padding_length = []
        for i, item in enumerate(items):
            if dim == 1:
                if padding_side == "left":
                    tensor[i, -len(item) :] = item.clone()
                else:
                    tensor[i, : len(item)] = item.clone()
            elif dim == 2:
                if padding_side == "left":
                    tensor[i, -len(item) :, :] = item.clone()
                else:
                    tensor[i, : len(item), :] = item.clone()
            padding_length.append(tensor.shape[-1] - len(item))

        return tensor, padding_length

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
"""
Processor class for GroundingDino.
"""

from typing import Dict, List, Optional, Union

import paddle
import paddle.vision.transforms as T
from paddlenlp.taskflow.utils import pad_batch_data

from .base_processing import ProcessorMixin
from .image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, valid_images
from .processing_utils import BaseImageProcessor, BaseTextProcessor
from .utils import _max_by_axis

__all__ = [
    "GroundingDinoProcessor",
    "GroundingDinoImageProcessor",
    "GroundingDinoTextProcessor",
]


class GroundingDinoProcessor(ProcessorMixin):

    attributes = ["image_processor", "text_processor", "tokenizer"]
    image_processor_class = "GroundingDinoImageProcessor"
    text_processor_class = "GroundingDinoTextProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, text_processor, tokenizer):
        super().__init__(image_processor, text_processor, tokenizer)

    def __call__(
        self,
        images=None,
        text: str = None,
        **kwargs,
    ):

        if images is None or text is None:
            raise ValueError("You have to specify either images and text.")

        self.prompt = self.text_processor.pre_caption(text)
        input_ids = self.tokenizer([self.prompt]).input_ids
        special_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
        tokenized_out = self.text_processor(input_ids, special_tokens)

        image_tensor, mask = self.image_processor(images)

        return image_tensor, mask, tokenized_out

    def decode(self, posmap):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        assert isinstance(posmap, paddle.Tensor), "posmap must be paddle.Tensor"
        tokenized = self.tokenizer(self.prompt)
        if posmap.dim() == 1:
            non_zero_idx = posmap.nonzero(as_tuple=True)[0].squeeze(-1).tolist()
            token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
            return self.tokenizer.decode(token_ids)
        else:
            raise NotImplementedError("posmap must be 1-dim")

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


class GroundingDinoTextProcessor(BaseTextProcessor):
    r"""
    Constructs a GroundingDino text processor.
    """

    def __init__(
        self,
        max_words: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.max_words = max_words
        self.caption = None

    def __call__(
        self,
        input_ids,
        special_tokens_list,
        **kwargs,
    ):
        """
        Preprocess the text with tokenization.
        """
        tokenized_out = {}
        input_ids = pad_batch_data(input_ids)
        input_ids = paddle.to_tensor(input_ids, dtype=paddle.int64).squeeze(-1)
        tokenized_out["input_ids"] = input_ids
        tokenized_out["attention_mask"] = paddle.cast(input_ids != 0, paddle.int64)

        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = self.generate_masks_with_special_tokens_and_transfer_map(tokenized_out, special_tokens_list)

        if text_self_attention_masks.shape[1] > self.max_words:
            text_self_attention_masks = text_self_attention_masks[:, : self.max_words, : self.max_words]
            position_ids = position_ids[:, : self.max_words]
            tokenized_out["input_ids"] = tokenized_out["input_ids"][:, : self.max_words]
            tokenized_out["attention_mask"] = tokenized_out["attention_mask"][:, : self.max_words]
        tokenized_out["position_ids"] = position_ids
        tokenized_out["text_self_attention_masks"] = text_self_attention_masks

        return tokenized_out

    def pre_caption(self, caption: str) -> str:
        """
        Preprocess the text before tokenization.
        """
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        self.caption = caption
        return caption

    def generate_masks_with_special_tokens_and_transfer_map(self, tokenized, special_tokens_list):
        """Generate attention mask between each pair of special tokens
        Args:
            input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
            special_tokens_mask (list): special tokens mask.
        Returns:
            torch.Tensor: attention mask between each special tokens.
        """
        input_ids = tokenized["input_ids"]
        bs, num_token = input_ids.shape
        # special_tokens_mask: bs, num_token. 1 for special tokens. 0 for normal tokens
        special_tokens_mask = paddle.zeros((bs, num_token), dtype=paddle.bool)
        for special_token in special_tokens_list:
            special_tokens_mask |= input_ids == special_token

        # idxs: each row is a list of indices of special tokens
        idxs = paddle.nonzero(special_tokens_mask)

        # generate attention mask and positional ids
        attention_mask = paddle.eye(num_token, dtype=paddle.int32).cast(paddle.bool).unsqueeze(0).tile([bs, 1, 1])
        position_ids = paddle.zeros((bs, num_token), dtype=paddle.int64)
        cate_to_token_mask_list = [[] for _ in range(bs)]
        previous_col = 0

        for i in range(idxs.shape[0]):
            row, col = idxs[i]
            if (col == 0) or (col == num_token - 1):
                attention_mask[row, col, col] = True
                position_ids[row, col] = 0
            else:
                attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
                position_ids[row, previous_col + 1 : col + 1] = paddle.arange(0, col - previous_col)
                c2t_maski = paddle.zeros(
                    [
                        num_token,
                    ]
                ).cast(paddle.bool)
                c2t_maski[previous_col + 1 : col] = True
                cate_to_token_mask_list[row].append(c2t_maski)
            previous_col = col

        return attention_mask, position_ids.cast(paddle.int64), cate_to_token_mask_list


class GroundingDinoImageProcessor(BaseImageProcessor):
    r"""
    Constructs a GroundingDino image processor.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: List[int] = None,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_nested: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else 800

        self.do_resize = do_resize
        self.size = size
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_nested = do_nested

    def resize(self, image, target=None, size=None, max_size=1333):
        def get_size_with_aspect_ratio(image_size, size, max_size=None):
            w, h = image_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)

        def get_size(image_size, size, max_size=None):
            if isinstance(size, (list, tuple)):
                return size[::-1]
            else:
                return get_size_with_aspect_ratio(image_size, size, max_size)

        size = get_size(image.size, size, max_size)
        rescaled_image = T.resize(image, size)

        if target is None:
            return rescaled_image

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
        ratio_width, ratio_height = ratios

        target = target.copy()
        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * paddle.to_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
            target["boxes"] = scaled_boxes

        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        h, w = size
        target["size"] = paddle.to_tensor([h, w])

        # if "masks" in target:
        #     target["masks"] = (
        #         interpolate(target["masks"][:, None].cast(paddle.float32), size, mode="nearest")[:, 0] > 0.5
        #     )

        return rescaled_image, target

    def nested_tensor_from_tensor_list(self, tensor_list: List[paddle.Tensor]):
        # TODO make this more general
        if tensor_list[0].ndim == 3:

            # TODO make it support different-sized images
            max_size = _max_by_axis([list(img.shape) for img in tensor_list])
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = [len(tensor_list)] + max_size
            b, c, h, w = batch_shape
            dtype = tensor_list[0].dtype
            tensor = paddle.zeros(batch_shape, dtype=dtype)
            mask = paddle.ones((b, h, w), dtype=paddle.bool)
            for i in range(b):
                img = tensor_list[i]
                tensor[i, : img.shape[0], : img.shape[1], : img.shape[2]] = img
                mask[i, : img.shape[1], : img.shape[2]] = False
        else:
            raise ValueError("not supported")
        return tensor, mask

    def preprocess(
        self,
        images,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_nested: bool = None,
        **kwargs,
    ):
        """
        Preprocess an image or batch of images.

        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_nested = do_nested if do_nested is not None else self.do_nested

        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        size = size if size is not None else self.size

        if not isinstance(images, (list, tuple)):
            images = [images]

        # if isinstance(images[0], str):
        #     images = [load_image(image) for image in images]

        if not valid_images(images):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, " "paddle.Tensor.")

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        if do_resize:
            images = [T.to_tensor(self.resize(image=image, size=size)) for image in images]

        if do_normalize:
            images = T.normalize(images, mean=image_mean, std=image_std)

        if do_nested:
            tensors, masks = self.nested_tensor_from_tensor_list(images)

        return tensors, masks

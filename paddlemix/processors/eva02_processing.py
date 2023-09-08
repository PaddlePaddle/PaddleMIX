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
Processor class for EVA-02.
"""
from typing import Dict, List, Optional, Tuple, Union

import paddle
import PIL
from paddlenlp.transformers.tokenizer_utils_base import BatchEncoding, TensorType

from .base_processing import ProcessorMixin
from .eva02_transforms import DataAugmentationForEVA, create_transform
from .image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
)
from .processing_utils import BaseImageProcessor

__all__ = [
    "EVA02Processor",
    "EVA02PretrainImageProcessor",
    "EVA02FinetuneImageProcessor",
]


class EVA02Processor(ProcessorMixin):

    attributes = ["image_processor"]
    image_processor_class = "EVA02FinetuneImageProcessor"

    def __init__(self, image_processor):
        super().__init__(image_processor)

    def __call__(
        self,
        images=None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        mode="train",
        **kwargs,
    ) -> BatchEncoding:
        if images is None:
            raise ValueError("You have to specify images")

        # images PIL list
        encoding_image_processor = self.image_processor(images, return_tensors=return_tensors, mode=mode)
        return encoding_image_processor

    @property
    def model_input_names(self):
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(image_processor_input_names))


class EVA02PretrainImageProcessor(BaseImageProcessor):

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        input_size: Dict[str, int] = None,
        second_input_size: Dict[str, int] = None,
        crop_scale: Optional[Union[List[float], Tuple[float]]] = [0.2, 1.0],
        crop_ratio: Optional[Union[List[float], Tuple[float]]] = [3.0 / 4.0, 4.0 / 3.0],
        interpolation: PILImageResampling = PILImageResampling.BICUBIC,
        second_interpolation: PILImageResampling = PILImageResampling.BICUBIC,
        color_jitter: float = 0.0,
        window_size: float = None,
        num_mask_patches: int = 105,
        max_mask_patches_per_block: int = None,
        min_mask_patches_per_block: int = 16,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        mode: str = "train",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size = input_size
        self.second_input_size = second_input_size
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.interpolation = interpolation
        self.second_interpolation = second_interpolation
        self.color_jitter = color_jitter
        self.window_size = window_size
        self.num_mask_patches = num_mask_patches
        self.max_mask_patches_per_block = max_mask_patches_per_block
        self.min_mask_patches_per_block = min_mask_patches_per_block
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.mode = mode

    def preprocess(
        self,
        images: ImageInput,
        input_size: Optional[Dict[str, int]] = None,
        second_input_size: Optional[Dict[str, int]] = None,
        crop_scale: Optional[Union[List[float], Tuple[float]]] = None,
        crop_ratio: Optional[Union[List[float], Tuple[float]]] = None,
        interpolation: PILImageResampling = None,
        second_interpolation: PILImageResampling = None,
        color_jitter: Optional[float] = None,
        window_size: Optional[float] = None,
        num_mask_patches: Optional[float] = None,
        max_mask_patches_per_block: Optional[float] = None,
        min_mask_patches_per_block: Optional[float] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        mode: str = None,
        **kwargs,
    ) -> PIL.Image.Image:
        input_size = input_size if input_size is not None else self.input_size
        second_input_size = second_input_size if second_input_size is not None else self.second_input_size
        crop_scale = crop_scale if crop_scale is not None else self.crop_scale
        crop_ratio = crop_ratio if crop_ratio is not None else self.crop_ratio
        interpolation = interpolation if interpolation is not None else self.interpolation
        second_interpolation = second_interpolation if second_interpolation is not None else self.second_interpolation
        color_jitter = color_jitter if color_jitter is not None else self.color_jitter
        window_size = window_size if window_size is not None else self.window_size
        num_mask_patches = num_mask_patches if num_mask_patches is not None else self.num_mask_patches
        max_mask_patches_per_block = (
            max_mask_patches_per_block if max_mask_patches_per_block is not None else self.max_mask_patches_per_block
        )
        min_mask_patches_per_block = (
            min_mask_patches_per_block if min_mask_patches_per_block is not None else self.min_mask_patches_per_block
        )
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        mode = mode if mode is not None else self.mode

        transform = DataAugmentationForEVA(
            input_size=input_size,
            second_input_size=second_input_size,
            crop_scale=crop_scale,
            crop_ratio=crop_ratio,
            interpolation=interpolation,
            second_interpolation=second_interpolation,
            color_jitter=color_jitter,
            window_size=window_size,
            num_mask_patches=num_mask_patches,
            max_mask_patches_per_block=max_mask_patches_per_block,
            min_mask_patches_per_block=min_mask_patches_per_block,
            image_mean=image_mean,
            image_std=image_std,
        )

        samples, image, bool_masked_pos = [], [], []
        for im in images:
            patch, vis_token, mask_pos = transform(im)
            samples.append(patch.unsqueeze(0))
            image.append(vis_token.unsqueeze(0))
            bool_masked_pos.append(paddle.to_tensor(mask_pos).unsqueeze(0))

        samples = paddle.concat(samples, 0)
        image = paddle.concat(image, 0)
        bool_masked_pos = paddle.concat(bool_masked_pos, 0)
        return {"samples": samples, "image": image, "bool_masked_pos": bool_masked_pos}


class EVA02FinetuneImageProcessor(BaseImageProcessor):

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        input_size: Dict[str, int] = None,
        no_aug: bool = False,
        color_jitter: float = 0.4,
        auto_augment: str = "rand-m9-mstd0.5-inc1",
        interpolation: PILImageResampling = PILImageResampling.BICUBIC,
        reprob: float = 0,
        re_mode: str = "pixel",
        recount: int = 1,
        scale: Optional[Union[List[float], Tuple[float]]] = (0.08, 1.0),
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        crop_pct: float = 1,
        mode: str = "train",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size = input_size
        self.no_aug = no_aug
        self.color_jitter = color_jitter
        self.auto_augment = auto_augment
        self.interpolation = interpolation
        self.reprob = reprob
        self.re_mode = re_mode
        self.recount = recount
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.scale = scale
        self.crop_pct = crop_pct
        self.mode = mode

    def preprocess(
        self,
        images: ImageInput,
        input_size: Optional[Dict[str, int]] = None,
        no_aug: Optional[bool] = None,
        color_jitter: Optional[float] = None,
        auto_augment: Optional[str] = None,
        interpolation: PILImageResampling = None,
        reprob: Optional[float] = None,
        re_mode: Optional[str] = None,
        recount: Optional[float] = None,
        scale: Optional[Union[float, List[float]]] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        crop_pct: Optional[float] = None,
        mode: str = None,
        **kwargs,
    ) -> PIL.Image.Image:
        input_size = input_size if input_size is not None else self.input_size
        no_aug = no_aug if no_aug is not None else self.no_aug
        color_jitter = color_jitter if color_jitter is not None else self.color_jitter
        interpolation = interpolation if interpolation is not None else self.interpolation
        reprob = reprob if reprob is not None else self.reprob
        re_mode = re_mode if re_mode is not None else self.re_mode
        recount = recount if recount is not None else self.recount
        scale = scale if scale is not None else self.scale
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        crop_pct = crop_pct if crop_pct is not None else self.crop_pct
        mode = mode if mode is not None else self.mode

        resize_im = input_size > 32

        if mode == "train":
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=input_size,
                is_training=True,
                use_prefetcher=False,
                no_aug=no_aug,
                scale=scale,
                ratio=None,
                hflip=0.5,
                vflip=0.0,
                color_jitter=color_jitter,
                auto_augment=auto_augment,
                interpolation=interpolation,
                mean=image_mean,
                std=image_std,
                re_prob=reprob,
                re_mode=re_mode,
                re_count=recount,
                re_num_splits=0,
                crop_pct=None,
                crop_mode=None,
                tf_preprocessing=False,
                separate=False,
            )
            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = paddle.vision.transforms.RandomCrop(input_size, padding=4)
        else:
            t = []
            if resize_im:
                if crop_pct is None:
                    if input_size < 384:
                        crop_pct = 224 / 256
                    else:
                        crop_pct = 1.0
                size = int(input_size / crop_pct)
                t.append(
                    paddle.vision.transforms.Resize(
                        size, interpolation=interpolation
                    ),  # to maintain same ratio w.r.t. 224 images
                )
                t.append(paddle.vision.transforms.CenterCrop(input_size))

            t.append(paddle.vision.transforms.ToTensor())
            t.append(paddle.vision.transforms.Normalize(image_mean, image_std))
            transform = paddle.vision.transforms.Compose(t)

        inputs = []
        for inp in images:
            inputs.append(transform(inp).unsqueeze(0))
        inputs = paddle.concat(inputs, 0)
        return {"image": inputs}

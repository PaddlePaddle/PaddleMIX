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
Processor class for BLIP-2.
"""

import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
from paddlenlp.transformers.tokenizer_utils_base import (
    BatchEncoding,
    PreTokenizedInput,
    TensorType,
    TextInput,
)

from .base_processing import ProcessorMixin
from .image_transform_utils import (
    convert_to_rgb,
    normalize,
    random_horizontal_flip,
    random_resized_crop,
    rescale,
    resize,
    to_channel_dimension_format,
)
from .image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    load_image,
    to_numpy_array,
    valid_images,
)
from .processing_utils import BaseImageProcessor, BaseTextProcessor, get_size_dict

__all__ = [
    "Blip2Processor",
    "BlipImageProcessor",
    "BlipTextProcessor",
]


class Blip2Processor(ProcessorMixin):
    r"""
    Constructs a BLIP-2 processor which wraps a BLIP2 image processor and an OPT/T5 tokenizer into a single processor.
    [`Blip2Processor`] offers all the functionalities of [`BlipImageProcessor`] and [`AutoTokenizer`]. See the docstring
    of [`~Blip2Processor.__call__`] and [`~Blip2Processor.decode`] for more information.
    Args:
        image_processor (`BlipImageProcessor`):
            An instance of [`BlipImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
    """
    attributes = ["image_processor", "text_processor", "tokenizer"]
    image_processor_class = "BlipImageProcessor"
    text_processor_class = "BlipTextProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, text_processor, tokenizer):
        super().__init__(image_processor, text_processor, tokenizer)

    def __call__(
        self,
        images=None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        max_length=32,
        mode="train",
        **kwargs,
    ) -> BatchEncoding:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Bert's [`~BertTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to
        Blip2ImageProcessor's [`~Blip2ImageProcessor.__call__`] if `images` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:

            images (`PIL.Image.Image`, `np.ndarray`, `paddle.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[paddle.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or Paddle
                tensor. In case of a NumPy array/Paddle tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'pd'`: Return Paddle `paddle.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
            max_length (`int`, *optional*):
                If set to a number, will limit the total sequence returned so
                that it has a maximum length.
            mode (`str`, *optional*):
                The mode of ("train", "val", "test")

        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if images is None and text is None:
            raise ValueError("You have to specify either images or text.")

        # Get only text
        if images is None:
            text_encoding = self.text_processor(text, mode=mode)

            text_encoding = self.tokenizer(
                text=text_encoding,
                return_tensors=return_tensors,
                return_token_type_ids=False,
                max_length=32,
                padding="longest",
                **kwargs,
            )
            return text_encoding

        # add pixel_values
        encoding_image_processor = self.image_processor(images, return_tensors=return_tensors, mode=mode)

        if text is not None:
            if "t5" not in self.tokenizer.name_or_path:
                text_encoding = self.text_processor(text, mode=mode)
                text_encoding = self.tokenizer(
                    text=text_encoding,
                    return_tensors="pd",
                    padding="longest",
                    truncation=True,
                    max_length=max_length,
                    return_attention_mask=True,
                )
            else:
                text_encoding = self.text_processor(text, mode=mode)
                input_encoding = self.tokenizer(
                    text=text_encoding["input"],
                    return_tensors="pd",
                    padding="longest",
                    truncation=True,
                    max_length=max_length,
                    return_attention_mask=True,
                )
                output_encoding = self.tokenizer(
                    text=text_encoding["output"],
                    return_tensors="pd",
                    padding="longest",
                    truncation=True,
                    max_length=max_length,
                    return_attention_mask=True,
                )
                text_encoding = input_encoding
                text_encoding["decoder_input_ids"] = output_encoding.input_ids
                text_encoding["decoder_attention_mask"] = output_encoding.attention_mask

        else:
            text_encoding = None
            # eos_token_id = None

        if text_encoding is not None:
            encoding_image_processor.update(text_encoding)

        return encoding_image_processor

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


class BlipTextProcessor(BaseTextProcessor):
    r"""
    Constructs a BLIP text processor.

    Args:
        prompt(`str`, *optional*, defaults to `""`):
            The prompt (used for generating prompts) that will be prepended to each generated text.
        do_caption (`bool`, *optional*, defaults to `False`):
            Whether to do the caption task.
        do_question(`bool`, *optional*, defaults to `False`):
            Whether to do the question task.
        max_words (`int`, *optional*, defaults to `50`):
            The maximum number of words to keep in the span of text.

    """

    def __init__(
        self,
        prompt: str = "",
        do_caption: bool = False,
        do_question: bool = False,
        max_words: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if do_question and do_caption:
            raise ValueError("do_caption and do_question cannot be set at the same time.")
        if not do_caption and not do_question:
            raise ValueError("Either do_caption or do_question must be set to True.")
        self.prompt = prompt
        self.do_caption = do_caption
        self.do_question = do_question
        self.max_words = max_words

    def __call__(
        self,
        text,
        do_caption: Optional[bool] = None,
        do_question: Optional[bool] = None,
        mode: str = "train",
        **kwargs,
    ):
        """
        Preprocess the text before tokenization.

        Args:
            text (`str`):
                Text to preprocess.
            do_caption (`bool`, *optional*, defaults to `False`):
                Whether to do the caption task.
            do_question(`bool`, *optional*, defaults to `False`):
                Whether to do the question task.
            mode(`str`, *optional*, defaults to `train`):
                The mode of ("train", "val", "test")

        """
        do_caption = do_caption if do_caption is not None else self.do_caption
        do_question = do_question if do_question is not None else self.do_question
        if do_caption and do_question:
            raise ValueError("do_caption and do_question cannot be set at the same time.")
        if not do_caption and not do_question:
            raise ValueError("Either do_caption or do_question must be set to True.")

        if not isinstance(text, (list, tuple)):
            text = [text]
        # import pdb; pdb.set_trace()
        if do_caption:
            results = [self.prompt + self.pre_caption(t) for t in text]
        if do_question:
            results = [self.prompt.format(self.pre_question(t)) for t in text]
        if mode == "train":
            results = [res + "\n" for res in results]
        return results

    def pre_caption(self, caption: str) -> str:
        """
        Preprocess the text before tokenization.
        """
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption

    def pre_question(self, question: str) -> str:
        """
        Preprocess the text before tokenization.
        """
        question = re.sub(
            r"([.!\"()*#:;~])",
            "",
            question.lower(),
        )
        question = question.rstrip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > self.max_words:
            question = " ".join(question_words[: self.max_words])

        return question


class BlipImageProcessor(BaseImageProcessor):
    r"""
    Constructs a BLIP image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 384, "width": 384}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        do_rand_resize_crop (`bool`, *optional*, defaults to `False`):
            Whether to *randomly crop* the image at random in the height and width dimensions.
        rand_resize_crop_prob (`float`, *optional*, defaults to `0.5`):
            Probability of applying a random crop to the image.
        scale (`list|tuple`, *optional*, defaults to `(0.08, 1.0)`):
            Scale range of the cropped image before resizing, relatively to the origin image.
        mode (`str`, *optional*):
                The mode of ("train", "val", "test")
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        do_flip: bool = False,
        flip_prob: float = 0.5,
        do_rand_resize_crop: bool = False,
        scale: Optional[Union[List[float], Tuple[float]]] = (0.08, 1.0),
        do_collate: bool = False,
        mode: str = "train",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 384, "width": 384}
        size = get_size_dict(size, default_to_square=True)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_convert_rgb = do_convert_rgb
        self.do_flip = do_flip
        self.flip_prob = flip_prob
        self.do_rand_resize_crop = do_rand_resize_crop
        self.scale = scale
        self.do_collate = do_collate

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image.

        Resizes the shorter side of the image to `size["shortest_edge"]` while preserving the aspect ratio. If the
        longer side is larger than the max size `(int(`size["shortest_edge"]` * 1333 / 800))`, the longer side is then
        resized to the max size while preserving the aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Controls the size of the output image. Should be of the form `{"shortest_edge": int}`.
            resample (`PILImageResampling` filter, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        size = get_size_dict(size, default_to_square=True)
        output_size = (size["width"], size["height"])
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            **kwargs,
        )

    def rescale(
        self,
        image: np.ndarray,
        scale: Union[int, float],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Rescale an image by a scale factor. image = image * scale.

        Args:
            image (`np.ndarray`):
                Image to rescale.
            scale (`int` or `float`):
                Scale to apply to the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        return rescale(image, scale=scale, data_format=data_format, **kwargs)

    def normalize(
        self,
        image: np.ndarray,
        mean: Union[float, List[float]],
        std: Union[float, List[float]],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Normalize an image. image = (image - image_mean) / image_std.

        Args:
            image (`np.ndarray`):
                Image to normalize.
            mean (`float` or `List[float]`):
                Image mean.
            std (`float` or `List[float]`):
                Image standard deviation.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        return normalize(image, mean=mean, std=std, data_format=data_format, **kwargs)

    def random_resized_crop(
        self,
        image: np.ndarray,
        size: Union[int, List, Tuple],
        scale: float,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        **kwargs,
    ) -> np.ndarray:
        """
        Crop the input data to random size and aspect ratio.
        A crop of random size (default: of 0.08 to 1.0) of the original size and a random
        aspect ratio (default: of 3/4 to 1.33) of the original aspect ratio is made.
        After applying crop transform, the input data will be resized to given size.

        Args:
            image (`np.ndarray`):
                Image to resize to and crop.
            size (Union[int, List, Tuple]):
                Size of cropped image.
            scale (`float`):
                Scale to apply to the image.
            resample (`PILImageResampling` filter, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resizing the image.
        """
        size = list(size.values())
        return random_resized_crop(image, size=size, scale=scale, resample=resample, **kwargs)

    def random_horizontal_flip(self, image: np.ndarray, flip_prob: float, **kwargs) -> np.ndarray:
        """
        Horizontally flip the input data randomly with a given probability.

        Args:
        image (`np.ndarray`):
            Image to flip.
        flip_prob (`float`):
            Probability of flipping the image.
        """
        return random_horizontal_flip(image, flip_prob=flip_prob, **kwargs)

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        do_convert_rgb: bool = None,
        do_flip: bool = None,
        flip_prob: float = None,
        do_rand_resize_crop: bool = None,
        scale: Optional[Union[List[float], Tuple[float]]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        mode: str = None,
        **kwargs,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Controls the size of the image after `resize`. The shortest edge of the image is resized to
                `size["shortest_edge"]` whilst preserving the aspect ratio. If the longest edge of this resized image
                is > `int(size["shortest_edge"] * (1333 / 800))`, then the image is resized again to make the longest
                edge equal to `int(size["shortest_edge"] * (1333 / 800))`.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to normalize the image by if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            do_rand_resize_crop (`bool`, *optional*, defaults to `False`):
                Whether to *randomly crop* the image at random in the height and width dimensions.
            scale (`list|tuple`, *optional*, defaults to `(0.08, 1.0)`):
                Scale range of the cropped image before resizing, relatively to the origin image.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.PADDLE` or `'pt'`: Return a batch of type `paddle.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: defaults to the channel dimension format of the input image.
            mode (`str`, *optional*):
                The mode of ("train", "val", "test")
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        do_flip = do_flip if do_flip is not None else self.do_flip
        flip_prob = flip_prob if flip_prob is not None else self.flip_prob
        scale = scale if scale is not None else self.scale
        do_rand_resize_crop = do_rand_resize_crop if do_rand_resize_crop is not None else self.do_rand_resize_crop

        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)

        if not isinstance(images, (list, tuple)):
            images = [images]

        if isinstance(images[0], str):
            images = [load_image(image) for image in images]

        if not valid_images(images):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, " "paddle.Tensor.")

        if do_resize and size is None or resample is None:
            raise ValueError("Size and resample must be specified if do_resize is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        if do_flip and flip_prob is None:
            raise ValueError("Flip probability must be specified if do_flip is True.")

        if do_rand_resize_crop and scale is None:
            raise ValueError("Random resize crop probability must be specified if do_rand_resize_crop is True.")

        # PIL RGBA images are converted to RGB
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]
        if do_rand_resize_crop and mode == "train":
            images = [
                self.random_resized_crop(image=image, size=size, scale=scale, resample=resample) for image in images
            ]
        elif do_resize and mode != "train":
            images = [self.resize(image=image, size=size, resample=resample) for image in images]

        if do_flip and mode == "train":
            images = [self.random_horizontal_flip(image=image, flip_prob=flip_prob) for image in images]

        if do_rescale:
            images = [self.rescale(image=image, scale=rescale_factor) for image in images]

        if do_normalize:
            images = [self.normalize(image=image, mean=image_mean, std=image_std) for image in images]

        images = [to_channel_dimension_format(image, data_format) for image in images]

        data = {"pixel_values": images}
        return BatchEncoding(data=data, tensor_type=return_tensors)

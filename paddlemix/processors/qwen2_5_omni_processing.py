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

"""
Processor class for Qwen2.5Omni.
"""
import sys
import typing
from typing import TYPE_CHECKING, List, Optional, TypedDict, Union

import paddle
import typing_extensions
from paddlenlp.transformers.feature_extraction_utils import BatchFeature
from paddlenlp.transformers.image_utils import ChannelDimension, ImageInput
from paddlenlp.transformers.tokenizer_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    TensorType,
    TextInput,
    TruncationStrategy,
)

from paddlemix.processors.base_processing import ProcessorMixin
from ppdiffusers.utils import logging

logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    import numpy as np
    import PIL
    import PIL.Image
    from PIL.Image import Resampling as PILImageResampling

__all__ = ["Qwen2_5OmniProcessor"]

VideoInput = Union[
    list["PIL.Image.Image"],
    "np.ndarray",
    "paddle.Tensor",
    list["np.ndarray"],
    list["paddle.Tensor"],
    list[list["PIL.Image.Image"]],
    list[list["np.ndarray"]],
    list[list["paddle.Tensor"]],
]  # noqa

if sys.version_info >= (3, 11):
    Unpack = typing.Unpack
else:
    Unpack = typing_extensions.Unpack


class TextKwargs(TypedDict, total=False):
    """
    Keyword arguments for text processing. For extended documentation, check out tokenization_utils_base methods and
    docstrings associated.

    Attributes:
        add_special_tokens (`bool`, *optional*)
            Whether or not to add special tokens when encoding the sequences.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*)
            Activates and controls padding.
        truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*):
            Activates and controls truncation.
        max_length (`int`, *optional*):
            Controls the maximum length to use by one of the truncation/padding parameters.
        stride (`int`, *optional*):
            If set, the overflowing tokens will contain some tokens from the end of the truncated sequence.
        is_split_into_words (`bool`, *optional*):
            Whether or not the input is already pre-tokenized.
        pad_to_multiple_of (`int`, *optional*):
            If set, will pad the sequence to a multiple of the provided value.
        return_token_type_ids (`bool`, *optional*):
            Whether to return token type IDs.
        return_attention_mask (`bool`, *optional*):
            Whether to return the attention mask.
        return_overflowing_tokens (`bool`, *optional*):
            Whether or not to return overflowing token sequences.
        return_special_tokens_mask (`bool`, *optional*):
            Whether or not to return special tokens mask information.
        return_offsets_mapping (`bool`, *optional*):
            Whether or not to return `(char_start, char_end)` for each token.
        return_length (`bool`, *optional*):
            Whether or not to return the lengths of the encoded inputs.
        verbose (`bool`, *optional*):
            Whether or not to print more information and warnings.
        padding_side (`str`, *optional*):
            The side on which padding will be applied.
    """

    text_pair: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]]
    text_target: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]
    text_pair_target: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]]
    add_special_tokens: Optional[bool]
    padding: Union[bool, str, PaddingStrategy]
    truncation: Union[bool, str, TruncationStrategy]
    max_length: Optional[int]
    stride: Optional[int]
    is_split_into_words: Optional[bool]
    pad_to_multiple_of: Optional[int]
    return_token_type_ids: Optional[bool]
    return_attention_mask: Optional[bool]
    return_overflowing_tokens: Optional[bool]
    return_special_tokens_mask: Optional[bool]
    return_offsets_mapping: Optional[bool]
    return_length: Optional[bool]
    verbose: Optional[bool]
    padding_side: Optional[str]


class ImagesKwargs(TypedDict, total=False):
    """
    Keyword arguments for image processing. For extended documentation, check the appropriate ImageProcessor
    class methods and docstrings.

    Attributes:
        do_resize (`bool`, *optional*):
            Whether to resize the image.
        size (`Dict[str, int]`, *optional*):
            Resize the shorter side of the input to `size["shortest_edge"]`.
        size_divisor (`int`, *optional*):
            The size by which to make sure both the height and width can be divided.
        crop_size (`Dict[str, int]`, *optional*):
            Desired output size when applying center-cropping.
        resample (`PILImageResampling`, *optional*):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*):
            Mean to use if normalizing the image.
        image_std (`float` or `List[float]`, *optional*):
            Standard deviation to use if normalizing the image.
        do_pad (`bool`, *optional*):
            Whether to pad the image to the `(max_height, max_width)` of the images in the batch.
        pad_size (`Dict[str, int]`, *optional*):
            The size `{"height": int, "width" int}` to pad the images to.
        do_center_crop (`bool`, *optional*):
            Whether to center crop the image.
        data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the output image.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image.
        device (`str`, *optional*):
            The device to use for processing (e.g. "cpu", "cuda"), only relevant for fast image processing.
    """

    do_resize: Optional[bool]
    size: Optional[dict[str, int]]
    size_divisor: Optional[int]
    crop_size: Optional[dict[str, int]]
    resample: Optional[Union["PILImageResampling", int]]
    do_rescale: Optional[bool]
    rescale_factor: Optional[float]
    do_normalize: Optional[bool]
    image_mean: Optional[Union[float, list[float]]]
    image_std: Optional[Union[float, list[float]]]
    do_pad: Optional[bool]
    pad_size: Optional[dict[str, int]]
    do_center_crop: Optional[bool]
    data_format: Optional[ChannelDimension]
    input_data_format: Optional[Union[str, ChannelDimension]]
    device: Optional[str]


class VideosKwargs(TypedDict, total=False):
    """
    Keyword arguments for video processing.

    Attributes:
        do_resize (`bool`):
            Whether to resize the image.
        size (`Dict[str, int]`, *optional*):
            Resize the shorter side of the input to `size["shortest_edge"]`.
        size_divisor (`int`, *optional*):
            The size by which to make sure both the height and width can be divided.
        resample (`PILImageResampling`, *optional*):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*):
            Mean to use if normalizing the image.
        image_std (`float` or `List[float]`, *optional*):
            Standard deviation to use if normalizing the image.
        do_pad (`bool`, *optional*):
            Whether to pad the image to the `(max_height, max_width)` of the images in the batch.
        do_center_crop (`bool`, *optional*):
            Whether to center crop the image.
        data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the output image.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image.
    """

    do_resize: Optional[bool]
    size: Optional[dict[str, int]]
    size_divisor: Optional[int]
    resample: Optional["PILImageResampling"]
    do_rescale: Optional[bool]
    rescale_factor: Optional[float]
    do_normalize: Optional[bool]
    image_mean: Optional[Union[float, list[float]]]
    image_std: Optional[Union[float, list[float]]]
    do_pad: Optional[bool]
    do_center_crop: Optional[bool]
    data_format: Optional[ChannelDimension]
    input_data_format: Optional[Union[str, ChannelDimension]]


class AudioKwargs(TypedDict, total=False):
    """
    Keyword arguments for audio processing.

    Attributes:
        sampling_rate (`int`, *optional*):
            The sampling rate at which the `raw_speech` input was sampled.
        raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
            The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
            values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
            stereo, i.e. single float per timestep.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding
            index) among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        truncation (`bool`, *optional*):
            Activates truncation to cut input sequences longer than *max_length* to *max_length*.
        pad_to_multiple_of (`int`, *optional*):
            If set, will pad the sequence to a multiple of the provided value.
        return_attention_mask (`bool`, *optional*):
            Whether or not [`~ASTFeatureExtractor.__call__`] should return `attention_mask`.
    """

    sampling_rate: Optional[int]
    raw_speech: Optional[Union["np.ndarray", list[float], list["np.ndarray"], list[list[float]]]]
    padding: Optional[Union[bool, str, PaddingStrategy]]
    max_length: Optional[int]
    truncation: Optional[bool]
    pad_to_multiple_of: Optional[int]
    return_attention_mask: Optional[bool]


class CommonKwargs(TypedDict, total=False):
    return_tensors: Optional[Union[str, TensorType]]


class ProcessingKwargs(TextKwargs, ImagesKwargs, VideosKwargs, AudioKwargs, CommonKwargs, total=False):
    """
    Base class for kwargs passing to processors.
    A model should have its own `ModelProcessorKwargs` class that inherits from `ProcessingKwargs` to provide:
        1) Additional typed keys and that this model requires to process inputs.
        2) Default values for existing keys under a `_defaults` attribute.
    New keys have to be defined as follows to ensure type hinting is done correctly.

    ```python
    # adding a new image kwarg for this model
    class ModelImagesKwargs(ImagesKwargs, total=False):
        new_image_kwarg: Optional[bool]

    class ModelProcessorKwargs(ProcessingKwargs, total=False):
        images_kwargs: ModelImagesKwargs
        _defaults = {
            "images_kwargs: {
                "new_image_kwarg": False,
            }
            "text_kwargs": {
                "padding": "max_length",
            },
        }

    ```

    For Python 3.8 compatibility, when inheriting from this class and overriding one of the kwargs,
    you need to manually update the __annotations__ dictionary. This can be done as follows:

    ```python
    class CustomProcessorKwargs(ProcessingKwargs, total=False):
        images_kwargs: CustomImagesKwargs

    CustomProcessorKwargs.__annotations__["images_kwargs"] = CustomImagesKwargs  # python 3.8 compatibility
    ```python

    """

    common_kwargs: CommonKwargs = {
        **CommonKwargs.__annotations__,
    }
    text_kwargs: TextKwargs = {
        **TextKwargs.__annotations__,
    }
    images_kwargs: ImagesKwargs = {
        **ImagesKwargs.__annotations__,
    }
    videos_kwargs: VideosKwargs = {
        **VideosKwargs.__annotations__,
    }
    audio_kwargs: AudioKwargs = {
        **AudioKwargs.__annotations__,
    }


class Qwen2_5OmniProcessorKwargs(ProcessingKwargs, total=(False)):
    _defaults = {
        "text_kwargs": {"padding": False, "padding_side": "left"},
        "audio_kwargs": {"sampling_rate": 16000, "padding": "max_length", "return_attention_mask": True},
    }


AudioInput = Union["np.ndarray", "paddle.Tensor", List["np.ndarray"], List["paddle.Tensor"]]


class Qwen2_5OmniProcessor(ProcessorMixin):
    """
    Constructs a Qwen2.5Omni processor.
    [`Qwen2_5OmniProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`], [`WhisperFeatureExtractor`], and [`Qwen2TokenizerFast`]. See the
    [`~Qwen2_5OmniProcessor.__call__`] and [`~Qwen2_5OmniProcessor.decode`] for more information.

    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor.
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The audio feature extractor.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The text tokenizer.
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided, the default chat template is used.
    """

    attributes = ["image_processor", "feature_extractor", "tokenizer"]
    image_processor_class = "Qwen2VLImageProcessor"
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "Qwen2Tokenizer", "Qwen2TokenizerFast"
    valid_kwargs = ["chat_template"]

    def __init__(self, image_processor=None, feature_extractor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, feature_extractor, tokenizer, **kwargs)
        self.image_token = self.tokenizer.image_token
        self.audio_token = self.tokenizer.audio_token
        self.video_token = self.tokenizer.video_token
        self.vision_bos_token = self.tokenizer.vision_bos_token
        self.vision_eos_token = self.tokenizer.vision_eos_token
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_eos_token = self.tokenizer.audio_eos_token

    def _merge_kwargs(
        self,
        ModelProcessorKwargs: ProcessingKwargs,
        tokenizer_init_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> dict[str, dict]:
        """
        Method to merge dictionaries of kwargs cleanly separated by modality within a Processor instance.
        The order of operations is as follows:
            1) kwargs passed as before have highest priority to preserve BC.
                ```python
                high_priority_kwargs = {"crop_size" = {"height": 222, "width": 222}, "padding" = "max_length"}
                processor(..., **high_priority_kwargs)
                ```
            2) kwargs passed as modality-specific kwargs have second priority. This is the recommended API.
                ```python
                processor(..., text_kwargs={"padding": "max_length"}, images_kwargs={"crop_size": {"height": 222, "width": 222}}})
                ```
            3) kwargs passed during instantiation of a modality processor have fourth priority.
                ```python
                tokenizer = tokenizer_class(..., {"padding": "max_length"})
                image_processor = image_processor_class(...)
                processor(tokenizer, image_processor) # will pass max_length unless overridden by kwargs at call
                ```
            4) defaults kwargs specified at processor level have lowest priority.
                ```python
                class MyProcessingKwargs(ProcessingKwargs, CommonKwargs, TextKwargs, ImagesKwargs, total=False):
                    _defaults = {
                        "text_kwargs": {
                            "padding": "max_length",
                            "max_length": 64,
                        },
                    }
                ```
        Args:
            ModelProcessorKwargs (`ProcessingKwargs`):
                Typed dictionary of kwargs specifically required by the model passed.
            tokenizer_init_kwargs (`Dict`, *optional*):
                Dictionary of kwargs the tokenizer was instantiated with and need to take precedence over defaults.

        Returns:
            output_kwargs (`Dict`):
                Dictionary of per-modality kwargs to be passed to each modality-specific processor.

        """
        # Initialize dictionaries
        output_kwargs = {
            "text_kwargs": {},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
            "common_kwargs": {},
        }

        default_kwargs = {
            "text_kwargs": {},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
            "common_kwargs": {},
        }

        possible_modality_keywords = {"text", "audio", "videos", "images"}
        used_keys = set()

        # get defaults from set model processor kwargs if they exist
        for modality in default_kwargs:
            default_kwargs[modality] = ModelProcessorKwargs._defaults.get(modality, {}).copy()
            # update defaults with arguments from tokenizer init
            for modality_key in ModelProcessorKwargs.__annotations__[modality].__annotations__.keys():
                # init with tokenizer init kwargs if necessary
                if modality_key in tokenizer_init_kwargs:
                    value = (
                        getattr(self.tokenizer, modality_key)
                        if hasattr(self.tokenizer, modality_key)
                        else tokenizer_init_kwargs[modality_key]
                    )
                    default_kwargs[modality][modality_key] = value
        # now defaults kwargs are updated with the tokenizers defaults.
        # pass defaults to output dictionary
        output_kwargs.update(default_kwargs)

        # update modality kwargs with passed kwargs
        non_modality_kwargs = set(kwargs) - set(output_kwargs)
        for modality in output_kwargs:
            for modality_key in ModelProcessorKwargs.__annotations__[modality].__annotations__.keys():
                # check if we received a structured kwarg dict or not to handle it correctly
                if modality in kwargs:
                    kwarg_value = kwargs[modality].pop(modality_key, "__empty__")
                    # check if this key was passed as a flat kwarg.
                    if kwarg_value != "__empty__" and modality_key in non_modality_kwargs:
                        raise ValueError(
                            f"Keyword argument {modality_key} was passed two times:\n"
                            f"in a dictionary for {modality} and as a **kwarg."
                        )
                elif modality_key in kwargs:
                    # we get a modality_key instead of popping it because modality-specific processors
                    # can have overlapping kwargs
                    kwarg_value = kwargs.get(modality_key, "__empty__")
                else:
                    kwarg_value = "__empty__"
                if not isinstance(kwarg_value, str) or kwarg_value != "__empty__":
                    output_kwargs[modality][modality_key] = kwarg_value
                    used_keys.add(modality_key)

        # Determine if kwargs is a flat dictionary or contains nested dictionaries
        if any(key in default_kwargs for key in kwargs):
            # kwargs is dictionary-based, and some keys match modality names
            for modality, subdict in kwargs.items():
                if modality in default_kwargs:
                    for subkey, subvalue in subdict.items():
                        if subkey not in used_keys:
                            output_kwargs[modality][subkey] = subvalue
                            used_keys.add(subkey)
        else:
            # kwargs is a flat dictionary
            for key in kwargs:
                if key not in used_keys:
                    if key in ModelProcessorKwargs.__annotations__["common_kwargs"].__annotations__.keys():
                        output_kwargs["common_kwargs"][key] = kwargs[key]
                    elif key not in possible_modality_keywords:
                        logger.warning_once(
                            f"Keyword argument `{key}` is not a valid argument for this processor and will be ignored."
                        )

        # all modality-specific kwargs are updated with common kwargs
        for modality in output_kwargs:
            output_kwargs[modality].update(output_kwargs["common_kwargs"])
        return output_kwargs

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        fps: Optional[List[int]] = None,
        use_audio_in_video: Optional[bool] = False,
        seconds_per_chunk: Optional[float] = 2.0,
        position_id_per_seconds: Optional[int] = 25,
        **kwargs: Unpack[Qwen2_5OmniProcessorKwargs]
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audio` and `kwrags` arguments to
        WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] if `audio` is not `None`. To prepare the vision inputs,
        this method forwards the `vision_infos` and `kwrags` arguments to Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`]
        if `vision_infos` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            audio (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audio to be prepared. Each audio can be a NumPy array.
        """
        output_kwargs = self._merge_kwargs(
            Qwen2_5OmniProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        if audio is not None:
            output_kwargs["audio_kwargs"]["padding"] = "max_length"
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")
            audio_inputs["input_features"] = audio_inputs.pop("input_features")
            input_lengths = (audio_inputs["feature_attention_mask"].sum(axis=-1).numpy() - 1) // 2 + 1
            audio_lengths = (input_lengths - 2) // 2 + 1
        else:
            audio_inputs = {}
            audio_lengths = None
        if images is not None:
            images_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = images_inputs["image_grid_thw"]
        else:
            images_inputs = {}
            image_grid_thw = None
        if videos is not None:
            videos_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["videos_kwargs"])
            if fps is None:
                fps = [2.0] * len(videos)
            videos_inputs["video_second_per_grid"] = [
                (self.image_processor.temporal_patch_size / fps[i]) for i in range(len(fps))
            ]
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None
        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")
        if not isinstance(text, list):
            text = [text]
        text = text.copy()
        merge_length = self.image_processor.merge_size**2
        audio_index = 0
        image_index = 0
        video_index = 0
        for i in range(len(text)):
            positions = []
            for special_token in [self.audio_token, self.image_token, self.video_token]:
                start = 0
                while True:
                    pos = text[i].find(special_token, start)
                    if pos == -1:
                        break
                    positions.append((pos, special_token))
                    start = pos + len(special_token)
            positions.sort(key=lambda x: x[0])
            for _, special_token in positions:
                if audio is not None and special_token == self.audio_token:
                    text[i] = text[i].replace(
                        self.audio_token, "<|audio_placeholder|>" * audio_lengths[audio_index], 1
                    )
                    audio_index += 1
                elif images is not None and special_token == self.image_token:
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|image_placeholder|>" * (image_grid_thw[image_index].prod().item() // merge_length),
                        1,
                    )
                    image_index += 1
                elif videos is not None and special_token == self.video_token:
                    if not use_audio_in_video:
                        text[i] = text[i].replace(
                            self.video_token,
                            "<|video_placeholder|>" * (video_grid_thw[video_index].prod().item() // merge_length),
                            1,
                        )
                        video_index += 1
                    else:
                        audio_t_index = paddle.arange(end=audio_lengths[audio_index])
                        video_t_index = (
                            paddle.arange(end=video_grid_thw[video_index][0])
                            .view([-1, 1, 1])
                            .expand(
                                shape=[
                                    -1,
                                    video_grid_thw[video_index][1] // self.image_processor.merge_size,
                                    video_grid_thw[video_index][2] // self.image_processor.merge_size,
                                ]
                            )
                            .flatten()
                            * videos_inputs["video_second_per_grid"][video_index]
                            * position_id_per_seconds
                        ).astype(dtype="int64")
                        t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                        video_chunk_indexes = self.get_chunked_index(video_t_index, t_ntoken_per_chunk)
                        audio_chunk_indexes = self.get_chunked_index(audio_t_index, t_ntoken_per_chunk)
                        placeholder_string = str()
                        placeholder_string += self.vision_bos_token + self.audio_bos_token
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            video_chunk_index = video_chunk_indexes[j] if j < len(video_chunk_indexes) else None
                            audio_chunk_index = audio_chunk_indexes[j] if j < len(audio_chunk_indexes) else None
                            if video_chunk_index is not None:
                                placeholder_string += "<|video_placeholder|>" * (
                                    video_chunk_index[1] - video_chunk_index[0]
                                )
                            if audio_chunk_index is not None:
                                placeholder_string += "<|audio_placeholder|>" * (
                                    audio_chunk_index[1] - audio_chunk_index[0]
                                )
                        placeholder_string += self.audio_eos_token + self.vision_eos_token
                        text[i] = text[i].replace(
                            self.vision_bos_token + self.video_token + self.vision_eos_token, placeholder_string, 1
                        )
                        audio_index += 1
                        video_index += 1
            text[i] = text[i].replace("<|audio_placeholder|>", self.audio_token)
            text[i] = text[i].replace("<|image_placeholder|>", self.image_token)
            text[i] = text[i].replace("<|video_placeholder|>", self.video_token)
        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        return BatchFeature(
            data={**texts_inputs, **images_inputs, **videos_inputs, **audio_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def get_chunked_index(self, t_index, t_ntoken_per_chunk):
        def _iter():
            i, start_idx = 0, 0
            current_chunk = 1
            while i < len(t_index):
                if t_index[i] >= current_chunk * t_ntoken_per_chunk:
                    yield start_idx, i
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield start_idx, len(t_index)

        return list(_iter())

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        for conversation in conversations:
            if (
                conversation[0]["role"] != "system"
                or conversation[0]["content"]
                != "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            ):
                logging.warning(
                    "System prompt modified, audio output may not work as expected. "
                    + "Audio output mode only works when using default system prompt 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'"
                )
        return [self.tokenizer.apply_chat_template(conv, chat_template) for conv in conversations]

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + feature_extractor_input_names
                + image_processor_input_names
                + ["feature_attention_mask"]
                + ["video_second_per_grid"]
            )
        )

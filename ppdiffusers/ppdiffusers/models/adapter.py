# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from typing import List, Optional

import paddle
import paddle.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from .modeling_utils import ModelMixin
from .resnet import Downsample2D


class MultiAdapter(ModelMixin):
    r"""
    MultiAdapter is a wrapper model that contains multiple adapter models and merges their outputs according to
    user-assigned weighting.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        adapters (`List[T2IAdapter]`, *optional*, defaults to None):
            A list of `T2IAdapter` model instances.
    """

    def __init__(self, adapters: List["T2IAdapter"]):
        super(MultiAdapter, self).__init__()
        self.num_adapter = len(adapters)
        self.adapters = nn.LayerList(adapters)

    def forward(self, xs: paddle.Tensor, adapter_weights: Optional[List[float]] = None) -> List[paddle.Tensor]:
        r"""
        Args:
            xs (`paddle.Tensor`):
                (batch, channel, height, width) input images for multiple adapter models concated along dimension 1,
                `channel` should equal to `num_adapter` * "number of channel of image".
            adapter_weights (`List[float]`, *optional*, defaults to None):
                List of floats representing the weight which will be multiply to each adapter's output before adding
                them together.
        """
        if adapter_weights is None:
            adapter_weights = paddle.to_tensor([1 / self.num_adapter] * self.num_adapter)
        else:
            adapter_weights = paddle.to_tensor(adapter_weights)
        if xs.shape[1] % self.num_adapter != 0:
            raise ValueError(
                f"Expecting multi-adapter's input have number of channel that cab be evenly divisible by num_adapter: {xs.shape[1]} % {self.num_adapter} != 0"
            )
        x_list = paddle.chunk(xs, chunks=self.num_adapter, axis=1)
        accume_state = None
        for x, w, adapter in zip(x_list, adapter_weights, self.adapters):
            features = adapter(x)
            if accume_state is None:
                accume_state = features
            else:
                for i in range(len(features)):
                    accume_state[i] += w * features[i]
        return accume_state

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Callable = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `[`~models.adapter.MultiAdapter.from_pretrained`]` class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
        """
        idx = 0
        model_path_to_save = save_directory
        for adapter in self.adapters:
            adapter.save_pretrained(
                model_path_to_save,
                is_main_process=is_main_process,
                save_function=save_function,
                safe_serialization=safe_serialization,
                variant=variant,
            )

            idx += 1
            model_path_to_save = model_path_to_save + f"_{idx}"

    @classmethod
    def from_pretrained(cls, pretrained_model_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a pretrained MultiAdapter model from multiple pre-trained adapter models.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_path (`os.PathLike`):
                A path to a *directory* containing model weights saved using
                [`~ppdiffusers.models.adapter.MultiAdapter.save_pretrained`], e.g., `./my_model_directory/adapter`.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype. If `"auto"` is passed the dtype
                will be automatically derived from the model's weights.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier to maximum memory. Will default to the maximum memory available for each
                GPU and the available CPU RAM if unset.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading by not initializing the weights and only loading the pre-trained weights. This
                also tries to not use more than 1x model size in CPU memory (including peak memory) while loading the
                model. This is only supported when torch version >= 1.9.0. If you are using an older version of torch,
                setting this argument to `True` will raise an error.
            variant (`str`, *optional*):
                If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. `variant` is
                ignored when using `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the `safetensors` weights will be downloaded if they're available **and** if the
                `safetensors` library is installed. If set to `True`, the model will be forcibly loaded from
                `safetensors` weights. If set to `False`, loading will *not* use `safetensors`.
        """
        idx = 0
        adapters = []

        # load adapter and append to list until no adapter directory exists anymore
        # first adapter has to be saved under `./mydirectory/adapter` to be compliant with `DiffusionPipeline.from_pretrained`
        # second, third, ... adapters have to be saved under `./mydirectory/adapter_1`, `./mydirectory/adapter_2`, ...
        model_path_to_load = pretrained_model_path
        while os.path.isdir(model_path_to_load):
            adapter = T2IAdapter.from_pretrained(model_path_to_load, **kwargs)
            adapters.append(adapter)

            idx += 1
            model_path_to_load = pretrained_model_path + f"_{idx}"

        logger.info(f"{len(adapters)} adapters loaded from {pretrained_model_path}.")

        if len(adapters) == 0:
            raise ValueError(
                f"No T2IAdapters found under {os.path.dirname(pretrained_model_path)}. Expected at least {pretrained_model_path + '_0'}."
            )

        return cls(adapters)


class T2IAdapter(ModelMixin, ConfigMixin):
    r"""
    A simple ResNet-like model that accepts images containing control signals such as keyposes and depth. The model
    generates multiple feature maps that are used as additional conditioning in [`UNet2DConditionModel`]. The model's
    architecture follows the original implementation of
    [Adapter](https://github.com/TencentARC/T2I-Adapter/blob/686de4681515662c0ac2ffa07bf5dda83af1038a/ldm/modules/encoders/adapter.py#L97)
     and
     [AdapterLight](https://github.com/TencentARC/T2I-Adapter/blob/686de4681515662c0ac2ffa07bf5dda83af1038a/ldm/modules/encoders/adapter.py#L235).
    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)
    Parameters:
        in_channels (`int`, *optional*, defaults to 3):
            Number of channels of Aapter's input(*control image*). Set this parameter to 1 if you're using gray scale
            image as *control image*.
        channels (`List[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The number of channel of each downsample block's output hidden state. The `len(block_out_channels)` will
            also determine the number of downsample blocks in the Adapter.
        num_res_blocks (`int`, *optional*, defaults to 2):
            Number of ResNet blocks in each downsample block
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 8,
        adapter_type: str = "full_adapter",
    ):
        super().__init__()

        if adapter_type == "full_adapter":
            self.adapter = FullAdapter(in_channels, channels, num_res_blocks, downscale_factor)
        elif adapter_type == "light_adapter":
            self.adapter = LightAdapter(in_channels, channels, num_res_blocks, downscale_factor)
        else:
            raise ValueError(f"unknown adapter_type: {type}. Choose either 'full_adapter' or 'simple_adapter'")

    def forward(self, x: paddle.Tensor) -> List[paddle.Tensor]:
        return self.adapter(x)

    @property
    def total_downscale_factor(self):
        return self.adapter.total_downscale_factor


# full adapter


class FullAdapter(nn.Layer):
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 8,
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor**2

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv_in = nn.Conv2D(in_channels, channels[0], kernel_size=3, padding=1)

        self.body = nn.LayerList(
            [
                AdapterBlock(channels[0], channels[0], num_res_blocks),
                *[
                    AdapterBlock(channels[i - 1], channels[i], num_res_blocks, down=True)
                    for i in range(1, len(channels))
                ],
            ]
        )

        self.total_downscale_factor = downscale_factor * 2 ** (len(channels) - 1)

    def forward(self, x: paddle.Tensor) -> List[paddle.Tensor]:
        r"""
        This method processes the input tensor `x` through the FullAdapter model and performs operations including
        pixel unshuffling, convolution, and a stack of AdapterBlocks. It returns a list of feature tensors, each
        capturing information at a different stage of processing within the FullAdapter model. The number of feature
        tensors in the list is determined by the number of downsample blocks specified during initialization.
        """
        x = self.unshuffle(x)
        x = self.conv_in(x)

        features = []

        for block in self.body:
            x = block(x)
            features.append(x)

        return features


class FullAdapterXL(nn.Layer):
    r"""
    See [`T2IAdapter`] for more information.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 16,
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor**2

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv_in = nn.Conv2D(in_channels, channels[0], kernel_size=3, padding=1)

        self.body = []
        # blocks to extract XL features with dimensions of [320, 64, 64], [640, 64, 64], [1280, 32, 32], [1280, 32, 32]
        for i in range(len(channels)):
            if i == 1:
                self.body.append(AdapterBlock(channels[i - 1], channels[i], num_res_blocks))
            elif i == 2:
                self.body.append(AdapterBlock(channels[i - 1], channels[i], num_res_blocks, down=True))
            else:
                self.body.append(AdapterBlock(channels[i], channels[i], num_res_blocks))

        self.body = nn.LayerList(self.body)
        # XL has only one downsampling AdapterBlock.
        self.total_downscale_factor = downscale_factor * 2

    def forward(self, x: paddle.Tensor) -> List[paddle.Tensor]:
        r"""
        This method takes the tensor x as input and processes it through FullAdapterXL model. It consists of operations
        including unshuffling pixels, applying convolution layer and appending each block into list of feature tensors.
        """
        x = self.unshuffle(x)
        x = self.conv_in(x)

        features = []

        for block in self.body:
            x = block(x)
            features.append(x)

        return features


class AdapterBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, num_res_blocks, down=False):
        super().__init__()

        self.downsample = None
        if down:
            self.downsample = Downsample2D(in_channels)

        self.in_conv = None
        if in_channels != out_channels:
            self.in_conv = nn.Conv2D(in_channels, out_channels, kernel_size=1)

        self.resnets = nn.Sequential(
            *[AdapterResnetBlock(out_channels) for _ in range(num_res_blocks)],
        )

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        if self.in_conv is not None:
            x = self.in_conv(x)

        x = self.resnets(x)

        return x


class AdapterResnetBlock(nn.Layer):
    r"""
    An `AdapterResnetBlock` is a helper model that implements a ResNet-like block.

    Parameters:
        channels (`int`):
            Number of channels of AdapterResnetBlock's input and output.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block1 = nn.Conv2D(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2D(channels, channels, kernel_size=1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        r"""
        This method takes input tensor x and applies a convolutional layer, ReLU activation, and another convolutional
        layer on the input tensor. It returns addition with the input tensor.
        """

        h = self.act(self.block1(x))
        h = self.block2(h)

        return h + x


# light adapter


class LightAdapter(nn.Layer):
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280],
        num_res_blocks: int = 4,
        downscale_factor: int = 8,
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor**2

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)

        self.body = nn.LayerList(
            [
                LightAdapterBlock(in_channels, channels[0], num_res_blocks),
                *[
                    LightAdapterBlock(channels[i], channels[i + 1], num_res_blocks, down=True)
                    for i in range(len(channels) - 1)
                ],
                LightAdapterBlock(channels[-1], channels[-1], num_res_blocks, down=True),
            ]
        )

        self.total_downscale_factor = downscale_factor * (2 ** len(channels))

    def forward(self, x):
        x = self.unshuffle(x)

        features = []

        for block in self.body:
            x = block(x)
            features.append(x)

        return features


class LightAdapterBlock(nn.Layer):
    r"""
    A `LightAdapterBlock` is a helper model that contains multiple `LightAdapterResnetBlocks`. It is used in the
    `LightAdapter` model.

    Parameters:
        in_channels (`int`):
            Number of channels of LightAdapterBlock's input.
        out_channels (`int`):
            Number of channels of LightAdapterBlock's output.
        num_res_blocks (`int`):
            Number of LightAdapterResnetBlocks in the LightAdapterBlock.
        down (`bool`, *optional*, defaults to `False`):
            Whether to perform downsampling on LightAdapterBlock's input.
    """

    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int, down: bool = False):
        super().__init__()
        mid_channels = out_channels // 4

        self.downsample = None
        if down:
            self.downsample = Downsample2D(in_channels)

        self.in_conv = nn.Conv2D(in_channels, mid_channels, kernel_size=1)
        self.resnets = nn.Sequential(*[LightAdapterResnetBlock(mid_channels) for _ in range(num_res_blocks)])
        self.out_conv = nn.Conv2D(mid_channels, out_channels, kernel_size=1)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        x = self.in_conv(x)
        x = self.resnets(x)
        x = self.out_conv(x)

        return x


class LightAdapterResnetBlock(nn.Layer):
    """
    A `LightAdapterResnetBlock` is a helper model that implements a ResNet-like block with a slightly different
    architecture than `AdapterResnetBlock`.

    Parameters:
        channels (`int`):
            Number of channels of LightAdapterResnetBlock's input and output.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block1 = nn.Conv2D(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2D(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.act(h)
        h = self.block2(h)

        return h + x

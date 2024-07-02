# Copyright 2024 Vchitect/Latte

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.# Modified from Latte

# - This file is adapted from https://github.com/Vchitect/Latte/blob/main/datasets/video_transforms.py


import numbers
import random

import paddle


def _is_tensor_video_clip(clip):
    if not paddle.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True


def crop(clip, i, j, h, w):
    """
    Args:
        clip (paddle.Tensor): Video clip to be cropped. Size is (T, C, H, W)
    """
    if len(clip.shape) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i : i + h, j : j + w]


def resize(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    return paddle.nn.functional.interpolate(clip, size=target_size, mode=interpolation_mode, align_corners=False)


def resize_scale(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    H, W = clip.shape[-2], clip.shape[-1]
    scale_ = target_size[0] / min(H, W)
    return paddle.nn.functional.interpolate(clip, scale_factor=scale_, mode=interpolation_mode, align_corners=False)


def resized_crop(clip, i, j, h, w, size, interpolation_mode="bilinear"):
    """
    Do spatial cropping and resizing to the video clip
    Args:
        clip (paddle.Tensor): Video clip to be cropped. Size is (T, C, H, W)
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped region.
        w (int): Width of the cropped region.
        size (tuple(int, int)): height and width of resized clip
    Returns:
        clip (paddle.Tensor): Resized and cropped clip. Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D paddle.Tensor")
    clip = crop(clip, i, j, h, w)
    clip = resize(clip, size, interpolation_mode)
    return clip


def center_crop(clip, crop_size):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D paddle.Tensor")
    h, w = clip.shape[-2], clip.shape[-1]
    th, tw = crop_size
    if h < th or w < tw:
        raise ValueError("height and width must be no smaller than crop_size")

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw)


def center_crop_using_short_edge(clip):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D paddle.Tensor")
    h, w = clip.shape[-2], clip.shape[-1]
    if h < w:
        th, tw = h, h
        i = 0
        j = int(round((w - tw) / 2.0))
    else:
        th, tw = w, w
        i = int(round((h - th) / 2.0))
        j = 0
    return crop(clip, i, j, th, tw)


def resize_crop_to_fill(clip, target_size):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D paddle.Tensor")
    h, w = clip.shape[-2], clip.shape[-1]
    th, tw = target_size[0], target_size[1]
    rh, rw = th / h, tw / w
    if rh > rw:

        sh, sw = th, round(w * rh)
        clip = resize(clip, (sh, sw), "bilinear")
        i = 0
        j = int(round(sw - tw) / 2.0)
    else:

        sh, sw = round(h * rw), tw
        clip = resize(clip, (sh, sw), "bilinear")
        i = int(round(sh - th) / 2.0)
        j = 0
    assert i + th <= clip.shape[-2] and j + tw <= clip.shape[-1]
    return crop(clip, i, j, th, tw)


def random_shift_crop(clip):
    """
    Slide along the long edge, with the short edge as crop size
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D paddle.Tensor")
    h, w = clip.shape[-2], clip.shape[-1]

    if h <= w:
        short_edge = h
    else:
        short_edge = w

    th, tw = short_edge, short_edge

    i = paddle.randint(low=0, high=h - th + 1, shape=(1,)).item()
    j = paddle.randint(low=0, high=w - tw + 1, shape=(1,)).item()

    return crop(clip, i, j, th, tw)


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (paddle.Tensor, dtype=paddle.uint8): Size is (T, C, H, W)
    Return:
        clip (paddle.Tensor, dtype=paddle.float): Size is (T, C, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == paddle.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    # return clip.float().permute(3, 0, 1, 2) / 255.0
    return clip.astype(dtype="float32") / 255.0


def normalize(clip, mean, std, inplace=False):
    """
    Args:
        clip (paddle.Tensor): Video clip to be normalized. Size is (T, C, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (paddle.Tensor): Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D paddle.Tensor")
    if not inplace:
        clip = clip.clone()
    mean = paddle.to_tensor(mean, dtype=clip.dtype)
    std = paddle.to_tensor(std, dtype=clip.dtype)
    clip.subtract_(y=paddle.to_tensor(mean[:, None, None, None])).divide_(y=paddle.to_tensor(std[:, None, None, None]))
    clip = clip.transpose((1, 0, 2, 3))
    return clip


def hflip(clip):
    """
    Args:
        clip (paddle.Tensor): Video clip to be normalized. Size is (T, C, H, W)
    Returns:
        flipped clip (paddle.Tensor): Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D paddle.Tensor")
    return clip.flip(axis=-1)


class ResizeCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):

        clip = resize_crop_to_fill(clip, self.size)
        return clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class RandomCropVideo:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        """
        Args:
            clip (paddle.Tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            paddle.Tensor: randomly cropped video clip.
                size is (T, C, OH, OW)
        """
        i, j, h, w = self.get_params(clip)
        return crop(clip, i, j, h, w)

    def get_params(self, clip):
        h, w = clip.shape[-2:]
        th, tw = self.size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = paddle.randint(low=0, high=h - th + 1, shape=(1,)).item()
        j = paddle.randint(low=0, high=w - tw + 1, shape=(1,)).item()

        return i, j, th, tw

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class CenterCropResizeVideo:
    """
    First use the short side for cropping length,
    center crop video, then resize to the specified size
    """

    def __init__(
        self,
        size,
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (paddle.Tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            paddle.Tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_center_crop = center_crop_using_short_edge(clip)
        clip_center_crop_resize = resize(
            clip_center_crop, target_size=self.size, interpolation_mode=self.interpolation_mode
        )
        return clip_center_crop_resize

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class UCFCenterCropVideo:
    """
    First scale to the specified size in equal proportion to the short edge,
    then center cropping
    """

    def __init__(
        self,
        size,
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (paddle.Tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            paddle.Tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_resize = resize_scale(clip=clip, target_size=self.size, interpolation_mode=self.interpolation_mode)
        clip_center_crop = center_crop(clip_resize, self.size)
        return clip_center_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class KineticsRandomCropResizeVideo:
    """
    Slide along the long edge, with the short edge as crop size. And resie to the desired size.
    """

    def __init__(
        self,
        size,
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        clip_random_crop = random_shift_crop(clip)
        clip_resize = resize(clip_random_crop, self.size, self.interpolation_mode)
        return clip_resize


class CenterCropVideo:
    def __init__(
        self,
        size,
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (paddle.Tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            paddle.Tensor: center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_center_crop = center_crop(clip, self.size)
        return clip_center_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class NormalizeVideo:
    """
    Normalize the video clip by mean subtraction and division by standard deviation
    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (paddle.Tensor): video clip must be normalized. Size is (C, T, H, W)
        """
        clip = clip.transpose((1, 0, 2, 3))
        return normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, inplace={self.inplace})"


class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (paddle.Tensor, dtype=paddle.uint8): Size is (T, C, H, W)
        Return:
            clip (paddle.Tensor, dtype=paddle.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


class RandomHorizontalFlipVideo:
    """
    Flip the video clip along the horizontal direction with a given probability
    Args:
        p (float): probability of the clip being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            clip (paddle.Tensor): Size is (T, C, H, W)
        Return:
            clip (paddle.Tensor): Size is (T, C, H, W)
        """
        if random.random() < self.p:
            clip = hflip(clip)
        return clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


#  ------------------------------------------------------------
#  ---------------------  Sampling  ---------------------------
#  ------------------------------------------------------------
class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    Args:
            size (int): Desired length of frames will be seen in the model.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, total_frames):
        rand_end = max(0, total_frames - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, total_frames)
        return begin_index, end_index

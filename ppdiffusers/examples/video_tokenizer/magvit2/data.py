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

from functools import partial
from pathlib import Path

import cv2
import numpy as np
import paddle
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import List, Tuple
from einops import rearrange
from paddle.vision.transforms import functional as F
from PIL import Image
from utils import _FUNCTIONAL_PAD


def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


def pair(val):
    return val if isinstance(val, tuple) else (val, val)


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = -dim - 1 if dim < 0 else t.ndim - dim - 1
    zeros = (0, 0) * dims_from_right
    return _FUNCTIONAL_PAD(pad=(*zeros, *pad), value=value, x=t)


def cast_num_frames(t, *, frames):
    f = tuple(t.shape)[-3]
    if f == frames:
        return t
    if f > frames:
        return t[(...), :frames, :, :]
    return pad_at_dim(t, (0, frames - f), dim=-3)


def convert_image_to_fn(img_type, image):
    if not exists(img_type) or image.mode == img_type:
        return image
    return image.convert(img_type)


def append_if_no_suffix(path: str, suffix: str):
    path = Path(path)
    if path.suffix == "":
        path = path.parent / (path.name + suffix)
    assert path.suffix == suffix, f"{str(path)} needs to have suffix {suffix}"
    return str(path)


CHANNEL_TO_MODE = {(1): "L", (3): "RGB", (4): "RGBA"}


class ImageDataset(paddle.io.Dataset):
    def __init__(self, folder, image_size, channels=3, convert_image_to=None, exts=["jpg", "jpeg", "png"]):
        super().__init__()
        folder = Path(folder)
        assert folder.is_dir(), f"{str(folder)} must be a folder containing images"
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        exts = exts + [ext.upper() for ext in exts]
        self.paths = [p for ext in exts for p in folder.glob(f"**/*.{ext}")]
        print(f"{len(self.paths)} training samples found at {folder}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

    def transform(self, img):
        img = convert_image_to_fn(CHANNEL_TO_MODE.get(self.channels), img)
        img = F.resize(img, size=self.image_size)
        img = F.center_crop(img, output_size=self.image_size)
        img = F.paddle.vision.transforms.RandomHorizontalFlip()(img)
        img = F.to_tensor(img)
        return img


def seek_all_images(img: paddle.Tensor, channels=3):
    mode = CHANNEL_TO_MODE.get(channels)
    assert exists(mode), f"channels {channels} invalid"
    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1


def tensor_to_PIL(tensor: paddle.Tensor):
    image = tensor.cpu().numpy().astype(np.uint8)
    return Image.fromarray(image.transpose(1, 2, 0)).convert("RGB")


# @beartype
def video_tensor_to_gif(tensor: paddle.Tensor, path: str, duration=120, loop=0, optimize=True):
    path = append_if_no_suffix(path, ".gif")
    images = map(tensor_to_PIL, tensor.unbind(axis=1))
    first_img, *rest_imgs = images
    first_img.save(str(path), save_all=True, append_images=rest_imgs, duration=duration, loop=loop, optimize=optimize)
    return images


def gif_to_tensor(path: str, channels=3, transform=F.to_tensor):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return paddle.stack(x=tensors, axis=1)


def video_to_tensor(path: str, num_frames=-1, crop_size=None) -> paddle.Tensor:
    video = cv2.VideoCapture(path)
    frames = []
    check = True
    while check:
        check, frame = video.read()
        if not check:
            continue
        if exists(crop_size):
            frame = crop_center(frame, *pair(crop_size))
        frames.append(rearrange(frame, "... -> 1 ..."))
    frames = np.array(np.concatenate(frames[:-1], axis=0))
    frames = rearrange(frames, "f h w c -> c f h w")
    frames_paddle = paddle.to_tensor(data=frames).astype(dtype="float32")
    frames_paddle /= 255.0
    frames_paddle = frames_paddle.flip(axis=(0,))
    return frames_paddle[:, :num_frames, :, :]


@beartype
def tensor_to_video(tensor: paddle.Tensor, path: str, fps=25, video_format="MP4V"):
    path = append_if_no_suffix(path, ".mp4")
    tensor = tensor.cpu()
    num_frames, height, width = tuple(tensor.shape)[-3:]
    fourcc = cv2.VideoWriter_fourcc(*video_format)
    video = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    for idx in range(num_frames):
        numpy_frame = tensor[:, (idx), :, :].numpy()
        numpy_frame = np.uint8(rearrange(numpy_frame, "c h w -> h w c"))
        video.write(numpy_frame)
    video.release()
    cv2.destroyAllWindows()
    return video


def crop_center(img: paddle.Tensor, cropx: int, cropy: int) -> paddle.Tensor:
    y, x, c = tuple(img.shape)
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty : starty + cropy, startx : startx + cropx, :]


class VideoDataset(paddle.io.Dataset):
    def __init__(self, folder, image_size, channels=3, num_frames=17, force_num_frames=True, exts=["gif", "mp4"]):
        super().__init__()
        folder = Path(folder)
        assert folder.is_dir(), f"{str(folder)} must be a folder containing videos"
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in folder.glob(f"**/*.{ext}")]
        print(f"{len(self.paths)} training samples found at {folder}")
        self.gif_to_tensor = partial(gif_to_tensor, channels=self.channels, transform=self.transform)
        self.mp4_to_tensor = partial(video_to_tensor, crop_size=self.image_size)
        self.cast_num_frames_fn = partial(cast_num_frames, frames=num_frames) if force_num_frames else identity

    def transform(self, img):
        img = F.resize(img, size=self.image_size)
        img = F.center_crop(img, output_size=self.image_size)
        img = F.to_tensor(img)
        return img

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        ext = path.suffix
        path_str = str(path)
        if ext == ".gif":
            tensor = self.gif_to_tensor(path_str)
        elif ext == ".mp4":
            tensor = self.mp4_to_tensor(path_str)
            frames = tensor.unbind(axis=1)
            tensor = paddle.stack(x=[*map(self.transform, frames)], axis=1)
        else:
            raise ValueError(f"unknown extension {ext}")
        return self.cast_num_frames_fn(tensor)


def collate_tensors_and_strings(data):
    if is_bearable(data, List[paddle.Tensor]):
        return (paddle.stack(x=data),)
    data = zip(*data)
    output = []
    for datum in data:
        if is_bearable(datum, Tuple[paddle.Tensor, ...]):
            datum = paddle.stack(x=datum)
        elif is_bearable(datum, Tuple[str, ...]):
            datum = list(datum)
        else:
            raise ValueError("detected invalid type being passed from dataset")
        output.append(datum)
    return tuple(output)

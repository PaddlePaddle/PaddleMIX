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

import os
import re
from collections.abc import Sequence

import cv2
import imageio
import numpy as np
import paddle
import pandas as pd
import requests
from paddle.vision import transforms
from paddle.vision.datasets.folder import IMG_EXTENSIONS, pil_loader
from PIL import Image

from . import video_transforms

VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

regex = re.compile(
    r"^(?:http|ftp)s?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


def read_file(input_path):
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path)
    elif input_path.endswith(".parquet"):
        return pd.read_parquet(input_path)
    else:
        raise NotImplementedError(f"Unsupported file format: {input_path}")


def is_url(url):
    return re.match(regex, url) is not None


def temporal_random_crop(vframes, num_frames, frame_interval):
    temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
    total_frames = len(vframes)
    start_frame_ind, end_frame_ind = temporal_sample(total_frames)
    assert end_frame_ind - start_frame_ind >= num_frames
    frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, num_frames, dtype=int)
    video = vframes[frame_indice]
    return video


def download_url(input_path):
    output_dir = "cache"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, base_name)
    img_data = requests.get(input_path).content
    with open(output_path, "wb") as handler:
        handler.write(img_data)
    print(f"URL {input_path} downloaded to {output_path}")
    return output_path


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def resize_crop_to_fill(pil_image, image_size):
    w, h = pil_image.size  # PIL is (W, H)
    th, tw = image_size
    rh, rw = th / h, tw / w
    if rh > rw:
        sh, sw = th, round(w * rh)
        image = pil_image.resize((sw, sh), Image.BICUBIC)
        i = 0
        j = int(round((sw - tw) / 2.0))
    else:
        sh, sw = round(h * rw), tw
        image = pil_image.resize((sw, sh), Image.BICUBIC)
        i = int(round((sh - th) / 2.0))
        j = 0
    arr = np.array(image)
    assert i + th <= arr.shape[0] and j + tw <= arr.shape[1]
    return Image.fromarray(arr[i : i + th, j : j + tw])


class Lambda(transforms.BaseTransform):
    def __init__(self, lambd, keys=None):
        if keys is None:
            keys = ("image",)
        elif not isinstance(keys, Sequence):
            raise ValueError(f"keys should be a sequence, but got keys={keys}")
        for k in keys:
            if self._get_apply(k) is None:
                raise NotImplementedError(f"{k} is unsupported data structure")
        self.keys = keys

        if not callable(lambd):
            raise TypeError("Argument lambd should be callable, got {}".format(repr(type(lambd).__name__)))
        self.lambd = lambd

    def _apply_image(self, img):
        return self.lambd(img)


def get_transforms_image(name="center", image_size=(256, 256)):
    if name is None:
        return None
    elif name == "center":
        assert image_size[0] == image_size[1], "Image size must be square for center crop"
        transform = transforms.Compose(
            [
                Lambda(lambda pil_image: center_crop_arr(pil_image, image_size[0])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    elif name == "resize_crop":
        transform = transforms.Compose(
            [
                Lambda(lambda pil_image: resize_crop_to_fill(pil_image, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        raise NotImplementedError(f"Transform {name} not implemented")
    return transform


def get_transforms_video(name="center", image_size=(256, 256)):
    if name is None:
        return None
    elif name == "center":
        assert image_size[0] == image_size[1], "image_size must be square for center crop"
        transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                video_transforms.UCFCenterCropVideo(image_size[0]),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    elif name == "resize_crop":
        transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                video_transforms.ResizeCrop(image_size),
                video_transforms.NormalizeVideo(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        raise NotImplementedError(f"Transform {name} not implemented")
    return transform_video


def read_image_from_path(path, transform=None, transform_name="center", num_frames=1, image_size=(256, 256)):
    image = pil_loader(path)
    if transform is None:
        transform = get_transforms_image(image_size=image_size, name=transform_name)
    image = transform(image)
    video = image.unsqueeze(0).tile((num_frames, 1, 1, 1))
    video = video.transpose([1, 0, 2, 3])
    return video


def read_video_with_opencv(filename, pts_unit="sec"):
    cap = cv2.VideoCapture(filename)

    vframes = []
    aframes = []
    info = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        vframes.append(np.expand_dims(frame, axis=0))

    info["fps"] = cap.get(cv2.CAP_PROP_FPS)
    info["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    vframes = np.concatenate(vframes, axis=0)
    vframes = np.transpose(vframes, (0, 3, 1, 2))

    if pts_unit == "sec":
        pass

    return vframes, aframes, info


def read_video_from_path(path, transform=None, transform_name="center", image_size=(256, 256)):

    vframes, aframes, info = read_video_with_opencv(filename=path, pts_unit="sec")  # output_format="TCHW"

    vframes_tensor = paddle.to_tensor(vframes, place=paddle.CPUPlace())

    if transform is None:
        transform = get_transforms_video(image_size=image_size, name=transform_name)
    video = transform(vframes_tensor)  # T C H W

    video = video.transpose((1, 0, 2, 3))
    return video


def read_from_path(path, image_size, transform_name="center"):
    if is_url(path):
        path = download_url(path)
    ext = os.path.splitext(path)[-1].lower()
    if ext.lower() in VID_EXTENSIONS:
        return read_video_from_path(path, image_size=image_size, transform_name=transform_name)
    else:
        assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
    return read_image_from_path(path, image_size=image_size, transform_name=transform_name)


def save_sample(x, fps=8, save_path=None, normalize=True, value_range=(-1.0, 1.0)):
    """
    Saves a video sample from a tensor without using OpenCV.

    Args:
        x (Tensor): Tensor of shape [C, T, H, W].
        fps (int, optional): Frames per second for the saved video. Defaults to 8.
        save_path (str, optional): Path to save the video. If None, a default path is used.
        normalize (bool, optional): Whether to normalize the tensor values. Defaults to True.
        value_range (tuple, optional): Tuple specifying the (min, max) range for normalization. Defaults to (-1.0, 1.0).

    Returns:
        str: The path where the video is saved.
    """
    assert x.ndim == 4, f"Expected tensor with 4 dimensions [C, T, H, W], but got {x.ndim} dimensions."

    if save_path is None:
        raise ValueError("save_path must be provided.")

    save_path += ".mp4"

    if normalize:
        low, high = paddle.to_tensor(value_range, dtype="float32")
        x = x.clip(min=low, max=high)
        x = (x - low) / paddle.maximum(high - low, paddle.to_tensor(1e-5))

    # Scale to [0, 255] and convert to uint8
    video_data = (
        x.multiply(paddle.to_tensor(255.0, dtype="float32"))
        .add(paddle.to_tensor(0.5, dtype="float32"))  # For rounding
        .clip(0, 255)
    )
    video_data = video_data.transpose([1, 2, 3, 0])
    video_data = video_data.numpy().astype(np.uint8)

    frames, height, width, channels = video_data.shape

    # Initialize the video writer using imageio
    writer = imageio.get_writer(save_path, fps=fps, codec="libx264", format="mp4")

    try:
        for i in range(frames):
            frame = video_data[i]
            # Ensure frame is in RGB format
            writer.append_data(frame)
    finally:
        writer.close()

    print(f"Saved to {save_path}")
    return save_path

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

import os

import cv2
import numpy as np
import paddle
from tqdm import tqdm

try:
    import accimage
except ImportError:
    accimage = None
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

try:
    import av

    av.logging.set_level(av.logging.ERROR)
    if not hasattr(av.video.frame.VideoFrame, "pict_type"):
        av = ImportError("""Your version of PyAV is too old for the necessary video operations.""")
except ImportError:
    av = ImportError("""PyAV is not installed, and is necessary for the video operations.""")


def _check_av_available() -> None:
    if isinstance(av, Exception):
        raise av


def write_video(
    filename: str,
    video_array: paddle.Tensor,
    fps: float,
    video_codec: str = "libx264",
    options: Optional[Dict[str, Any]] = None,
    audio_array: Optional[paddle.Tensor] = None,
    audio_fps: Optional[float] = None,
    audio_codec: Optional[str] = None,
    audio_options: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file

    Args:
        filename (str): path where the video will be saved
        video_array (Tensor[T, H, W, C]): tensor containing the individual frames,
            as a uint8 tensor in [T, H, W, C] format
        fps (Number): video frames per second
        video_codec (str): the name of the video codec, i.e. "libx264", "h264", etc.
        options (Dict): dictionary containing options to be passed into the PyAV video stream
        audio_array (Tensor[C, N]): tensor containing the audio, where C is the number of channels
            and N is the number of samples
        audio_fps (Number): audio sample rate, typically 44100 or 48000
        audio_codec (str): the name of the audio codec, i.e. "mp3", "aac", etc.
        audio_options (Dict): dictionary containing options to be passed into the PyAV audio stream
    """
    _check_av_available()
    video_array = paddle.to_tensor(data=video_array).astype("uint8").numpy()
    if isinstance(fps, float):
        fps = np.round(fps)
    with av.open(filename, mode="w") as container:
        stream = container.add_stream(video_codec, rate=fps)
        stream.width = video_array.shape[2]
        stream.height = video_array.shape[1]
        stream.pix_fmt = "yuv420p" if video_codec != "libx264rgb" else "rgb24"
        stream.options = options or {}
        if audio_array is not None:
            audio_format_dtypes = {
                "dbl": "<f8",
                "dblp": "<f8",
                "flt": "<f4",
                "fltp": "<f4",
                "s16": "<i2",
                "s16p": "<i2",
                "s32": "<i4",
                "s32p": "<i4",
                "u8": "u1",
                "u8p": "u1",
            }
            a_stream = container.add_stream(audio_codec, rate=audio_fps)
            a_stream.options = audio_options or {}
            num_channels = audio_array.shape[0]
            audio_layout = "stereo" if num_channels > 1 else "mono"
            audio_sample_fmt = container.streams.audio[0].format.name
            format_dtype = np.dtype(audio_format_dtypes[audio_sample_fmt])
            audio_array = paddle.to_tensor(data=audio_array).numpy().astype(format_dtype)
            frame = av.AudioFrame.from_ndarray(audio_array, format=audio_sample_fmt, layout=audio_layout)
            frame.sample_rate = audio_fps
            for packet in a_stream.encode(frame):
                container.mux(packet)
            for packet in a_stream.encode():
                container.mux(packet)
        for img in video_array:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pict_type = "NONE"
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@paddle.no_grad()
def make_grid(
    tensor: Union[paddle.Tensor, List[paddle.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
) -> paddle.Tensor:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Returns:
        grid (Tensor): the tensor containing grid of images.
    """
    if not paddle.is_tensor(x=tensor):
        if isinstance(tensor, list):
            for t in tensor:
                if not paddle.is_tensor(x=t):
                    raise TypeError(f"tensor or list of tensors expected, got a list containing {type(t)}")
        else:
            raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")
    if isinstance(tensor, list):
        tensor = paddle.stack(x=tensor, axis=0)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(axis=0)
    if tensor.dim() == 3:
        if tensor.shape[0] == 1:
            tensor = paddle.concat(x=(tensor, tensor, tensor), axis=0)
        tensor = tensor.unsqueeze(axis=0)
    if tensor.dim() == 4 and tensor.shape[1] == 1:
        tensor = paddle.concat(x=(tensor, tensor, tensor), axis=1)
    if normalize is True:
        tensor = tensor.clone()
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img.clip_(min=low, max=high)
            img = img.subtract(low).divide(max(high - low, 1e-05))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)
    if not isinstance(tensor, paddle.Tensor):
        raise TypeError("tensor should be of type paddle.Tensor")
    if tensor.shape[0] == 1:
        return tensor.squeeze(axis=0)
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = paddle.full(
        shape=(num_channels, height * ymaps + padding, width * xmaps + padding),
        fill_value=pad_value,
        dtype=tensor.dtype,
    )
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            start_0 = grid.shape[1] + y * height + padding if y * height + padding < 0 else y * height + padding
            start_1 = (
                paddle.slice(grid, [1], [start_0], [start_0 + height - padding]).shape[2] + x * width + padding
                if x * width + padding < 0
                else x * width + padding
            )
            paddle.assign(
                tensor[k],
                output=paddle.slice(
                    paddle.slice(grid, [1], [start_0], [start_0 + height - padding]),
                    [2],
                    [start_1],
                    [start_1 + width - padding],
                ),
            )
            k = k + 1
    return grid


def _is_pil_image(img: Any) -> bool:
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def get_image_num_channels(img: Any) -> int:
    if _is_pil_image(img):
        if hasattr(img, "getbands"):
            return len(img.getbands())
        else:
            return img.channels
    raise TypeError(f"Unexpected type {type(img)}")


def to_tensor(pic) -> paddle.Tensor:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    See :class:`~paddle.vision.transforms.ToTensor` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    default_float_dtype = paddle.get_default_dtype()
    if isinstance(pic, np.ndarray):
        if pic.ndim == 2:
            pic = pic[:, :, (None)]
        img = paddle.to_tensor(data=pic.transpose((2, 0, 1)))
        if img.dtype == paddle.uint8:
            return paddle.divide(img.cast(default_float_dtype), paddle.to_tensor(255, dtype=paddle.float32))
        else:
            return img
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = paddle.to_tensor(data=np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    if pic.mode == "1":
        img = 255 * img
    img = img.reshape([pic.size[1], pic.size[0], get_image_num_channels(pic)])
    img = img.transpose(perm=(2, 0, 1))
    if img.dtype == paddle.uint8:
        return paddle.divide(img.cast(default_float_dtype), 255)
    else:
        return img


def load_num_videos(data_path, num_videos):
    if isinstance(data_path, str):
        videos = np.load(data_path)["arr_0"]
    elif isinstance(data_path, np.ndarray):
        videos = data_path
    else:
        raise Exception
    if num_videos is not None:
        videos = videos[:num_videos, :, :, :, :]
    return videos


def fill_with_black_squares(video, desired_len: int) -> paddle.Tensor:
    if len(video) >= desired_len:
        return video
    return paddle.concat(
        x=[
            video,
            paddle.zeros_like(x=video[0]).unsqueeze(axis=0).tile(repeat_times=[desired_len - len(video), 1, 1, 1]),
        ],
        axis=0,
    )


def npz_to_video_grid(data_path, out_path, num_frames=None, fps=8, num_videos=None, nrow=None, verbose=True):
    if isinstance(data_path, str):
        videos = load_num_videos(data_path, num_videos)
    elif isinstance(data_path, np.ndarray):
        videos = data_path
    else:
        raise Exception
    n, t, h, w, c = videos.shape
    videos_th = []
    for i in range(n):
        video = videos[(i), :, :, :, :]
        images = [video[(j), :, :, :] for j in range(t)]
        images = [to_tensor(img) for img in images]

        video = paddle.stack(x=images)
        videos_th.append(video)
    if num_frames is None:
        num_frames = videos.shape[1]
    if verbose:
        videos = [fill_with_black_squares(v, num_frames) for v in tqdm(videos_th, desc="Adding empty frames")]
    else:
        videos = [fill_with_black_squares(v, num_frames) for v in videos_th]
    frame_grids = paddle.stack(x=videos).transpose(perm=[1, 0, 2, 3, 4])
    if nrow is None:
        nrow = int(np.ceil(np.sqrt(n)))
    if verbose:
        frame_grids = [make_grid(fs, nrow=nrow) for fs in tqdm(frame_grids, desc="Making grids")]

    else:
        frame_grids = [make_grid(fs, nrow=nrow) for fs in frame_grids]

    if os.path.dirname(out_path) != "":
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if isinstance("uint8", paddle.dtype):
        dtype = "uint8"
    elif isinstance("uint8", str) and "uint8" not in ["cpu", "cuda", "ipu", "xpu"]:
        dtype = "uint8"
    elif isinstance("uint8", paddle.Tensor):
        dtype = "uint8".dtype
    else:
        dtype = (paddle.stack(x=frame_grids) * 255).dtype
    frame_grids = (paddle.stack(x=frame_grids) * 255).transpose(perm=[0, 2, 3, 1]).cast(dtype)
    write_video(out_path, frame_grids, fps=fps, video_codec="h264", options={"crf": "10"})


def savenp2sheet(imgs, savepath, nrow=None):
    """save multiple imgs (in numpy array type) to a img sheet.
        img sheet is one row.

    imgs:
        np array of size [N, H, W, 3] or List[array] with array size = [H,W,3]
    """
    if imgs.ndim == 4:
        img_list = [imgs[i] for i in range(imgs.shape[0])]
        imgs = img_list
    imgs_new = []
    for i, img in enumerate(imgs):
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        assert img.ndim == 3 and img.shape[-1] == 3, img.shape
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imgs_new.append(img)
    n = len(imgs)
    if nrow is not None:
        n_cols = nrow
    else:
        n_cols = int(n**0.5)
    n_rows = int(np.ceil(n / n_cols))
    print(n_cols)
    print(n_rows)
    imgsheet = cv2.vconcat([cv2.hconcat(imgs_new[i * n_cols : (i + 1) * n_cols]) for i in range(n_rows)])
    cv2.imwrite(savepath, imgsheet)
    print(f"saved in {savepath}")


def npz_to_imgsheet_5d(data_path, res_dir, nrow=None):
    if isinstance(data_path, str):
        imgs = np.load(data_path)["arr_0"]
    elif isinstance(data_path, np.ndarray):
        imgs = data_path
    else:
        raise Exception
    if os.path.isdir(res_dir):
        res_path = os.path.join(res_dir, "samples.jpg")
    else:
        assert res_dir.endswith(".jpg")
        res_path = res_dir
    imgs = np.concatenate([imgs[i] for i in range(imgs.shape[0])], axis=0)
    savenp2sheet(imgs, res_path, nrow=nrow)


def save_results(
    videos,
    save_dir,
    save_name="results",
    save_fps=8,
    save_mp4=True,
    save_npz=False,
    save_mp4_sheet=False,
    save_jpg=False,
):
    if save_mp4:
        save_subdir = os.path.join(save_dir, "videos")
        os.makedirs(save_subdir, exist_ok=True)
        shape_str = "x".join([str(x) for x in videos[0:1, (...)].shape])
        for i in range(videos.shape[0]):
            npz_to_video_grid(
                videos[i : i + 1, (...)],
                os.path.join(save_subdir, f"{save_name}_{i:03d}_{shape_str}.mp4"),
                fps=save_fps,
            )
        print(f"Successfully saved videos in {save_subdir}")
    shape_str = "x".join([str(x) for x in videos.shape])
    if save_npz:
        save_path = os.path.join(save_dir, f"{save_name}_{shape_str}.npz")
        np.savez(save_path, videos)
        print(f"Successfully saved npz in {save_path}")
    if save_mp4_sheet:
        save_path = os.path.join(save_dir, f"{save_name}_{shape_str}.mp4")
        npz_to_video_grid(videos, save_path, fps=save_fps)
        print(f"Successfully saved mp4 sheet in {save_path}")
    if save_jpg:
        save_path = os.path.join(save_dir, f"{save_name}_{shape_str}.jpg")
        npz_to_imgsheet_5d(videos, save_path, nrow=videos.shape[1])
        print(f"Successfully saved jpg sheet in {save_path}")

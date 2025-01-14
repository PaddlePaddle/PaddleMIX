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
import random
import warnings
from threading import Thread

import decord
import numpy as np
import paddle
from PIL import Image
from tqdm import tqdm


def get_connected_components(mask):
    """
    Get the connected components (8-connectivity) of binary masks of shape (N, 1, H, W).

    Inputs:
    - mask: A binary mask tensor of shape (N, 1, H, W), where 1 is foreground and 0 is
            background.

    Outputs:
    - labels: A tensor of shape (N, 1, H, W) containing the connected component labels
              for foreground pixels and 0 for background pixels.
    - counts: A tensor of shape (N, 1, H, W) containing the area of the connected
              components for foreground pixels and 0 for background pixels.
    """
    from sam2 import _C

    return _C.get_connected_componnets(mask.to("uint8"))


def mask_to_box(masks: paddle.Tensor):
    """
    compute bounding box given an input mask

    Inputs:
    - masks: [B, 1, H, W] masks, dtype=torch.Tensor

    Returns:
    - box_coords: [B, 1, 4], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.Tensor
    """
    B, _, h, w = tuple(masks.shape)

    xs = paddle.arange(dtype="int32", end=w)
    ys = paddle.arange(dtype="int32", end=h)
    grid_xs, grid_ys = list([i.T for i in paddle.meshgrid(xs, ys)])
    grid_xs = grid_xs[None, None, ...].expand(shape=[B, 1, h, w])
    grid_ys = grid_ys[None, None, ...].expand(shape=[B, 1, h, w])
    min_xs, _ = paddle.min(
        x=paddle.where(condition=masks, x=grid_xs, y=w).flatten(start_axis=-2), axis=-1
    ), paddle.argmin(x=paddle.where(condition=masks, x=grid_xs, y=w).flatten(start_axis=-2), axis=-1)
    max_xs, _ = paddle.max(
        x=paddle.where(condition=masks, x=grid_xs, y=-1).flatten(start_axis=-2), axis=-1
    ), paddle.argmax(x=paddle.where(condition=masks, x=grid_xs, y=-1).flatten(start_axis=-2), axis=-1)
    min_ys, _ = paddle.min(
        x=paddle.where(condition=masks, x=grid_ys, y=h).flatten(start_axis=-2), axis=-1
    ), paddle.argmin(x=paddle.where(condition=masks, x=grid_ys, y=h).flatten(start_axis=-2), axis=-1)
    max_ys, _ = paddle.max(
        x=paddle.where(condition=masks, x=grid_ys, y=-1).flatten(start_axis=-2), axis=-1
    ), paddle.argmax(x=paddle.where(condition=masks, x=grid_ys, y=-1).flatten(start_axis=-2), axis=-1)
    bbox_coords = paddle.stack(x=(min_xs, min_ys, max_xs, max_ys), axis=-1)
    return bbox_coords


def _load_img_as_tensor(img_path, image_size):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img = paddle.to_tensor(data=img_np).transpose(perm=[2, 0, 1])
    video_width, video_height = img_pil.size

    return img_pil, img, video_height, video_width


class AsyncVideoFrameLoader:
    """
    A list of video frames to be load asynchronously without blocking session start.
    """

    def __init__(self, img_paths, image_size, offload_video_to_cpu, img_mean, img_std):
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        self.images = [None] * len(img_paths)
        self.exception = None
        self.video_height = None
        self.video_width = None

        self.__getitem__(0)

        def _load_frames():
            try:
                for n in tqdm(range(len(self.images)), desc="frame loading (JPEG)"):
                    self.__getitem__(n)
            except Exception as e:
                self.exception = e

        self.thread = Thread(target=_load_frames, daemon=True)
        self.thread.start()

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception
        img = self.images[index]
        if img is not None:
            return img
        img_pil, img, video_height, video_width = _load_img_as_tensor(self.img_paths[index], self.image_size)
        self.video_height = video_height
        self.video_width = video_width
        img -= self.img_mean
        img /= self.img_std

        self.images[index] = img
        return img

    def __len__(self):
        return len(self.images)


def load_video_frames(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
):
    """
    Load the video frames from video_path. The frames are resized to image_size as in
    the model and are loaded to GPU if offload_video_to_cpu=False. This is used by the demo.
    """
    is_bytes = isinstance(video_path, bytes)
    is_str = isinstance(video_path, str)
    is_mp4_path = is_str and os.path.splitext(video_path)[-1] in [".mp4", ".MP4"]
    if is_bytes or is_mp4_path:
        return load_video_frames_from_video_file(
            video_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
        )
    elif is_str and os.path.isdir(video_path):
        return load_video_frames_from_jpg_images(
            video_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
        )
    else:
        raise NotImplementedError("Only MP4 video and JPEG folder are supported at this moment")


def load_video_frames_from_jpg_images(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
):
    """
    Load the video frames from a directory of JPEG files ("<frame_index>.jpg" format).

    The frames are resized to image_size x image_size and are loaded to GPU if
    `offload_video_to_cpu` is `False` and to CPU if `offload_video_to_cpu` is `True`.

    You can load a frame asynchronously by setting `async_loading_frames` to `True`.
    """
    if isinstance(video_path, str) and os.path.isdir(video_path):
        jpg_folder = video_path
    else:
        raise NotImplementedError(
            """Only JPEG frames are supported at this moment. For video files, you may use ffmpeg (https://ffmpeg.org/) to extract frames into a folder of JPEG files"""
        )
    frame_names = [
        p for p in os.listdir(jpg_folder) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    num_frames = len(frame_names)
    if num_frames == 0:
        raise RuntimeError(f"no images found in {jpg_folder}")
    img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]
    img_mean = paddle.to_tensor(data=img_mean, dtype="float32")[:, None, None]
    img_std = paddle.to_tensor(data=img_std, dtype="float32")[:, None, None]
    if async_loading_frames:
        lazy_images = AsyncVideoFrameLoader(img_paths, image_size, offload_video_to_cpu, img_mean, img_std)
        return lazy_images, lazy_images.video_height, lazy_images.video_width
    images = paddle.zeros(shape=[num_frames, 3, image_size, image_size], dtype="float32")
    images_pil = []
    for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG)")):
        img_pil, images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
        images_pil.append(img_pil)

    images -= img_mean
    images /= img_std
    return images_pil, images, video_height, video_width, 0


def load_video_frames_from_video_file(
    video_path, image_size, offload_video_to_cpu, img_mean=(0.485, 0.456, 0.406), img_std=(0.229, 0.224, 0.225)
):
    """Load the video frames from a video file."""
    img_mean = paddle.to_tensor(data=img_mean, dtype="float32")[:, None, None]
    img_std = paddle.to_tensor(data=img_std, dtype="float32")[:, None, None]

    decord_vr = decord.VideoReader(video_path)
    total_frames = len(decord_vr)
    avg_fps = float(decord_vr.get_avg_fps())
    video_height, video_width, _ = tuple(decord_vr.next().shape)
    num_frames = total_frames
    if total_frames < num_frames:
        frame_id_list = list(range(total_frames))
    else:
        max_interval_frame_candidate = total_frames // num_frames
        max_interval_frame_candidate = min(avg_fps // 6, max_interval_frame_candidate)
        interval_frame = random.randint(1, max_interval_frame_candidate)
        segment_length = interval_frame * num_frames
        bg_frame_id = random.randint(0, total_frames - segment_length)
        frame_id_list = list(range(bg_frame_id, bg_frame_id + segment_length, interval_frame))
    video_data = decord_vr.get_batch(frame_id_list).asnumpy()
    trans = paddle.vision.transforms.Resize((image_size, image_size))
    images = []
    org_images = []
    for frame in video_data:
        org_images.append(frame)
        frame = paddle.to_tensor(frame).transpose(perm=[2, 0, 1])

        images.append(trans(frame))
    images = paddle.stack(x=images, axis=0).astype(dtype="float32") / 255.0

    images -= img_mean
    images /= img_std
    return org_images, images, video_height, video_width, avg_fps


def fill_holes_in_mask_scores(mask, max_area):
    """
    A post processor to fill small holes in mask scores with area under `max_area`.
    """
    assert max_area > 0, "max_area must be positive"
    input_mask = mask
    try:
        labels, areas = get_connected_components(mask <= 0)
        is_hole = (labels > 0) & (areas <= max_area)
        mask = paddle.where(condition=is_hole, x=0.1, y=mask)
    except Exception as e:
        warnings.warn(
            f"""{e}

Skipping the post-processing step due to the error above. You can still use SAM 2 and it's OK to ignore the error above, although some post-processing functionality may be limited (which doesn't affect the results in most cases; see https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).""",
            category=UserWarning,
            stacklevel=2,
        )
        mask = input_mask
    return mask


def concat_points(old_point_inputs, new_points, new_labels):
    """Add new points and labels to previous point inputs (add at the end)."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = paddle.concat(x=[old_point_inputs["point_coords"], new_points], axis=1)
        labels = paddle.concat(x=[old_point_inputs["point_labels"], new_labels], axis=1)
    return {"point_coords": points, "point_labels": labels}

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

import glob
import json
import os
import random

import paddle
from decord import VideoReader, cpu
from einops import rearrange

from ._transforms_video import CenterCropVideo, RandomCropVideo


class WebVidDataset(paddle.io.Dataset):
    """
    Taichi Dataset.
    Assumes data is structured as follows.
    Taichi/
        train/
            xxx.mp4
            ...
        test/
            xxx.mp4
            ...
    """

    def __init__(
        self,
        data_root,
        resolution,
        video_length,
        subset_split,
        frame_stride,
        spatial_transform="",
        load_method="decord",
        annotation_path=None,
        tokenizer=None,
    ):
        self.annotation_path = annotation_path
        self.data_root = data_root
        self.resolution = resolution
        self.video_length = video_length
        self.subset_split = subset_split
        self.frame_stride = frame_stride
        self.spatial_transform = spatial_transform
        self.load_method = load_method
        assert self.load_method in ["decord", "readvideo", "videoclips"]
        assert self.subset_split in ["train", "test", "all", "results_10M_train"]
        self.exts = ["avi", "mp4", "webm"]
        if isinstance(self.resolution, int):
            self.resolution = [self.resolution, self.resolution]
        assert isinstance(self.resolution, list) and len(self.resolution) == 2
        self.max_resolution = max(self.resolution)
        if self.spatial_transform == "center_crop_resize":
            print("Spatial transform: center crop and then resize")
            self.video_transform = paddle.vision.transforms.Compose(
                [
                    paddle.vision.transforms.Resize(resolution),
                    CenterCropVideo(resolution),
                ]
            )
            self.video_transform_step1 = paddle.vision.transforms.Compose(
                [
                    paddle.vision.transforms.Resize(resolution),
                ]
            )
            self.video_transform_step2 = paddle.vision.transforms.Compose([CenterCropVideo(resolution)])
        elif self.spatial_transform == "resize":
            print("Spatial transform: resize with no crop")
            self.video_transform = paddle.vision.transforms.Resize((resolution, resolution))
        elif self.spatial_transform == "random_crop":
            self.video_transform = paddle.vision.transforms.Compose([RandomCropVideo(resolution)])
        elif self.spatial_transform == "":
            self.video_transform = None
        else:
            raise NotImplementedError
        self._make_dataset()

        if tokenizer:
            self.text_processing = lambda caption: tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pd",
                return_overflowing_tokens=False,
            ).input_ids[0]
        else:
            self.text_processing = None

    def _make_dataset(self):
        if self.subset_split == "all":
            data_folder = self.data_root
        else:
            data_folder = os.path.join(self.data_root, self.subset_split)
        if self.annotation_path is not None:
            print("Reading annotation_path as FILE type ...")
            with open(self.annotation_path, "r") as fp:
                self.annotations = fp.read().splitlines()
        else:
            self.annotations = sum(
                [glob.glob(os.path.join(data_folder, "**", f"*.{ext}"), recursive=True) for ext in self.exts],
                [],
            )
        print(f"Number of videos = {len(self.annotations)}")

    def get_annotation(self, index):
        if self.annotation_path is not None:
            annotation = json.loads(self.annotations[index])
            caption = annotation["name"]
            if self.subset_split == "all":
                video_path = annotation["afs_path"]
            else:
                video_path = annotation["afs_path"]
        else:
            caption = "caption"
            video_path = self.annotations[index]
        return caption, video_path

    def get_data_decord(self, index):
        while True:
            caption, video_path = self.get_annotation(index)
            try:
                video_reader = VideoReader(
                    video_path,
                    ctx=cpu(0),
                    width=self.max_resolution,
                    height=self.max_resolution,
                )
                if len(video_reader) < self.video_length:
                    index += 1
                    continue
                else:
                    break
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
        all_frames = list(range(0, len(video_reader), self.frame_stride))
        if len(all_frames) < self.video_length:
            all_frames = list(range(0, len(video_reader), 1))
        rand_idx = random.randint(0, len(all_frames) - self.video_length)
        frame_indices = list(range(rand_idx, rand_idx + self.video_length))
        frames = video_reader.get_batch(frame_indices)
        assert frames.shape[0] == self.video_length, f"{len(frames)}, self.video_length={self.video_length}"
        frames = paddle.to_tensor(data=frames.asnumpy()).astype(dtype="float32").transpose(perm=[0, 3, 1, 2])
        if self.video_transform is not None:
            if self.spatial_transform == "center_crop_resize":
                temp_frames = rearrange(frames, "c t h w -> (c t) h w")
                temp_frames = self.video_transform_step1(temp_frames)
                frames = rearrange(temp_frames, "(c t) h w -> c t h w", c=frames.shape[0])
                frames = self.video_transform_step2(frames)
            else:
                frames = self.video_transform(frames)
        frames = frames.transpose(perm=[1, 0, 2, 3]).astype(dtype="float32")
        assert (
            frames.shape[2] == self.resolution[0] and frames.shape[3] == self.resolution[1]
        ), f"frames={frames.shape}, self.resolution={self.resolution}"
        frames = (frames / 255 - 0.5) * 2
        data = {"video": frames, "caption": caption}

        if self.text_processing:
            tensor_out = {
                "pixel_values": data["video"],
                "input_ids": self.text_processing(data["caption"]),
            }
        else:
            tensor_out = {
                "pixel_values": data["video"],
            }
        return tensor_out

    def get_data_readvideo(self, index):
        return

    def __getitem__(self, index):
        if self.load_method == "decord":
            data = self.get_data_decord(index)
        elif self.load_method == "readvideo":
            data = self.get_data_readvideo(index)
        return data

    def __len__(self):
        return len(self.annotations)


def main():
    import time

    data_root = ""
    annotation_path = ""
    resolution = 256
    video_length = 16
    subset_split = "all"
    frame_stride = 4
    spatial_transform = "center_crop_resize"
    dataset = WebVidDataset(
        data_root=data_root,
        resolution=resolution,
        video_length=video_length,
        subset_split=subset_split,
        frame_stride=frame_stride,
        spatial_transform=spatial_transform,
        annotation_path=annotation_path,
    )
    dataloader = paddle.io.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    starttime = time.time()
    for id, data in enumerate(dataloader):
        endtime = time.time()
        print(
            id,
            "time:",
            endtime - starttime,
            " shape:",
            data["video"].shape,
            data["caption"],
        )
        starttime = endtime
    return


if __name__ == "__main__":
    main()

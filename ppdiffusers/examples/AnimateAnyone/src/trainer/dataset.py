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

import json
import random
from typing import List

import numpy as np
import paddle
import paddle.vision.transforms as transforms
from decord import VideoReader
from paddlenlp.transformers import CLIPImageProcessor
from PIL import Image


class HumanDanceDataset(paddle.io.Dataset):
    def __init__(
        self,
        img_size,
        img_scale=(1.0, 1.0),
        img_ratio=(1.0, 1.0),
        drop_ratio=0.1,
        data_meta_paths="./ubcNbili_data/meta_data/ubcNbili_meta.json",
        sample_margin=30,
    ):
        super().__init__()

        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.sample_margin = sample_margin

        # -----
        # vid_meta format:
        # [{'video_path': , 'kps_path': , 'other':},
        #  {'video_path': , 'kps_path': , 'other':}]
        # -----
        vid_meta = []

        vid_meta.extend(json.load(open(data_meta_paths, "r")))

        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation="bilinear",
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation="bilinear",
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, image, transform, state=None):
        if state is not None:
            paddle.set_cuda_rng_state(state_list=state)
        return transform(image)

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]

        video_reader = VideoReader(video_path)
        kps_reader = VideoReader(kps_path)

        assert len(video_reader) == len(kps_reader), f"{len(video_reader) = } != {len(kps_reader) = } in {video_path}"

        video_length = len(video_reader)

        margin = min(self.sample_margin, video_length)

        ref_img_idx = random.randint(0, video_length - 1)
        if ref_img_idx + margin < video_length:
            tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
        elif ref_img_idx - margin > 0:
            tgt_img_idx = random.randint(0, ref_img_idx - margin)
        else:
            tgt_img_idx = random.randint(0, video_length - 1)

        ref_img = video_reader[ref_img_idx]
        ref_img_pil = Image.fromarray(ref_img.asnumpy())
        tgt_img = video_reader[tgt_img_idx]
        tgt_img_pil = Image.fromarray(tgt_img.asnumpy())

        tgt_pose = kps_reader[tgt_img_idx]
        tgt_pose_pil = Image.fromarray(tgt_pose.asnumpy())

        state = paddle.get_cuda_rng_state()

        tgt_img = self.augmentation(tgt_img_pil, self.transform, state)
        tgt_pose_img = self.augmentation(tgt_pose_pil, self.cond_transform, state)
        ref_img_vae = self.augmentation(ref_img_pil, self.transform, state)
        clip_image = self.clip_image_processor(images=ref_img_pil, return_tensors="pd").pixel_values[0]

        sample = dict(
            video_dir=video_path,
            img=tgt_img,
            tgt_pose=tgt_pose_img,
            ref_img=ref_img_vae,
            clip_images=clip_image,
        )

        tensor_out = {
            "batch": sample,
        }

        return tensor_out

    def __len__(self):
        return len(self.vid_meta)


class HumanDanceVideoDataset(paddle.io.Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        width,
        height,
        img_scale=(1.0, 1.0),
        img_ratio=(1.0, 1.0),
        drop_ratio=0.1,
        data_meta_paths="./ubcNbili_data/meta_data/ubcNbili_meta.json",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio

        vid_meta = []
        vid_meta.extend(json.load(open(data_meta_paths, "r")))

        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation="bilinear",
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation="bilinear",
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, images, transform, state=None):
        if state is not None:
            paddle.set_cuda_rng_state(state_list=state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = paddle.stack(transformed_images, axis=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]

        video_reader = VideoReader(video_path)
        kps_reader = VideoReader(kps_path)

        assert len(video_reader) == len(kps_reader), f"{len(video_reader) = } != {len(kps_reader) = } in {video_path}"

        video_length = len(video_reader)

        clip_length = min(video_length, (self.n_sample_frames - 1) * self.sample_rate + 1)
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int).tolist()

        # read frames and kps
        vid_pil_image_list = []
        pose_pil_image_list = []
        for index in batch_index:
            img = video_reader[index]
            vid_pil_image_list.append(Image.fromarray(img.asnumpy()))
            img = kps_reader[index]
            pose_pil_image_list.append(Image.fromarray(img.asnumpy()))

        ref_img_idx = random.randint(0, video_length - 1)
        ref_img = Image.fromarray(video_reader[ref_img_idx].asnumpy())

        # transform
        state = paddle.get_cuda_rng_state()
        pixel_values_vid = self.augmentation(vid_pil_image_list, self.pixel_transform, state)
        pixel_values_pose = self.augmentation(pose_pil_image_list, self.cond_transform, state)
        pixel_values_ref_img = self.augmentation(ref_img, self.pixel_transform, state)
        clip_ref_img = self.clip_image_processor(images=ref_img, return_tensors="pd").pixel_values[0]

        sample = dict(
            video_dir=video_path,
            pixel_values_vid=pixel_values_vid,
            pixel_values_pose=pixel_values_pose,
            pixel_values_ref_img=pixel_values_ref_img,
            clip_ref_img=clip_ref_img,
        )

        # return sample
        tensor_out = {
            "batch": sample,
        }

        return tensor_out

    def __len__(self):
        return len(self.vid_meta)

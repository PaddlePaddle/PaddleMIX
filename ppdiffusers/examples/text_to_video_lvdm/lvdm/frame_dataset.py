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
import random
import re

import paddle
from einops import rearrange
from PIL import Image, ImageFile

from ._transforms_video import CenterCropVideo, RandomCropVideo

""" VideoFrameDataset """
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def pil_loader(path):
    """
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
    """
    Im = Image.open(path)
    return Im.convert("RGB")


def accimage_loader(path):
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
    """
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
    """
    return pil_loader(path)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    assert os.path.exists(dir), f"{dir} does not exist"
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def class_name_to_idx(annotation_dir):
    """
    return class indices from 0 ~ num_classes-1
    """
    fpath = os.path.join(annotation_dir, "classInd.txt")
    with open(fpath, "r") as f:
        data = f.readlines()
        class_to_idx = {x.strip().split(" ")[1].lower(): int(x.strip().split(" ")[0]) - 1 for x in data}
    return class_to_idx


def make_dataset(dir, nframes, class_to_idx, frame_stride=1, **kwargs):
    """
    videos are saved in second-level directory:
    dir: video dir. Format:
        videoxxx
            videoxxx_1
                frame1.jpg
                frame2.jpg
            videoxxx_2
                frame1.jpg
                ...
        videoxxx

    nframes: num of frames of every video clips
    class_to_idx: for mapping video name to video id
    """
    if frame_stride != 1:
        raise NotImplementedError
    clips = []
    videos = []
    n_clip = 0
    video_frames = []
    for video_name in sorted(os.listdir(dir)):
        if os.path.isdir(os.path.join(dir, video_name)):
            subfolder_path = os.path.join(dir, video_name)
            for subsubfold in sorted(os.listdir(subfolder_path)):
                subsubfolder_path = os.path.join(subfolder_path, subsubfold)
                if os.path.isdir(subsubfolder_path):
                    clip_frames = []
                    i = 1
                    for fname in sorted(os.listdir(subsubfolder_path)):
                        if is_image_file(fname):
                            img_path = os.path.join(subsubfolder_path, fname)
                            frame_info = img_path, class_to_idx[video_name]
                            clip_frames.append(frame_info)
                            video_frames.append(frame_info)
                            if i % nframes == 0 and i > 0:
                                clips.append(clip_frames)
                                n_clip += 1
                                clip_frames = []
                            i = i + 1
                    if len(video_frames) >= nframes:
                        videos.append(video_frames)
                    video_frames = []
    print("number of long videos:", len(videos))
    print("number of short videos", len(clips))
    return clips, videos


def split_by_captical(s):
    s_list = re.sub("([A-Z])", " \\1", s).split()
    string = ""
    for s in s_list:
        string += s + " "
    return string.rstrip(" ").lower()


def make_dataset_ucf(dir, nframes, class_to_idx, frame_stride=1, clip_step=None):
    """
    Load consecutive clips and consecutive frames from `dir`.

    args:
        nframes: num of frames of every video clips
        class_to_idx: for mapping video name to video id
        frame_stride: select frames with a stride.
        clip_step: select clips with a step. if clip_step< nframes,
            there will be overlapped frames among two consecutive clips.

    assert videos are saved in first-level directory:
        dir:
            videoxxx1
                frame1.jpg
                frame2.jpg
            videoxxx2
    """
    if clip_step is None:
        clip_step = nframes
    clips = []
    videos = []
    print(dir)
    for video_name in sorted(os.listdir(dir)):
        if video_name != "_broken_clips":
            video_path = os.path.join(dir, video_name)
            assert os.path.isdir(video_path)
            frames = []
            for i, fname in enumerate(sorted(os.listdir(video_path))):
                assert is_image_file(fname), f"fname={fname},video_path={video_path},dir={dir}"
                img_path = os.path.join(video_path, fname)
                class_name = video_name.split("_")[1].lower()  # v_BoxingSpeedBag_g12_c05 -> boxingspeedbag
                class_caption = split_by_captical(
                    video_name.split("_")[1]
                )  # v_BoxingSpeedBag_g12_c05 -> BoxingSpeedBag -> boxing speed bag
                frame_info = {
                    "img_path": img_path,
                    "class_index": class_to_idx[class_name],
                    "class_name": class_name,
                    "class_caption": class_caption,
                }
                frames.append(frame_info)
            if len(frames) >= nframes:
                videos.append(frames)
            frames = frames[::frame_stride]
            start_indices = list(range(len(frames)))[::clip_step]
            for i in start_indices:
                clip = frames[i : i + nframes]
                if len(clip) == nframes:
                    clips.append(clip)
    return clips, videos


def load_and_transform_frames(frame_list, loader, img_transform=None):
    assert isinstance(frame_list, list)
    clip = []
    labels = []
    for frame in frame_list:
        if isinstance(frame, tuple):
            fpath, label = frame
        elif isinstance(frame, dict):
            fpath = frame["img_path"]
            label = {
                "class_index": frame["class_index"],
                "class_name": frame["class_name"],
                "class_caption": frame["class_caption"],
            }
        labels.append(label)

        oriimg = loader(fpath)
        if img_transform is not None:
            img = img_transform(oriimg)

        img = img.reshape([img.shape[0], 1, img.shape[1], img.shape[2]])
        clip.append(img)
    return clip, labels[0]


class VideoFrameDataset(paddle.io.Dataset):
    def __init__(
        self,
        data_root,
        resolution,
        video_length,
        dataset_name="",
        subset_split="",
        annotation_dir=None,
        spatial_transform="",
        temporal_transform="",
        frame_stride=1,
        clip_step=None,
        tokenizer=None,
    ):
        self.loader = default_loader
        self.video_length = video_length
        self.subset_split = subset_split
        self.temporal_transform = temporal_transform
        self.spatial_transform = spatial_transform
        self.frame_stride = frame_stride
        self.dataset_name = dataset_name
        assert subset_split in ["train", "test", "all", ""]
        assert self.temporal_transform in ["", "rand_clips"]
        if subset_split == "all":
            print(111)
            video_dir = os.path.join(data_root, "train")
        else:
            video_dir = os.path.join(data_root, subset_split)
        if dataset_name == "UCF-101":
            if annotation_dir is None:
                annotation_dir = os.path.join(data_root, "ucfTrainTestlist")
            class_to_idx = class_name_to_idx(annotation_dir)
            assert len(class_to_idx) == 101, f"num of classes = {len(class_to_idx)}, not 101"
        elif dataset_name == "sky":
            classes, class_to_idx = find_classes(video_dir)
        else:
            class_to_idx = None
        if dataset_name == "UCF-101":
            func = make_dataset_ucf
        else:
            func = make_dataset
        self.clips, self.videos = func(
            video_dir,
            video_length,
            class_to_idx,
            frame_stride=frame_stride,
            clip_step=clip_step,
        )
        assert len(self.clips[0]) == video_length, f"Invalid clip length = {len(self.clips[0])}"
        if self.temporal_transform == "rand_clips":
            self.clips = self.videos
        if subset_split == "all":
            video_dir = video_dir.rstrip("train") + "test"
            cs, vs = func(video_dir, video_length, class_to_idx)
            if self.temporal_transform == "rand_clips":
                self.clips += vs
            else:
                self.clips += cs
        print("[VideoFrameDataset] number of videos:", len(self.videos))
        print("[VideoFrameDataset] number of clips", len(self.clips))
        print("[VideoFrameDataset] video_length", self.video_length)
        if len(self.clips) == 0:
            raise RuntimeError(
                f"Found 0 clips in {video_dir}. \nSupported image extensions are: " + ",".join(IMG_EXTENSIONS)
            )
        self.img_transform = paddle.vision.transforms.Compose(
            [
                paddle.vision.transforms.ToTensor(),
                paddle.vision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
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

        if tokenizer:
            self.text_processing = lambda caption: tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="np",
            ).input_ids[0]
        else:
            self.text_processing = None

    def __getitem__(self, index):
        if self.temporal_transform == "rand_clips":
            raw_video = self.clips[index]
            rand_idx = random.randint(0, len(raw_video) - self.video_length)
            clip = raw_video[rand_idx : rand_idx + self.video_length]
        else:
            clip = self.clips[index]
        assert (
            len(clip) == self.video_length
        ), f"current clip_length={len(clip)}, target clip_length={self.video_length}, {clip}"
        frames, labels = load_and_transform_frames(clip, self.loader, self.img_transform)

        assert (
            len(frames) == self.video_length
        ), f"current clip_length={len(frames)}, target clip_length={self.video_length}, {clip}"
        frames = paddle.concat(x=frames, axis=1)
        if self.video_transform is not None:
            if self.spatial_transform == "center_crop_resize":
                temp_frames = rearrange(frames, "c t h w -> (c t) h w")
                temp_frames = self.video_transform_step1(temp_frames)
                frames = rearrange(temp_frames, "(c t) h w -> c t h w", c=frames.shape[0])
                frames = self.video_transform_step2(frames)
            else:
                frames = self.video_transform(frames)

        example = dict()
        example["image"] = frames
        if labels is not None and self.dataset_name == "UCF-101":
            example["caption"] = labels["class_caption"]
            example["class_label"] = labels["class_index"]
            example["class_name"] = labels["class_name"]
        example["frame_stride"] = self.frame_stride

        if self.text_processing:
            tensor_out = {
                "pixel_values": example["image"],
                "input_ids": self.text_processing(example["caption"]),
            }
        else:
            tensor_out = {
                "pixel_values": example["image"],
            }
        return tensor_out

    def __len__(self):
        return len(self.clips)

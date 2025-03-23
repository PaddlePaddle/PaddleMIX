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

import io
import json
import logging
import math
import os
import random
import re
import traceback
from collections import Counter
from copy import deepcopy
from typing import TYPE_CHECKING, Dict

import cv2
import imageio
import numpy as np
import paddle
from decord import VideoReader
from paddle.io import ConcatDataset
from PIL import Image, UnidentifiedImageError

from paddlemix.datasets.internvl_dataset import (
    WeightedConcatDataset,
    build_transform,
    dynamic_preprocess,
    pil_loader,
    preprocess,
    preprocess_internlm,
    preprocess_mpt,
    preprocess_phi3,
)
from paddlemix.models.internvl2.constants import (
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
)
from paddlemix.models.internvl2.conversation import get_conv_template

from .dataset_packed import PackedDataset
from .trainer_utils import LabelSmoother

logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


try:
    from petrel_client.client import Client
except ImportError:
    print("petrel_client is not installed. If you read data locally instead of from ceph, ignore it.")

if TYPE_CHECKING:
    from paddlenlp.transformers import PreTrainedTokenizer


def calculate_ngram_repetition(text, n):
    words = text.split()
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    ngram_counts = Counter(ngrams)
    total_ngrams = len(ngrams)
    repeated_ngrams = sum(1 for count in ngram_counts.values() if count > 1)
    return repeated_ngrams / total_ngrams if total_ngrams > 0 else 0


def check_conversations_repetition(conversations, repeat_threshold=0.4, ngram=10):
    for conversation in conversations:
        if conversation["from"] == "gpt":
            model_answer = conversation["value"]
            repeat_ratio = calculate_ngram_repetition(model_answer, ngram)
            if repeat_ratio > repeat_threshold:
                raise Exception


def get_frame_indices(num_frames, vlen, sample="rand", fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]:
        acc_samples = min(num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                paddle.sort(x=frame_indices), paddle.argsort(x=frame_indices)
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [(x[0] + fix_start) for x in ranges]
        elif sample == "middle":
            frame_indices = [((x[0] + x[1]) // 2) for x in ranges]
        else:
            raise NotImplementedError
        if len(frame_indices) < num_frames:
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[: len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
    else:
        raise ValueError
    return frame_indices


def read_frames_gif(video_path, num_frames, sample="rand", fix_start=None, client=None, min_num_frames=4):
    if "s3://" in video_path:
        video_bytes = client.get(video_path)
        gif = imageio.get_reader(io.BytesIO(video_bytes))
    else:
        gif = imageio.get_reader(video_path)
    vlen = len(gif)
    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    for index, frame in enumerate(gif):
        if index in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB).astype(np.uint8)
            frame = Image.fromarray(frame)
            frames.append(frame)
    return frames


def read_frames_decord(
    video_path, num_frames, sample="rand", fix_start=None, client=None, clip=None, min_num_frames=4
):
    if "s3://" in video_path:
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)
    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start, input_fps=fps)
    if clip:
        frame_indices = [(f + start_index) for f in frame_indices]
    frames = video_reader.get_batch(frame_indices).asnumpy()
    frames = [Image.fromarray(frames[i]) for i in range(tuple(frames.shape)[0])]
    return frames


def extract_frame_number(filename):
    match = re.search("_(\\d+).jpg$", filename)
    return int(match.group(1)) if match else -1


def sort_frames(frame_paths):
    return sorted(frame_paths, key=lambda x: extract_frame_number(os.path.basename(x)))


def read_frames_folder(
    video_path, num_frames, sample="rand", fix_start=None, client=None, clip=None, min_num_frames=4
):
    if "s3://" in video_path:
        image_list = sort_frames(client.list(video_path))
        frames = []
        for image in image_list:
            fp = os.path.join(video_path, image)
            frame = Image.open(io.BytesIO(client.get(fp)))
            frames.append(frame)
    else:
        image_list = sort_frames(list(os.listdir(video_path)))
        frames = []
        for image in image_list:
            fp = os.path.join(video_path, image)
            frame = Image.open(fp).convert("RGB")
            frames.append(frame)
    vlen = len(frames)
    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    if vlen > t_num_frames:
        frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start)
        frames = [frames[i] for i in frame_indices]
    return frames


class TCSLoader(object):
    def __init__(self, conf_path, sc_config_key="sensecore"):
        print(f"[TCSLoader] config_path: {conf_path}")
        print("--> before Client(conf_path)")
        self.client = Client(conf_path)
        self.sc_config_key = sc_config_key
        print("--> after Client(conf_path)")

    def __call__(self, fn, image_type="image", max_num_frames=-1, min_num_frames=8, sample="rand", clip=None):
        if image_type == "image":
            img_value_str = self.client.get(fn)
            img = pil_loader(img_value_str)
            return img
        elif image_type == "video":
            if fn.endswith("/"):
                frames = read_frames_folder(
                    fn, num_frames=max_num_frames, min_num_frames=min_num_frames, client=self.client, sample=sample
                )
            elif fn.endswith(".gif"):
                frames = read_frames_gif(
                    fn, num_frames=max_num_frames, min_num_frames=min_num_frames, client=self.client, sample=sample
                )
            else:
                frames = read_frames_decord(
                    fn,
                    num_frames=max_num_frames,
                    min_num_frames=min_num_frames,
                    client=self.client,
                    sample=sample,
                    clip=clip,
                )
            return frames


class LazySupervisedDataset(paddle.io.Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=448,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        min_num_frame=8,
        max_num_frame=32,
        sampling_method="rand",
        repeat_time=1,
        normalize_type="imagenet",
        use_packed_ds=False,
        data_rank=0,
        data_world_size=1,
        distributed_mode=False,
        force_shuffle=False,
        random_seed=0,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f"[Dataset] num_image_token: {num_image_token}")
        logger.info(f"[Dataset] dynamic_image_size: {dynamic_image_size}")
        logger.info(f"[Dataset] use_thumbnail: {use_thumbnail}")
        logger.info(f"[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}")
        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode
        self.dataset_type = "pair"
        self.max_num_images = 1
        self.max_tokens = tokenizer.model_max_length
        self.force_shuffle = force_shuffle
        self._state_dict = {}
        logger.info("Formatting inputs...Skip in lazy mode")
        assert meta["annotation"].endswith("jsonl"), f"annotation must be jsonl, but got {meta['annotation']}"
        total_ranks = paddle.distributed.get_world_size()
        self.total_ranks = total_ranks
        current_rank = paddle.distributed.get_rank()
        """
        This section of the code is used to read hundreds of millions of data entries.
        By using caching and splitting the data according to rank, it ensures fast reading
        speed and prevents out-of-memory.
        """
        basename = os.path.basename(meta["annotation"]).replace(".jsonl", "")
        data_dir = os.path.join(os.path.dirname(meta["annotation"]), f"{basename}_temp")
        os.makedirs(data_dir, exist_ok=True)
        temp_path = os.path.join(data_dir, f"{basename}_{current_rank}_of_{total_ranks}.jsonl")
        if os.path.exists(temp_path):
            with open(temp_path, "r") as f:
                self.raw_data = f.readlines()
        else:
            with open(meta["annotation"], "r") as f:
                self.raw_data = f.readlines()
            if repeat_time < 1:
                self.raw_data = self.raw_data[: int(len(self.raw_data) * repeat_time)]
            else:
                self.raw_data = self.raw_data * int(repeat_time)
            total_lines = len(self.raw_data)
            logger.info(f"total_ranks: {total_ranks}, current_rank: {current_rank}, total_lines: {total_lines}")
            lines_per_rank = total_lines // total_ranks
            lines_per_rank = max(1, lines_per_rank)
            start_line = lines_per_rank * current_rank
            end_line = start_line + lines_per_rank
            self.raw_data = self.raw_data[start_line:end_line]
            with open(temp_path, "w") as f:
                f.writelines(self.raw_data)
        self.rng = np.random.default_rng(seed=random_seed)
        if self.force_shuffle:
            self.rng.shuffle(self.raw_data)
        self.root = meta["root"]
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type
        assert not group_by_length
        if self.group_by_length:
            self.conv2length = {}
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if "length" in data_item:
                    token_length = data_item["length"]
                else:
                    conversations = "\n".join([temp["value"] for temp in data_item["conversations"]])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors="pt", padding=False, truncation=False
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                            max_dynamic_patch + use_thumbnail
                        )
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

    def __len__(self):
        if not self.use_packed_ds:
            return len(self.raw_data) * self.total_ranks
        else:
            return len(self.raw_data)

    def get_preprocess_function(self):
        if self.template_name == "Hermes-2":
            preprocess_function = preprocess_mpt
        elif self.template_name == "internlm2-chat":
            preprocess_function = preprocess_internlm
        elif self.template_name == "phi3-chat":
            preprocess_function = preprocess_phi3
        elif self.template_name == "internvl2_5":
            preprocess_function = preprocess_internvl2_5
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        if self.tcs_loader is not None and "s3://" in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert("RGB")

    def get_image_path(self, image_path):
        if image_path.startswith("s3://"):
            image_path = self.root + image_path
        else:
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        transform = build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type,
        )
        return transform

    def multi_modal_get_item(self, data_item):
        transform = self.get_transform()
        if "<image>" not in data_item["conversations"][0]["value"]:
            data_item["conversations"][0]["value"] = "<image>\n" + data_item["conversations"][0]["value"]
        image_path = self.get_image_path(data_item["image"])
        image = self.load_image(image_path)
        if self.dynamic_image_size:
            images = dynamic_preprocess(
                image,
                min_num=self.min_dynamic_patch,
                max_num=self.max_dynamic_patch,
                image_size=self.image_size,
                use_thumbnail=self.use_thumbnail,
            )
        else:
            images = [image]
        pixel_values = [transform(image) for image in images]
        pixel_values = paddle.stack(x=pixel_values)
        num_patches = pixel_values.shape[0]
        if not self.dynamic_image_size:
            assert num_patches == 1, f"The number of patches should be 1, but got {num_patches}."
        preprocess_function = self.get_preprocess_function()
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["conversations"])],
            self.tokenizer,
            [self.num_image_token * num_patches],
            group_by_length=self.group_by_length,
            use_packed_ds=self.use_packed_ds,
            ds_name=self.ds_name,
        )
        position_ids = ret["attention_mask"].astype(dtype="int64").cumsum(axis=-1) - 1
        position_ids.masked_fill_(mask=ret["attention_mask"] == 0, value=1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (
            ret["input_ids"][0] == image_end_token_id
        ).sum() == 1, f"image tokens are truncated, this dataset is {self.ds_name}"
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=paddle.to_tensor(data=[1] * num_patches, dtype="int64"),
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        transform = self.get_transform()
        images, num_tiles = [], []
        num_image = len(data_item["image"])
        for image_path in data_item["image"]:
            image_path = self.get_image_path(image_path)
            image = self.load_image(image_path)
            if self.dynamic_image_size:
                image = dynamic_preprocess(
                    image,
                    min_num=self.min_dynamic_patch,
                    max_num=max(1, self.max_dynamic_patch // num_image),
                    image_size=self.image_size,
                    use_thumbnail=self.use_thumbnail,
                )
                images += image
                num_tiles.append(len(image))
            else:
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = paddle.stack(x=pixel_values)
        num_patches = pixel_values.shape[0]
        preprocess_function = self.get_preprocess_function()
        num_image_tokens = [(self.num_image_token * num_tile) for num_tile in num_tiles]
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["conversations"])],
            self.tokenizer,
            num_image_tokens,
            group_by_length=self.group_by_length,
            use_packed_ds=self.use_packed_ds,
            ds_name=self.ds_name,
            num_image=num_image,
        )
        position_ids = ret["attention_mask"].astype(dtype="int64").cumsum(axis=-1) - 1
        position_ids.masked_fill_(mask=ret["attention_mask"] == 0, value=1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (
            ret["input_ids"][0] == image_end_token_id
        ).sum() == num_image, f"image tokens are truncated, this dataset is {self.ds_name}"
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=paddle.to_tensor(data=[1] * num_patches, dtype="int64"),
        )
        return ret

    def video_get_item(self, data_item):
        transform = self.get_transform()
        if "<video>" not in data_item["conversations"][0]["value"]:
            data_item["conversations"][0]["value"] = "<video>\n" + data_item["conversations"][0]["value"]
        video_file = data_item["video"]
        video_path = os.path.join(self.root, video_file)
        image_list = self.tcs_loader(
            video_path,
            image_type="video",
            max_num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method,
            clip=data_item.get("clip", None),
        )
        special_tokens = "\n".join(["Frame-{}: <image>".format(i + 1) for i in range(len(image_list))])
        data_item["conversations"][0]["value"] = data_item["conversations"][0]["value"].replace(
            "<video>\n", special_tokens + "\n"
        )
        pixel_values = [transform(image) for image in image_list]
        pixel_values = paddle.stack(x=pixel_values)
        num_patches = pixel_values.shape[0]
        preprocess_function = self.get_preprocess_function()
        num_image_tokens = [self.num_image_token] * num_patches
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["conversations"])],
            self.tokenizer,
            num_image_tokens,
            group_by_length=self.group_by_length,
            use_packed_ds=self.use_packed_ds,
            ds_name=self.ds_name,
            num_image=num_patches,
        )
        position_ids = ret["attention_mask"].astype(dtype="int64").cumsum(axis=-1) - 1
        position_ids.masked_fill_(mask=ret["attention_mask"] == 0, value=1)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=paddle.to_tensor(data=[1] * num_patches, dtype="int64"),
        )
        return ret

    def pure_text_get_item(self, data_item):
        transform = self.get_transform()
        image = Image.new("RGB", (224, 224), (255, 255, 255))
        images = dynamic_preprocess(
            image,
            min_num=self.min_dynamic_patch,
            max_num=1,
            image_size=self.image_size,
            use_thumbnail=self.use_thumbnail,
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = paddle.stack(x=pixel_values)
        num_patches = pixel_values.shape[0]
        assert num_patches == 1, f"The number of patches should be 1, but got {num_patches}."
        preprocess_function = self.get_preprocess_function()
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["conversations"])],
            self.tokenizer,
            [self.num_image_token * num_patches],
            text_only=True,
            group_by_length=self.group_by_length,
            use_packed_ds=self.use_packed_ds,
            ds_name=self.ds_name,
        )
        position_ids = ret["attention_mask"].astype(dtype="int64").cumsum(axis=-1) - 1
        position_ids.masked_fill_(mask=ret["attention_mask"] == 0, value=1)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=paddle.to_tensor(data=[0] * num_patches, dtype="int64"),
        )
        return ret

    def _enable_worker_distributed(self):
        if self.distributed_mode and not self.worker_distributed and self.worker_id is not None:
            self.worker_distributed = True
            num_worker_per_rank = self.num_workers // self.total_ranks
            self.raw_data = self.raw_data[self.worker_id % num_worker_per_rank :: num_worker_per_rank]
            logger.info(
                f"worker_distributed is enabled, self.num_workers={self.num_workers!r}, len(self.raw_data)={len(self.raw_data)!r}"
            )

    def __getitem__(self, i) -> Dict[str, paddle.Tensor]:
        if i >= len(self.raw_data):
            if self.use_packed_ds:
                raise NotImplementedError
            else:
                i = i % len(self.raw_data)
        while True:
            try:
                data_item = json.loads(self.raw_data[i])
                if "image" in data_item and len(data_item["image"]) != 0:
                    if type(data_item["image"]) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif "video" in data_item and data_item["video"] is not None and data_item["video"] != "":
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                print(e, self.ds_name, flush=True)
                if not isinstance(e, UnidentifiedImageError):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if "image" in data_item:
                    if type(data_item["image"]) == list:
                        images = [(self.root + item) for item in data_item["image"]]
                        print(f"Failed to load image: {images}, the dataset is: {self.ds_name}")
                    else:
                        if data_item["image"].startswith("s3://"):
                            data_path = self.root + data_item["image"]
                        else:
                            data_path = os.path.join(self.root, data_item["image"])
                        print(f"Failed to load image: {data_path}, the dataset is: {self.ds_name}")
                elif "video" in data_item:
                    data_path = os.path.join(self.root, data_item["video"])
                    print(f"Failed to load video: {data_path}, the dataset is: {self.ds_name}")
                i = random.randint(0, len(self.raw_data) - 1)
        return ret

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0
        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]["current_idx"]
            self._state_dict.pop(self.worker_state_key)
        if self.worker_id == 0:
            logger.info(f"[{self.ds_name}] [Worker id {self.worker_id}] begin to iter with start_idx={start_idx!r}")
        for i in range(start_idx, len(self)):
            yield self[i]


def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    normalize_type="imagenet",
):
    datasets = []
    lengths = []
    data_rank = paddle.distributed.get_rank()
    data_world_size = paddle.distributed.get_world_size()
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]["repeat_time"]
        if "max_dynamic_patch" in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]["max_dynamic_patch"]
            logger.info(f"max_dynamic_patch is set to {max_num} according to the meta file")
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style,
            ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]["data_augment"],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length and not data_args.use_packed_ds,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            use_packed_ds=data_args.use_packed_ds,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=data_args.use_packed_ds,
            force_shuffle=data_args.use_packed_ds,
            random_seed=ds_idx,
        )
        logger.info(f"Add dataset: {ds_name} with length: {len(dataset)}")
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))
    if data_args.use_packed_ds:
        total_length = sum(lengths)
        train_dataset = PackedDataset(
            tokenizer=tokenizer,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=[(l / total_length) for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode,
            replacement=data_args.replacement,
            allow_overflow=data_args.allow_overflow,
            allow_deduplicated_ds_name=False,
        )
    elif data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [(l / total_length) for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset

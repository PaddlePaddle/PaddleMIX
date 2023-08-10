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
import io
import logging
import os
import base64
import gzip
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from easydict import EasyDict as edict
from PIL import Image
import random

# import braceexpand
import torch
import paddle
import paddle.vision.datasets as datasets
from paddle.io import DataLoader, IterableDataset, get_worker_info
from paddle.io import Dataset

from .dataset import ImageFolder


def paddle_worker_info(group=None):
    """Return node and worker info for paddle and some distributed environments."""
    rank = 0
    world_size = 1
    worker = 0
    num_workers = 1
    if "WORKER" in os.environ and "NUM_WORKERS" in os.environ:
        worker = int(os.environ["WORKER"])
        num_workers = int(os.environ["NUM_WORKERS"])
    else:
        try:
            worker_info = get_worker_info()
            if worker_info is not None:
                worker = worker_info.id
                num_workers = worker_info.num_workers
        except ModuleNotFoundError:
            pass

    return rank, world_size, worker, num_workers


class LaionDataset(IterableDataset):
    def __init__(self,
                 file_list,
                 get_text_emb='',
                 data_world_rank=0,
                 data_world_size=1,
                 buffer_size=1,
                 shuffle_every_n_samples=1000,
                 total_seen_samples=None):

        with open(file_list, 'r', encoding='utf-8') as f:
            self.file_list = f.read().strip().split('\n')
        self.get_text_emb = get_text_emb
        self.buffer_size = buffer_size
        self.shuffle_every_n_samples = shuffle_every_n_samples
        self.min_size = 5
        self.total_seen_samples = total_seen_samples
        self.data_world_rank = data_world_rank
        self.data_world_size = data_world_size

    def parse_line(self, line, filename):
        try:
            vec = line.strip().split("\t")

            if '0-4.5' in filename:
                img_b64 = vec[7]
            elif '4.5-5' in filename:
                img_b64 = vec[9]
            elif '5-10' in filename:
                img_b64 = vec[12]
            caption_en = vec[2]
            text_embs_b64 = vec[-1]

            image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert(
                'RGB')
            text_embs = torch.load(io.BytesIO(base64.b64decode(text_embs_b64)))

            text_emb = text_embs.get(self.get_text_emb, None)
            if text_emb:
                text_emb = paddle.to_tensor(text_emb, dtype=paddle.float32)
            return dict(
                image=image,
                text=caption_en,
                text_emb=text_emb, )

        except Exception as err:
            print(f'error when parse file {filename} with error {err}')
            return None

    def get_data(self, data):
        w, h = data['image'].size
        if w < self.min_size or h < self.min_size:
            return None
        return data

    def __len__(self):
        return self.total_seen_samples

    def sample(self):
        _, _, worker, num_workers = paddle_worker_info()
        total_num_workers = num_workers * self.data_world_size
        global_worker_id = self.data_world_rank * num_workers + worker

        print('[CHECK ME] LaionDataset', global_worker_id, total_num_workers)
        while True:
            random.shuffle(self.file_list)
            for i in range(len(self.file_list)):
                if i % total_num_workers == global_worker_id:
                    filename = self.file_list[i].strip("\n")

                    with gzip.open(filename,
                                   'rb') if filename.endswith('.gz') else open(
                                       filename, 'rb') as f:
                        retry = 0
                        while True:
                            line = f.readline()

                            if line == b'':
                                break
                            try:
                                try:
                                    line = line.decode(encoding='utf-8')
                                except:
                                    line = line.decode(encoding='gb18030')
                            except:
                                print(f'error on file {filename}')
                                continue
                            data = self.parse_line(line, filename)

                            if data is None:
                                continue
                            else:
                                data = self.get_data(data)
                                if data is None:
                                    continue
                                yield data

    def shuffle(self, iterator):
        buffer_list = []
        for _ in range(self.buffer_size):
            buffer_list.append(next(iterator))
        i = 0
        while True:
            if i % self.shuffle_every_n_samples == 0:
                random.shuffle(buffer_list)
            yield buffer_list.pop()
            buffer_list.append(next(iterator))
            i += 1

    def __iter__(self):
        return self.shuffle(iter(self.sample()))


def get_classification(args, preprocess_fns):
    # support classification
    result = {}
    preprocess_train, preprocess_val = preprocess_fns

    preprocess_fn = preprocess_val
    data_paths = args.classification_eval.split(",")

    for data_path in data_paths:
        data_path = data_path.rstrip('/')
        logging.info(f"adding classification dataset: {data_path}")
        dataset = datasets.ImageFolder(
            f"{data_path}/images", transform=preprocess_fn)

        dataset = ImageFolder(f"{data_path}/images", transform=preprocess_fn)

        dataloader = DataLoader(
            dataset,
            batch_size=args.per_device_eval_batch_size,  # hard code
            num_workers=args.dataloader_num_workers,
            shuffle=False, )

        classname_filename = f"{data_path}/labels.txt"
        template_filename = f"{data_path}/templates.txt"

        result[f"{os.path.basename(data_path)}"] = edict(
            dataloader=dataloader,
            classname_filename=classname_filename,
            template_filename=template_filename)

    return result


def get_data(args, preprocess_fns):
    data = {}

    if args.classification_eval is not None:
        tmp = get_classification(args, preprocess_fns)
        for k, v in tmp.items():
            data[f'eval/classification/{k}'] = v

    return data

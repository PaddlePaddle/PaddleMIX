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

import glob
import os
import random

import numpy as np
import paddle
import paddle.distributed as dist


def worker_init_fn(_):
    worker_info = paddle.io.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id

    local_rank = dist.get_rank()
    # world_size = dist.get_world_size()
    num_workers = worker_info.num_workers
    worker_id = worker_info.id
    worker_global_id = local_rank * num_workers + worker_id

    dataset.rng = np.random.RandomState(worker_global_id)
    return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DatasetFactory(object):
    def __init__(self):
        self.train = None
        self.val = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "val":
            dataset = self.val
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):
        # to [B C H W] and [0, 1]
        v = 0.5 * (v + 1.0)
        v.clamp_(0.0, 1.0)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None


class CFGDataset(paddle.io.Dataset):
    # for classifier free guidance
    def __init__(self, dataset, p_uncond, empty_token):
        self.dataset = dataset
        self.p_uncond = p_uncond
        self.empty_token = empty_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        if random.random() < self.p_uncond:
            y = self.empty_token
        return x, y


def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, "*.npy"))
    files_caption = glob.glob(os.path.join(root, "*_*.npy"))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split("_")
        n_captions[int(k1)] += 1
    return num_data, n_captions


class MSCOCOFeatureDataset(paddle.io.Dataset):
    # the image features are got through sample
    def __init__(self, root):
        self.root = root
        self.num_data, self.n_captions = get_feature_dir_info(root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        z = np.load(os.path.join(self.root, f"{index}.npy"))
        k = random.randint(0, self.n_captions[index] - 1)
        c = np.load(os.path.join(self.root, f"{index}_{k}.npy"))
        return z, c


class MSCOCO256Features(DatasetFactory):
    # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, path="", cfg=True, p_uncond=0.1):
        super().__init__()
        self.path = path
        self.cfg = cfg
        self.p_uncond = p_uncond

        print("Prepare dataset...")
        self.train = MSCOCOFeatureDataset(os.path.join(path, "train"))
        self.val = MSCOCOFeatureDataset(os.path.join(path, "val"))
        print("train samples num: ", len(self.train))
        print("val samples num: ", len(self.val))

        self.empty_context = np.load(os.path.join(path, "empty_context.npy"))

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f"prepare the dataset for classifier free guidance with p_uncond={p_uncond}")
            self.train = CFGDataset(self.train, p_uncond, self.empty_context)  # cfg

        # text embedding extracted by clip
        # for visualization in t2i
        self.prompts, self.contexts = [], []
        for f in sorted(os.listdir(os.path.join(path, "run_vis")), key=lambda x: int(x.split(".")[0])):
            prompt, context = np.load(os.path.join(path, "run_vis", f), allow_pickle=True)
            self.prompts.append(prompt)
            self.contexts.append(context)
        self.contexts = np.array(self.contexts)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f"{self.path}/fid_stats/fid_stats_mscoco256_val.npz"

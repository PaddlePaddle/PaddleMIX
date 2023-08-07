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
import collections
import json
import os

import io
import random
import base64
from functools import partial
import gzip
from PIL import Image
import torch
from paddle.io import get_worker_info

from paddlevlp.utils.env import DATA_HOME
from paddlevlp.utils.log import logger

__all__ = ["LaionCLIP"]

from .dataset import DatasetBuilder


def paddle_worker_info(group=None):
    """Return node and worker info for paddle and some distributed environments."""
    # rank = 0
    # world_size = 1
    # worker = 0
    # num_workers = 1
    # if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    #     rank = int(os.environ["RANK"])
    #     world_size = int(os.environ["WORLD_SIZE"])
    # else:
    #     try:
    #         import paddle.distributed

    #         if paddle.distributed.is_available() and paddle.distributed.is_initialized():
    #             group = group or paddle.distributed.group.WORLD
    #             rank = paddle.distributed.get_rank(group=group)
    #             world_size = paddle.distributed.get_world_size(group=group)
    #     except ModuleNotFoundError:
    #         pass
    if "WORKER" in os.environ and "NUM_WORKERS" in os.environ:
        worker = int(os.environ["WORKER"])
        num_workers = int(os.environ["NUM_WORKERS"])
    else:

        worker_info = get_worker_info()
        if worker_info is not None:
            worker = worker_info.id
            num_workers = worker_info.num_workers
        else:
            print("error worker_info is None")
            exit()
            # import pdb;pdb.set_trace()


    return worker, num_workers


class LaionCLIP(DatasetBuilder):

    URL = "https://bj.bcebos.com/paddlemix/datasets/laionclip.zip"
    META_INFO = collections.namedtuple(
        "META_INFO",
        ("filelist", "package", "filelist_md5", "package_md5"))
    MD5 = ""
    SPLITS = {
        "train": META_INFO(
            os.path.join("laion", "./filelists/laion2b.filelist"),
            os.path.join("laion", ""),
            "7499b4b8c3a12d1b791614e14c495785",
            "", ),
    }

    def _get_data(self, mode, **kwargs):
        logger.info("default dataset root is {}".format(DATA_HOME))
        filelist, package, filelist_md5, package_md5 = self.SPLITS[mode]
        filelist_fullname = os.path.join(DATA_HOME, filelist)
        package_fullname = os.path.join(DATA_HOME, package)

        return filelist_fullname, package_fullname, mode

    def _gen_image_id(self, anno):
        img_ids = {}
        n = 0
        for ann in anno:
            img_id = ann["image_id"]
            if img_id not in img_ids.keys():
                img_ids[img_id] = n
                n += 1
        return img_ids

    def _read(self, filename, *args):
        filelist_fullname, package_root, mode = filename
        self.package_root = package_root
        with open(filelist_fullname, 'r', encoding='utf-8') as f:
            self.file_list = f.read().strip().split('\n')
        self.min_size = 5
        self.get_text_emb = 'open_clip_vit_g_14'

        worker, num_workers = paddle_worker_info()
        rank = self.config['data_world_rank']
        world_size = self.config['data_world_size']
        total_num_workers = num_workers * world_size
        global_worker_id = rank * num_workers + worker

        print('[CHECK ME] LaionDataset', global_worker_id, total_num_workers)
        while True:
            # random.shuffle(self.file_list)
            for i in range(len(self.file_list)):
                if i % total_num_workers == global_worker_id:
                    filename = self.file_list[i].strip("\n")
                    filename = os.path.join(self.package_root, filename)
                    # print(filename)
                    if not os.path.exists(filename):
                        print(filename)

                    with gzip.open(filename, 'rb') if filename.endswith('.gz') else open(filename, 'rb') as f:
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


    def parse_line(self, line, filename):
        try:
        # if True:
            vec = line.strip().split("\t")

            if '0-4.5' in filename:
                img_b64 = vec[7]
            elif '4.5-5' in filename:
                img_b64 = vec[9]
            elif '5-10' in filename:
                img_b64 = vec[12]
            caption_en = vec[2]
            text_embs_b64 = vec[-1]

            image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert('RGB')
            text_embs = torch.load(io.BytesIO(base64.b64decode(text_embs_b64)))
            
            text_emb = text_embs.get(self.get_text_emb, None)
            if text_emb:
                text_emb = text_emb
            return dict(
                image=image,
                text=caption_en,
                text_emb=text_emb,
            )

        except Exception as err:
        # else:
            print(f'error when parse file {filename} with error {err}')
            return None
    
    def get_data(self, data):

        w, h = data['image'].size
        if w < self.min_size or h < self.min_size:
            return None
        image = data['image']
        print("yield text:", data['text'])
        # image = self.preprocess(data['image'])
        # image = image.astype(paddle.float32)
        return data
        
        if data['text_emb'] is not None:
            return image, data['text_emb'], data['text']
        else:
            return image, data['text']

    def shuffle(self, iterator):
        buffer_list = []
        for _ in range(self.buffer_size):
            buffer_list.append(next(iterator))
        i = 0
        while True:
            # if i % self.shuffle_every_n_samples == 0:
            #     random.shuffle(buffer_list)
            yield buffer_list.pop()
            buffer_list.append(next(iterator))
            i += 1


    def __iter__(self):
        return self.shuffle(iter(self.sample()))


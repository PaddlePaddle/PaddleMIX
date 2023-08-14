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
import base64
import gzip
import io
import logging
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import paddle
import paddle.vision.datasets as datasets
from easydict import EasyDict as edict
from paddle.io import DataLoader, Dataset, IterableDataset, get_worker_info
from PIL import Image

from .dataset import ImageFolder


def get_classification(args, preprocess_fns):
    # support classification
    result = {}
    preprocess_train, preprocess_val = preprocess_fns

    preprocess_fn = preprocess_val
    data_paths = args.classification_eval.split(",")

    for data_path in data_paths:
        data_path = data_path.rstrip("/")
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
            template_filename=template_filename, )

    return result


def get_data(args, preprocess_fns):
    data = {}

    if args.classification_eval is not None:
        tmp = get_classification(args, preprocess_fns)
        for k, v in tmp.items():
            data[f"eval/classification/{k}"] = v

    return data

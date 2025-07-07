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

import argparse
import os
import pathlib
import re
import sys

import numpy as np
import paddle
import paddle.vision.transforms as TF
from calculate_psnr import img_psnr
from calculate_ssim import calculate_ssim_function
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fid_clip_score")))
from fid_score import ImagePathDataset


def extract_number(filename):
    filename = os.path.basename(filename)
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else float("inf")


IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of TGATE V2.")
    parser.add_argument(
        "--dataset1",
        type=str,
        default=None,
        required=True,
        help="Path to save the original generated results.",
    )
    parser.add_argument(
        "--dataset2",
        type=str,
        default=None,
        required=True,
        help="Path to save the speed up generated results.",
    )
    parser.add_argument("--resolution", type=int, default=None, help="The resolution to resize.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to use")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers to use for data loading")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    gen_path = pathlib.Path(args.dataset1)
    gen_files = sorted(
        [file for ext in IMAGE_EXTENSIONS for file in gen_path.glob("*.{}".format(ext))], key=extract_number
    )
    # get dataset1 path
    dataset_gen = ImagePathDataset(gen_files, transforms=TF.ToTensor(), resolution=args.resolution)
    dataloader_gen = paddle.io.DataLoader(
        dataset_gen,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    # get dataset2 path
    speedgen_path = pathlib.Path(args.dataset2)
    files = sorted(
        [file for ext in IMAGE_EXTENSIONS for file in speedgen_path.glob("*.{}".format(ext))], key=extract_number
    )
    dataset_speedgen = ImagePathDataset(files, transforms=TF.ToTensor(), resolution=args.resolution)
    dataloader_speedgen = paddle.io.DataLoader(
        dataset_speedgen,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    print(len(dataloader_gen))
    print(len(dataloader_speedgen))
    ssim_value_list = []
    psnr_value_list = []
    # calculate ssim与psnr
    for batch_gen, batch_speedgen in tqdm(
        zip(dataloader_gen, dataloader_speedgen), total=len(dataloader_gen), desc="Calculating SSIM and PSNR"
    ):
        batch_speedgen = batch_speedgen["img"]
        batch_gen = batch_gen["img"]
        batch_speedgen = batch_speedgen.squeeze().numpy()  # 将Tensor转换为numpy数组，并调整通道顺序
        batch_gen = batch_gen.squeeze().numpy()
        ssim_value = calculate_ssim_function(batch_gen, batch_speedgen)
        psnr_value = img_psnr(batch_gen, batch_speedgen)
        ssim_value_list.append(ssim_value)
        psnr_value_list.append(psnr_value)
    mean_ssim = np.mean(ssim_value_list)
    mean_psnr = np.mean(psnr_value_list)
    from pathlib import Path

    path = Path(args.dataset1)
    parent_path = path.parent
    # save the result
    res_txt = os.path.basename(args.dataset2)
    with open(os.path.join(parent_path, f"{res_txt}.txt"), "w") as f:  # ← 注意这里用 "a"
        f.write(f"mean_ssim: {mean_ssim}\n")
        f.write(f"mean_psnr: {mean_psnr}\n")
    print("mean_ssim: ", mean_ssim, "mean_psnr: ", mean_psnr)

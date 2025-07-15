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

# code is heavily based on https://github.com/tianweiy/DMD2

import argparse
import glob
import json
import os
import time

import numpy as np
import paddle
import scipy
import wandb
from edm.edm_guidance import open_url
from edm.edm_network import get_imagenet_edm_config
from edm.networks import EDMPrecond
from safetensors.paddle import load_file
from tqdm import tqdm

from ppdiffusers.accelerate import Accelerator
from ppdiffusers.accelerate.utils import ProjectConfiguration, set_seed


def get_imagenet_config():
    base_config = {
        "img_resolution": 64,
        "img_channels": 3,
        "label_dim": 1000,
        "use_fp16": False,
        "sigma_min": 0,
        "sigma_max": float("inf"),
        "sigma_data": 0.5,
        "model_type": "DhariwalUNet",
    }
    base_config.update(get_imagenet_edm_config())
    return base_config


def create_generator(checkpoint_path, base_model=None):
    if base_model is None:
        base_config = get_imagenet_config()
        generator = EDMPrecond(**base_config)
        del generator.model.map_augment
        generator.model.map_augment = None
    else:
        generator = base_model

    while True:
        try:
            state_dict = load_file(checkpoint_path)
            break
        except Exception as e:
            print(f"fail to load checkpoint {checkpoint_path}", e)
            time.sleep(1)

    # print(generator.set_state_dict(state_dict, strict=True))
    print(generator.set_state_dict(state_dict))

    return generator


def create_evaluator(detector_url):
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048

    from scripts.fid_clip_score.inception import InceptionV3

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[feature_dim]
    detector_net = InceptionV3([block_idx])
    detector_net.eval()

    return detector_net, detector_kwargs, feature_dim


@paddle.no_grad()
def sample(accelerator, current_model, args, model_index):
    timesteps = paddle.ones(args.eval_batch_size, dtype=paddle.int64)
    current_model.eval()
    all_images = []
    all_images_tensor = []

    current_index = 0

    all_labels = paddle.arange(0, args.total_eval_samples * 2, dtype=paddle.int64) % args.label_dim

    set_seed(args.seed + accelerator.process_index)

    while len(all_images_tensor) * args.eval_batch_size * accelerator.num_processes < args.total_eval_samples:
        print("index / total_sample:", current_index, "/", args.total_eval_samples, end="\r")
        noise = paddle.randn([args.eval_batch_size, 3, args.resolution, args.resolution])

        random_labels = all_labels[current_index : current_index + args.eval_batch_size]
        one_hot_labels = paddle.eye(args.label_dim)[random_labels]

        current_index += args.eval_batch_size

        eval_images = current_model(
            noise * args.conditioning_sigma, timesteps * args.conditioning_sigma, one_hot_labels
        )
        eval_images = ((eval_images + 1.0) * 127.5).clip(0, 255).to(paddle.uint8).transpose([0, 2, 3, 1])
        eval_images = eval_images.contiguous()

        gathered_images = accelerator.gather(eval_images)

        all_images.append(gathered_images.cpu().numpy())
        all_images_tensor.append(gathered_images.cpu())

    if accelerator.is_main_process:
        print("all_images len ", len(paddle.concat(all_images_tensor, axis=0)))

    all_images = np.concatenate(all_images, axis=0)[: args.total_eval_samples]
    all_images_tensor = paddle.concat(all_images_tensor, axis=0)[: args.total_eval_samples]

    if accelerator.is_main_process:
        # Uncomment if you need to save the images
        # np.savez(os.path.join(args.folder, f"eval_image_model_{model_index:06d}.npz"), all_images)
        # raise
        grid_size = int(args.test_visual_batch_size ** (1 / 2))
        eval_images_grid = all_images[: grid_size * grid_size].reshape(
            grid_size, grid_size, args.resolution, args.resolution, 3
        )
        eval_images_grid = np.swapaxes(eval_images_grid, 1, 2).reshape(
            grid_size * args.resolution, grid_size * args.resolution, 3
        )

        data_dict = {"generated_image_grid": wandb.Image(eval_images_grid)}

        data_dict["image_mean"] = all_images_tensor.astype("float32").mean().item()
        data_dict["image_std"] = all_images_tensor.astype("float32").std().item()

        wandb.log(data_dict, step=model_index)

    accelerator.wait_for_everyone()
    return all_images_tensor


@paddle.no_grad()
def calculate_inception_stats(
    all_images_tensor, evaluator, accelerator, evaluator_kwargs, feature_dim, max_batch_size
):
    mu = paddle.zeros([feature_dim], dtype=paddle.float64)  # , device=accelerator.device)
    sigma = paddle.ones([feature_dim, feature_dim], dtype=paddle.float64)
    # sigma = paddle.ones([feature_dim, feature_dim], dtype=paddle.float64, device=accelerator.device)
    num_batches = (
        (len(all_images_tensor) - 1) // (max_batch_size * accelerator.num_processes) + 1
    ) * accelerator.num_processes
    all_batches = paddle.arange(len(all_images_tensor)).tensor_split(num_batches)
    rank_batches = all_batches[accelerator.process_index :: accelerator.num_processes]

    for i in tqdm(
        range(num_batches // accelerator.num_processes), unit="batch", disable=not accelerator.is_main_process
    ):
        images = all_images_tensor[rank_batches[i]]
        features = evaluator(images.transpose([0, 3, 1, 2]).to(accelerator.device).astype("float32") / 255.0)[0].to(
            dtype=paddle.float64
        )
        # features = evaluator(images.permute(0, 3, 1, 2).to(accelerator.device), **evaluator_kwargs).to(dtype=paddle.float64)
        features = features.squeeze(-1).squeeze(-1)
        mu += features.sum(0)
        sigma += features.T @ features

    # Calculate grand totals.
    mu = accelerator.reduce(mu)
    sigma = accelerator.reduce(sigma)
    mu /= len(all_images_tensor)
    sigma -= mu.outer(mu) * len(all_images_tensor)
    sigma /= len(all_images_tensor) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()


def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))


@paddle.no_grad()
def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="pass to folder")
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_name", type=str)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--total_eval_samples", type=int, default=50000)
    parser.add_argument("--label_dim", type=int, default=10)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--test_visual_batch_size", type=int, default=100)
    parser.add_argument("--max_batch_size", type=int, default=128)
    parser.add_argument("--ref_path", type=str, help="reference fid statistics")
    parser.add_argument("--detector_url", type=str)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--dataset_name", type=str, default="imagenet")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--conditioning_sigma", type=float, default=80.0)

    args = parser.parse_args()

    folder = args.folder
    evaluated_checkpoints = set()
    overall_stats = {}

    # initialize accelerator
    accelerator_project_config = ProjectConfiguration(logging_dir=args.folder)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        log_with="wandb",
        project_config=accelerator_project_config,
    )
    print(accelerator.state)

    # assert accelerator.num_processes == 1, "currently multi-gpu inference generates images with biased class distribution and leads to much worse FID"

    # load previous stats
    info_path = os.path.join(folder, "stats.json")
    if os.path.isfile(info_path) and not args.no_resume:
        with open(info_path, "r") as f:
            overall_stats = json.load(f)
            evaluated_checkpoints = set(overall_stats.keys())
    if accelerator.is_main_process:
        print(f"folder to evaluate: {folder}")

    # initialize wandb
    if accelerator.is_main_process:
        run = wandb.init(
            config=args,
            dir=args.folder,
            **{"mode": "offline", "entity": args.wandb_entity, "project": args.wandb_project},
        )
        wandb.run.name = args.wandb_name
        print(f"wandb run dir: {run.dir}")

    # initialie evaluator
    evaluator, evaluator_kwargs, feature_dim = create_evaluator(args.detector_url)
    evaluator = accelerator.prepare(evaluator)
    generator = None

    # initialize reference statistics
    with open_url(args.ref_path) as f:
        ref_dict = dict(np.load(f))

    while True:
        new_checkpoints = sorted(glob.glob(os.path.join(folder, "*checkpoint_*")))
        new_checkpoints = set(new_checkpoints) - evaluated_checkpoints
        new_checkpoints = sorted(list(new_checkpoints))

        if len(new_checkpoints) == 0:
            continue

        for checkpoint in new_checkpoints:
            if accelerator.is_main_process:
                print(f"Evaluating {folder} {checkpoint}")
            model_index = int(checkpoint.replace("/", "").split("_")[-1])

            generator = create_generator(os.path.join(checkpoint, "model.safetensors"), base_model=generator)
            generator = generator.to(accelerator.device)

            all_images_tensor = sample(accelerator, generator, args, model_index)

            stats = {}

            pred_mu, pred_sigma = calculate_inception_stats(
                all_images_tensor,
                evaluator,
                accelerator,
                evaluator_kwargs,
                feature_dim,
                args.max_batch_size,
            )

            if accelerator.is_main_process:
                fid = calculate_fid_from_inception_stats(pred_mu, pred_sigma, ref_dict["mu"], ref_dict["sigma"])
                stats["fid"] = fid

                print(f"checkpoint {checkpoint} fid {fid}")
                overall_stats[checkpoint] = stats

            wandb.log(stats, step=model_index)

            paddle.device.cuda.empty_cache()

        evaluated_checkpoints.update(new_checkpoints)

        if accelerator.is_main_process:
            # dump stats to folder
            with open(os.path.join(folder, "stats.json"), "w") as f:
                json.dump(overall_stats, f, indent=2)


if __name__ == "__main__":
    evaluate()

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

import os
import random
import zipfile
from glob import glob

import numpy as np
import paddle
from coco_eval import cleanfid
from coco_eval.cleanfid.utils import (
    EXTENSIONS,
    ResizeArrayDataset,
    ResizeDataset,
    build_resizer,
)

# from PIL import Image
from scipy import linalg
from tqdm import tqdm

"""
Numpy implementation of the Frechet Distance.
The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
Stable version by Danica J. Sutherland.
Params:
    mu1   : Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    mu2   : The sample mean over activations, precalculated on an
            representative data set.
    sigma1: The covariance matrix over activations for generated samples.
    sigma2: The covariance matrix over activations, precalculated on an
            representative data set.
"""


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; " "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


"""
Compute the KID score given the sets of features
"""


def kernel_distance(feats1, feats2, num_subsets=100, max_subset_size=1000):
    n = feats1.shape[1]
    m = min(min(feats1.shape[0], feats2.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = feats2[np.random.choice(feats2.shape[0], m, replace=False)]
        y = feats1[np.random.choice(feats1.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)


"""
Compute the inception features for a batch of images
"""


def get_batch_features(batch, model, device):
    with paddle.no_grad():
        feat = model(batch.to(device))
    return feat[0].squeeze(-1).squeeze(-1).detach().cpu().numpy()


"""
Compute the inception features for a list of files
"""


def get_files_features(
    l_files,
    model=None,
    num_workers=12,
    batch_size=128,
    device=None,
    mode="clean",
    custom_fn_resize=None,
    description="",
    fdir=None,
    verbose=True,
    custom_image_tranform=None,
):
    # wrap the images in a dataloader for parallelizing the resize operation
    dataset = ResizeDataset(l_files, fdir=fdir, mode=mode)
    if custom_image_tranform is not None:
        dataset.custom_image_tranform = custom_image_tranform
    if custom_fn_resize is not None:
        dataset.fn_resize = custom_fn_resize

    dataloader = paddle.io.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
    )

    # collect all inception features
    l_feats = []
    if verbose:
        pbar = tqdm(dataloader, desc=description)
    else:
        pbar = dataloader

    for batch in pbar:
        l_feats.append(get_batch_features(batch / 255.0, model, device))
    np_feats = np.concatenate(l_feats)
    return np_feats


"""
Compute the inception features for a numpy array
"""


def get_array_features(
    l_array,
    model=None,
    num_workers=12,
    batch_size=128,
    device=None,
    mode="clean",
    custom_fn_resize=None,
    description="",
    verbose=True,
    custom_image_tranform=None,
):
    # wrap the images in a dataloader for parallelizing the resize operation
    dataset = ResizeArrayDataset(l_array, mode=mode)
    if custom_image_tranform is not None:
        dataset.custom_image_tranform = custom_image_tranform
    if custom_fn_resize is not None:
        dataset.fn_resize = custom_fn_resize

    dataloader = paddle.io.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    # collect all inception features
    l_feats = []
    if verbose:
        pbar = tqdm(dataloader, desc=description)
    else:
        pbar = dataloader

    for batch in pbar:
        l_feats.append(get_batch_features(batch / 255.0, model, device))
    np_feats = np.concatenate(l_feats)
    return np_feats


"""
Compute the inception features for a folder of image files
"""


def get_folder_features(
    fdir,
    model=None,
    num_workers=12,
    num=None,
    shuffle=False,
    seed=0,
    batch_size=128,
    device=None,
    mode="clean",
    custom_fn_resize=None,
    description="",
    verbose=True,
    custom_image_tranform=None,
):
    # get all relevant files in the dataset
    if ".zip" in fdir:
        files = list(set(zipfile.ZipFile(fdir).namelist()))
        # remove the non-image files inside the zip
        files = [x for x in files if os.path.splitext(x)[1].lower()[1:] in EXTENSIONS]
    else:
        files = sorted(
            [file for ext in EXTENSIONS for file in glob(os.path.join(fdir, f"**/*.{ext}"), recursive=True)]
        )
    if verbose:
        print(f"Found {len(files)} images in the folder {fdir}")
    # use a subset number of files if needed
    if num is not None:
        if shuffle:
            random.seed(seed)
            random.shuffle(files)
        files = files[:num]
    np_feats = get_files_features(
        files,
        model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        custom_fn_resize=custom_fn_resize,
        custom_image_tranform=custom_image_tranform,
        description=description,
        fdir=fdir,
        verbose=verbose,
    )
    return np_feats


"""
Compute the FID score given the inception features stack
"""


def fid_from_feats(feats1, feats2):
    mu1, sig1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sig2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
    return frechet_distance(mu1, sig1, mu2, sig2)


"""
Computes the FID score for a folder of images for a specific dataset
and a specific resolution
"""


"""
Compute the FID stats from a generator model
"""


def get_model_features(
    G,
    model,
    mode="clean",
    z_dim=512,
    num_gen=50_000,
    batch_size=128,
    device=None,
    desc="FID model: ",
    verbose=True,
    return_z=False,
    custom_image_tranform=None,
    custom_fn_resize=None,
):
    if custom_fn_resize is None:
        fn_resize = build_resizer(mode)
    else:
        fn_resize = custom_fn_resize

    # Generate test features
    num_iters = int(np.ceil(num_gen / batch_size))
    l_feats = []
    latents = []
    if verbose:
        pbar = tqdm(range(num_iters), desc=desc)
    else:
        pbar = range(num_iters)
    for idx in pbar:
        with paddle.no_grad():
            z_batch = paddle.randn((batch_size, z_dim))  # .to(device)
            if return_z:
                latents.append(z_batch)
            # generated image is in range [0,255]
            img_batch = G(z_batch)
            # split into individual batches for resizing if needed
            if mode != "legacy_tensorflow":
                l_resized_batch = []
                for idx in range(batch_size):
                    curr_img = img_batch[idx]
                    img_np = curr_img.cpu().numpy().transpose((1, 2, 0))
                    if custom_image_tranform is not None:
                        img_np = custom_image_tranform(img_np)
                    img_resize = fn_resize(img_np)
                    l_resized_batch.append(paddle.to_tensor(img_resize.transpose((2, 0, 1))).unsqueeze(0))
                resized_batch = paddle.concat(l_resized_batch, axis=0)
            else:
                resized_batch = img_batch
            feat = get_batch_features(resized_batch, model, device)
        l_feats.append(feat)
    np_feats = np.concatenate(l_feats)[:num_gen]
    if return_z:
        latents = paddle.concat(latents, 0)
        return np_feats, latents
    return np_feats


"""
Computes the FID score between the two given folders
"""


def compare_folders(
    fdir1,
    fdir2,
    feat_model,
    mode,
    num_workers=0,
    batch_size=8,
    device=None,
    verbose=True,
    custom_image_tranform=None,
    custom_fn_resize=None,
):
    # get all inception features for the first folder
    fbname1 = os.path.basename(fdir1)
    np_feats1 = get_folder_features(
        fdir1,
        feat_model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        description=f"FID {fbname1} : ",
        verbose=verbose,
        custom_image_tranform=custom_image_tranform,
        custom_fn_resize=custom_fn_resize,
    )
    mu1 = np.mean(np_feats1, axis=0)
    sigma1 = np.cov(np_feats1, rowvar=False)
    # get all inception features for the second folder
    fbname2 = os.path.basename(fdir2)
    np_feats2 = get_folder_features(
        fdir2,
        feat_model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        description=f"FID {fbname2} : ",
        verbose=verbose,
        custom_image_tranform=custom_image_tranform,
        custom_fn_resize=custom_fn_resize,
    )
    mu2 = np.mean(np_feats2, axis=0)
    sigma2 = np.cov(np_feats2, rowvar=False)
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


"""
Test if a custom statistic exists
"""


def test_stats_exists(name, mode, model_name="inception_v3", metric="FID"):
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    split, res = "custom", "na"
    if model_name == "inception_v3":
        model_modifier = ""
    else:
        model_modifier = "_" + model_name
    if metric == "FID":
        fname = f"{name}_{mode}{model_modifier}_{split}_{res}.npz"
    elif metric == "KID":
        fname = f"{name}_{mode}{model_modifier}_{split}_{res}_kid.npz"
    fpath = os.path.join(stats_folder, fname)
    return os.path.exists(fpath)


"""
Remove the custom FID features from the stats folder
"""


def remove_custom_stats(name, mode="clean", model_name="inception_v3"):
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    # remove the FID stats
    split, res = "custom", "na"
    if model_name == "inception_v3":
        model_modifier = ""
    else:
        model_modifier = "_" + model_name
    outf = os.path.join(stats_folder, f"{name}_{mode}{model_modifier}_{split}_{res}.npz".lower())
    if not os.path.exists(outf):
        msg = f"The stats file {name} does not exist."
        raise Exception(msg)
    os.remove(outf)
    # remove the KID stats
    outf = os.path.join(stats_folder, f"{name}_{mode}{model_modifier}_{split}_{res}_kid.npz")
    if not os.path.exists(outf):
        msg = f"The stats file {name} does not exist."
        raise Exception(msg)
    os.remove(outf)


"""
custom_image_tranform:
    function that takes an np_array image as input [0,255] and
    applies a custom transform such as cropping
"""


def compute_fid(
    fdir1=None,
    fdir2=None,
    gen=None,
    mode="clean",
    model_name="inception_v3",
    num_workers=8,
    batch_size=32,
    device=None,
    dataset_name="FFHQ",
    dataset_res=1024,
    dataset_split="train",
    num_gen=50_000,
    z_dim=512,
    custom_feat_extractor=None,
    verbose=True,
    custom_image_tranform=None,
    custom_fn_resize=None,
    use_dataparallel=True,
    pred_arr=None,
):
    # build the feature extractor based on the mode and the model to be used
    if custom_feat_extractor is None and model_name == "inception_v3":
        # feat_model = build_feature_extractor(mode, device, use_dataparallel=use_dataparallel)
        from inception import InceptionV3

        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        feat_model = InceptionV3([block_idx])
        feat_model.eval()
    elif custom_feat_extractor is None and model_name == "clip_vit_b_32":
        from cleanfid.clip_features import CLIP_fx, img_preprocess_clip

        clip_fx = CLIP_fx("ViT-B/32", device=device)
        feat_model = clip_fx
        custom_fn_resize = img_preprocess_clip
    else:
        feat_model = custom_feat_extractor

    # if both dirs are specified, compute FID between folders
    if fdir1 is not None and fdir2 is not None:
        if verbose:
            print("compute FID between two folders")
        score = compare_folders(
            fdir1,
            fdir2,
            feat_model,
            mode=mode,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            custom_image_tranform=custom_image_tranform,
            custom_fn_resize=custom_fn_resize,
            verbose=verbose,
        )
        return score

    # compute fid of a folder
    elif fdir1 is not None and fdir2 is None:
        if verbose:
            print(f"compute FID of a folder with {dataset_name} statistics")
        raise NotImplementedError

    # compute fid for a generator, using images in fdir2
    elif gen is not None and fdir2 is not None:
        if verbose:
            print("compute FID of a model, using references in fdir2")
        # get all inception features for the second folder
        fbname2 = os.path.basename(fdir2)
        np_feats2 = get_folder_features(
            fdir2,
            feat_model,
            num_workers=num_workers,
            batch_size=batch_size,
            device=device,
            mode=mode,
            description=f"FID {fbname2} : ",
            verbose=verbose,
            custom_fn_resize=custom_fn_resize,
            custom_image_tranform=custom_image_tranform,
        )
        mu2 = np.mean(np_feats2, axis=0)
        sigma2 = np.cov(np_feats2, rowvar=False)
        # Generate test features
        np_feats = get_model_features(
            gen,
            feat_model,
            mode=mode,
            z_dim=z_dim,
            num_gen=num_gen,
            custom_fn_resize=custom_fn_resize,
            custom_image_tranform=custom_image_tranform,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )

        mu = np.mean(np_feats, axis=0)
        sigma = np.cov(np_feats, rowvar=False)
        fid = frechet_distance(mu, sigma, mu2, sigma2)
        return fid

    # compute fid for a generator, using reference statistics
    elif gen is not None:
        if verbose:
            print(f"compute FID of a model with {dataset_name}-{dataset_res} statistics")
        raise NotImplementedError

    elif pred_arr is not None:
        if verbose:
            print("compute FID of a model, using references in fdir2")
        # get all inception features for the second folder
        fbname2 = os.path.basename(fdir2)
        np_feats2 = get_folder_features(
            fdir2,
            feat_model,
            num_workers=num_workers,
            batch_size=batch_size,
            device=device,
            mode=mode,
            description=f"FID {fbname2} : ",
            verbose=verbose,
            custom_fn_resize=custom_fn_resize,
            custom_image_tranform=custom_image_tranform,
        )
        mu2 = np.mean(np_feats2, axis=0)
        sigma2 = np.cov(np_feats2, rowvar=False)

        # compute fid statistcs using the numpy array
        np_feats = get_array_features(
            pred_arr,
            model=feat_model,
            num_workers=num_workers,
            batch_size=batch_size,
            device=device,
            mode=mode,
            custom_fn_resize=custom_fn_resize,
            custom_image_tranform=custom_image_tranform,
        )
        mu = np.mean(np_feats, axis=0)
        sigma = np.cov(np_feats, rowvar=False)
        fid = frechet_distance(mu, sigma, mu2, sigma2)
        return fid
        # return fid, np_feats, np_feats2
    else:
        raise ValueError("invalid combination of directories and models entered")

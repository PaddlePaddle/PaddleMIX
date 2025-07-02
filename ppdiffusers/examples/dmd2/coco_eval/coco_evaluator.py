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

# Part of this code is modified from GigaGAN: https://github.com/mingukkang/GigaGAN
# The MIT License (MIT)

import numpy as np
import paddle
import paddle.vision.transforms as transforms
from paddle.io import Dataset
from PIL import Image


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """

    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


@paddle.no_grad()
def compute_fid(fake_arr, gt_dir, device, resize_size=None, feature_extractor="inception", patch_fid=False):
    from coco_eval.cleanfid import fid

    center_crop_trsf = CenterCropLongEdge()

    def resize_and_center_crop(image_np):
        image_pil = Image.fromarray(image_np)
        if patch_fid:
            # directly crop to the 299 x 299 patch expected by the inception network
            if image_pil.size[0] >= 299 and image_pil.size[1] >= 299:
                image_pil = transforms.functional.center_crop(image_pil, 299)
            else:
                raise ValueError("Image is too small to crop to 299 x 299")
        else:
            image_pil = center_crop_trsf(image_pil)

            if resize_size is not None:
                image_pil = image_pil.resize((resize_size, resize_size), Image.LANCZOS)
        return np.array(image_pil)

    if feature_extractor == "inception":
        model_name = "inception_v3"
    elif feature_extractor == "clip":
        model_name = "clip_vit_b_32"
    else:
        raise ValueError("Unrecognized feature extractor [%s]" % feature_extractor)

    fid = fid.compute_fid(
        None,
        gt_dir,
        model_name=model_name,
        custom_image_tranform=resize_and_center_crop,
        use_dataparallel=False,
        device=device,
        pred_arr=fake_arr,
    )
    return fid


def evaluate_model(args, device, all_images, patch_fid=False):
    fid = compute_fid(
        fake_arr=all_images,
        gt_dir=args.ref_dir,
        device=device,
        resize_size=args.eval_res,
        feature_extractor="inception",
        patch_fid=patch_fid,
    )

    return fid


def tensor2pil(image: paddle.Tensor):
    """output image : tensor to PIL"""
    if isinstance(image, list) or image.ndim == 4:
        return [tensor2pil(im) for im in image]

    assert image.ndim == 3
    output_image = Image.fromarray(
        ((image + 1.0) * 127.5).clamp(0.0, 255.0).to(paddle.uint8).permute(1, 2, 0).detach().cpu().numpy()
    )
    return output_image


class CLIPScoreDataset(Dataset):
    def __init__(self, images, captions, transform, preprocessor) -> None:
        super().__init__()
        self.images = images
        self.captions = captions
        self.transform = transform
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image_pil = self.transform(image)
        image_pil = self.preprocessor(image_pil)
        caption = self.captions[index]
        return image_pil, caption


@paddle.no_grad()
def compute_diversity_score(lpips_loss_func, images, device):
    # resize all image to 512 and convert to tensor
    images = [Image.fromarray(image) for image in images]
    images = [image.resize((512, 512), Image.LANCZOS) for image in images]
    images = np.stack([np.array(image) for image in images], axis=0)
    images = paddle.tensor(images).to(device).float() / 255.0
    images = images.permute(0, 3, 1, 2)

    num_images = images.shape[0]
    loss_list = []

    for i in range(num_images):
        for j in range(i + 1, num_images):
            image1 = images[i].unsqueeze(0)
            image2 = images[j].unsqueeze(0)
            loss = lpips_loss_func(image1, image2)

            loss_list.append(loss.item())
    return np.mean(loss_list)

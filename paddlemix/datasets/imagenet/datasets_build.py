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
import math
import paddle
from paddle.vision import transforms
from paddle.vision.transforms import functional as F
from .transforms import RandomResizedCropAndInterpolationWithTwoResolution
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
from .transform_factory import create_transform
from .masking_generator import MaskingGenerator
from .dataset_folder import ImageFolder


def map2pixel4peco(x):
    return x * 255


class DataAugmentationForEVA(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = (
            0.48145466, 0.4578275, 0.40821073
        ) if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = (0.26862954, 0.26130258, 0.27577711
               ) if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        self.common_transform = [
            transforms.RandomHorizontalFlip(0.5),
            RandomResizedCropAndInterpolationWithTwoResolution(
                size=args.input_size,
                second_size=args.second_input_size,
                scale=[0.2, 1.0], #args.crop_scale,
                ratio=[3. / 4., 4. / 3.], #args.crop_ratio,
                interpolation=args.train_interpolation,
                second_interpolation=args.second_interpolation, ),
        ]

        if args.color_jitter > 0:
            self.common_transform = \
                [transforms.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter)] + \
                self.common_transform

        self.common_transform = transforms.Compose(self.common_transform)

        self.patch_transform = [
            transforms.ToTensor(), transforms.Normalize(
                mean=mean, std=std)
        ]
        self.patch_transform = transforms.Compose(self.patch_transform)

        self.visual_token_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073)
                if 'clip' in args.teacher_type else IMAGENET_INCEPTION_MEAN,
                std=(0.26862954, 0.26130258, 0.27577711)
                if 'clip' in args.teacher_type else IMAGENET_INCEPTION_STD, ),
        ])

        self.masked_position_generator = MaskingGenerator(
            args.window_size,
            num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block, )

    def __call__(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)
        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
            self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForEVA,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(
            self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(
            self.masked_position_generator)
        repr += ")"
        return repr


def build_eva_pretraining_dataset(args):
    transform = DataAugmentationForEVA(args)
    print("Data Aug = %s" % str(transform))
    # args.data_path = os.path.join(args.data_path, 'train')
    dataset = ImageFolder(args.data_path, transform=transform)
    return dataset


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)  ###

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = paddle.vision.datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_val_dataset_for_pt(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.val_data_set == 'IMNET':
        root = os.path.join(args.val_data_path, 'train' if is_train else 'val')
        dataset = paddle.vision.datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.val_data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()

    return dataset, nb_classes


class RandomResizedCrop(paddle.vision.transforms.RandomResizedCrop):
    """
    RandomResizedCrop for matching TF/TPU implementation: no for-loop is used.
    This may lead to results different with torchvision's version.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206
    """

    @staticmethod
    def get_params(img, scale, ratio):
        width, height = F.get_image_size(img)
        area = height * width

        target_area = area * paddle.empty(shape=[1]).uniform_(
            min=scale[0], max=scale[1]).item()
        log_ratio = paddle.log(x=paddle.to_tensor(data=ratio))
        aspect_ratio = paddle.exp(
            paddle.empty(shape=[1]).uniform_(
                min=log_ratio[0], max=log_ratio[1]).astype('float32')).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, width)
        h = min(h, height)

        i = paddle.randint(low=0, high=height - h + 1, shape=(1, )).item()
        j = paddle.randint(low=0, high=width - w + 1, shape=(1, )).item()

        return i, j, h, w


def build_transform(is_train, args):
    print('build_transform  ', args)
    
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = (0.48145466, 0.4578275, 0.40821073
            ) if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = (0.26862954, 0.26130258, 0.27577711
           ) if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        if args.linear_probe:
            return transforms.Compose(
                [
                    RandomResizedCrop(
                        args.input_size, interpolation='bicubic'),
                    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                    transforms.Normalize(
                        mean=mean, std=std)
                ], )
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            no_aug=args.no_aug,  # false
            input_size=args.input_size,
            is_training=True,  # 
            color_jitter=args.color_jitter,
            auto_augment=args.aa,  # 'rand-m9-mstd0.5-inc1'
            interpolation=args.train_interpolation, # 'bicubic'
            re_prob=args.reprob,  # 0
            re_mode=args.remode, # 'pixel'
            re_count=args.recount, # 1
            use_prefetcher=False,
            mean=mean,
            std=std,
            scale=[args.scale_low, 1.0])
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)

        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(
                size, interpolation='bicubic'
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

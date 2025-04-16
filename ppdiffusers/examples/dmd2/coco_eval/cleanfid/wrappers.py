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

import numpy as np
import torch
from cleanfid.features import build_feature_extractor
from cleanfid.fid import fid_from_feats, get_batch_features
from cleanfid.resize import build_resizer
from PIL import Image

"""
A helper class that allowing adding the images one batch at a time.
"""


class CleanFID:
    def __init__(self, mode="clean", model_name="inception_v3", device="cuda"):
        self.real_features = []
        self.gen_features = []
        self.mode = mode
        self.device = device
        if model_name == "inception_v3":
            self.feat_model = build_feature_extractor(mode, device)
            self.fn_resize = build_resizer(mode)
        elif model_name == "clip_vit_b_32":
            from cleanfid.clip_features import CLIP_fx, img_preprocess_clip

            clip_fx = CLIP_fx("ViT-B/32")
            self.feat_model = clip_fx
            self.fn_resize = img_preprocess_clip

    """
    Funtion that takes an image (PIL.Image or np.array or torch.tensor)
    and returns the corresponding feature embedding vector.
    The image x is expected to be in range [0, 255]
    """

    def compute_features(self, x):
        # if x is a PIL Image
        if isinstance(x, Image.Image):
            x_np = np.array(x)
            x_np_resized = self.fn_resize(x_np)
            x_t = torch.tensor(x_np_resized.transpose((2, 0, 1))).unsqueeze(0)
            x_feat = get_batch_features(x_t, self.feat_model, self.device)
        elif isinstance(x, np.ndarray):
            x_np_resized = self.fn_resize(x)
            x_t = torch.tensor(x_np_resized.transpose((2, 0, 1))).unsqueeze(0).to(self.device)
            # normalization happens inside the self.feat_model, expected image range here is [0,255]
            x_feat = get_batch_features(x_t, self.feat_model, self.device)
        elif isinstance(x, torch.Tensor):
            # pdb.set_trace()
            # add the batch dimension if x is passed in as C,H,W
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            b, c, h, w = x.shape
            # convert back to np array and resize
            l_x_np_resized = []
            for _ in range(b):
                x_np = x[_].cpu().numpy().transpose((1, 2, 0))
                l_x_np_resized.append(
                    self.fn_resize(x_np)[
                        None,
                    ]
                )
            x_np_resized = np.concatenate(l_x_np_resized)
            x_t = torch.tensor(x_np_resized.transpose((0, 3, 1, 2))).to(self.device)
            # normalization happens inside the self.feat_model, expected image range here is [0,255]
            x_feat = get_batch_features(x_t, self.feat_model, self.device)
        else:
            raise ValueError("image type could not be inferred")
        return x_feat

    """
    Extract the faetures from x and add to the list of reference real images
    """

    def add_real_images(self, x):
        x_feat = self.compute_features(x)
        self.real_features.append(x_feat)

    """
    Extract the faetures from x and add to the list of generated images
    """

    def add_gen_images(self, x):
        x_feat = self.compute_features(x)
        self.gen_features.append(x_feat)

    """
    Compute FID between the real and generated images added so far
    """

    def calculate_fid(self, verbose=True):
        feats1 = np.concatenate(self.real_features)
        feats2 = np.concatenate(self.gen_features)
        if verbose:
            print(f"# real images = {feats1.shape[0]}")
            print(f"# generated images = {feats2.shape[0]}")
        return fid_from_feats(feats1, feats2)

    """
    Remove the real image features added so far
    """

    def reset_real_features(self):
        self.real_features = []

    """
    Remove the generated image features added so far
    """

    def reset_gen_features(self):
        self.gen_features = []

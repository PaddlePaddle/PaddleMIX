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

import paddle
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

from paddlemix.models.blip2.eva_vit import interpolate_pos_embed
from paddlemix.utils.downloader import get_weights_path_from_url, is_url
from paddlemix.utils.log import logger

LLM_LIST = {
    "facebook/opt-2.7b":
    "https://bj.bcebos.com/paddlenlp/models/community/facebook/opt-2.7b/model_state.pdparams",
    "t5-small":
    "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-small/model_state.pdparams",
    "t5-base":
    "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-base/model_state.pdparams",
    "t5-large":
    "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-large/model_state.pdparams",
    "t5-3b":
    "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-3b/model_state.pdparams",
    "t5-11b":
    "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-11b/model_state.pdparams",
    "t5-v1_1-base":
    "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-v1_1-base/model_state.pdparams",
    "t5-v1_1-large":
    "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-v1_1-large/model_state.pdparams",
    "facebook/llama-7b":
    "https://bj.bcebos.com/paddlenlp/models/community/facebook/llama-7b/model_state.pdparams",
    "facebook/llama-13b":
    "https://bj.bcebos.com/paddlenlp/models/community/facebook/llama-13b/model_state.pdparams",
    "facebook/llama-30b":
    "https://bj.bcebos.com/paddlenlp/models/community/facebook/llama-30b/model_state.pdparams",
    "facebook/llama-65b":
    "https://bj.bcebos.com/paddlenlp/models/community/facebook/llama-65b/model_state.pdparams",
}


class BlipCollator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.

    Args:
        processor (`paddlemix.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor, mode="train"):
        self.processor = processor
        self.mode = mode

    def __call__(self, data_list):
        images = [sample["image"] for sample in data_list]
        if "text_input" not in data_list[0].keys():
            text = None
        else:
            text = [sample["text_input"] for sample in data_list]
        image_id = [sample["image_id"] for sample in data_list]
        batch = self.processor(
            images=images,
            text=text,
            max_length=32,
            return_tensors="pd",
            return_attention_mask=True,
            mode=self.mode, )
        batch.update({"image_id": image_id})
        return batch


def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {
        "val":
        "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
        "test":
        "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
    }
    filenames = {
        "val": "coco_karpathy_val_gt.json",
        "test": "coco_karpathy_test_gt.json",
    }

    # download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames["test"])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval


def load_model(args,
               model,
               optimizer=None,
               ckpt_dir="",
               load_language_model=True):
    """
    load the saved checkpoint file and update the state dicts of model and optimizer.
    """
    if ckpt_dir is None:
        return

    if not os.path.exists(ckpt_dir):
        ValueError("Cannot find pretrained model path: {}".format(ckpt_dir))

    if os.path.isfile(ckpt_dir):
        path = ckpt_dir
    elif is_url(ckpt_dir):
        path = get_weights_path_from_url(ckpt_dir)
    else:
        assert os.path.exists(ckpt_dir), f"{ckpt_dir} not exist"

    ckpt_dir = path
    if ckpt_dir and os.path.isfile(ckpt_dir):
        # breakpoint()
        print("Try to load a whole checkpoint from %s " % ckpt_dir)
        embedding_list = ["word_embeddings"]
        collinear_list = [
            "fc1",
            "fc2",
            "qkv",
            "proj",
            "query",
            "key",
            "value",
            "qkv_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "linear1",
            "linear2",
            "project_in",
            "project_out",
        ]
        rowlinear_list = ["out_proj"]
        all_list = collinear_list + rowlinear_list + embedding_list
        skip_list = [
            "visual_encoder.patch_embed.proj.weight",
            "visual_encoder.patch_embed.proj.bias",
        ]

        col_list = []
        row_list = []
        emb_list = []

        mp_rank = args.mp_rank
        mp_size = args.tensor_parallel_degree

        def renamebias(model_dict, whole_key):
            if "q_bias" in whole_key:
                key = whole_key.replace("q_bias", "q_proj.bias")
            elif "v_bias" in whole_key:
                key = whole_key.replace("v_bias", "v_proj.bias")
            model_dict[key] = model_dict[whole_key]
            del model_dict[whole_key]
            return model_dict

        def col_split_modeldict(model_dict):
            if len(model_dict.shape) == 2:
                subbatch = model_dict.shape[1] // mp_size
                return model_dict[:, mp_rank * subbatch:(mp_rank + 1) *
                                  subbatch]
            elif len(model_dict.shape) == 1:
                subbatch = model_dict.shape[0] // mp_size
                return model_dict[mp_rank * subbatch:(mp_rank + 1) * subbatch]

        def row_split_modeldict(model_dict):
            if len(model_dict.shape) == 2:
                subbatch = model_dict.shape[0] // mp_size
                return model_dict[mp_rank * subbatch:(mp_rank + 1) * subbatch]
            else:
                return model_dict

        def emb_split_modeldict(model_dict):
            subbatch = model_dict.shape[0] // mp_size
            return model_dict[mp_rank * subbatch:(mp_rank + 1) * subbatch]

        model_dict = paddle.load(ckpt_dir)
        for whole_key in model_dict.keys():
            if "." not in whole_key:
                continue

            key = whole_key.split(".")[-2]
            if whole_key in skip_list:
                continue
            if key in all_list:
                if key in collinear_list:
                    col_list.append((key, model_dict[whole_key].shape))
                    model_dict[whole_key] = col_split_modeldict(model_dict[
                        whole_key])
                elif key in rowlinear_list:
                    row_list.append((key, model_dict[whole_key].shape))
                    model_dict[whole_key] = row_split_modeldict(model_dict[
                        whole_key])
                else:
                    emb_list.append((key, model_dict[whole_key].shape))
                    model_dict[whole_key] = emb_split_modeldict(model_dict[
                        whole_key])

        param_state_dict = model_dict
        import numpy as np

        model_dict = model.state_dict()
        model_weight = {}
        incorrect_keys = 0
        for key, value in model_dict.items():
            if key in param_state_dict.keys():

                if isinstance(param_state_dict[key], np.ndarray):
                    param_state_dict[key] = paddle.to_tensor(param_state_dict[
                        key])
                if value.dtype == param_state_dict[key].dtype:
                    model_weight[key] = param_state_dict[key]
                else:
                    model_weight[key] = param_state_dict[key].astype(
                        value.dtype)
                if value.shape != param_state_dict[key].shape:
                    logger.info("Unmatched key: {}".format(key))
                    print(value.shape, param_state_dict[key].shape, key)

            else:
                if load_language_model is False and "language_model" in key:
                    continue
                logger.info("Unmatched key: {}".format(key))
                incorrect_keys += 1
        interpolate_pos_embed(model, model_weight)
        model.set_state_dict(model_weight)

        del model_dict
    else:
        print("`load` requires a valid value of `ckpt_dir`.")
        raise TypeError("`load` requires a valid value of `ckpt_dir`.")

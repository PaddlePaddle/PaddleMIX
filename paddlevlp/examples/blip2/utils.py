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
import os.path as osp
import tarfile
import shutil
import zipfile
import requests
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from paddlevlp.utils.downloader import is_url, _decompress, _md5check, WEIGHTS_HOME, DOWNLOAD_RETRY_LIMIT
from paddlevlp.models.blip2.eva_vit import interpolate_pos_embed
import paddle
from paddlevlp.utils.downloader import is_url
from paddlevlp.utils.log import logger
from tqdm.auto import tqdm
LLM_LIST = {
    "opt-2.7b":
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
}


class BlipCollator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.

    Args:
        processor (`paddlevlp.processors.ProcessorMixin`):
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
        batch.update({'image_id': image_id})
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

    #download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames['test'])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval


def load_pretrained_model(model, pretrained_model_path):
    if pretrained_model_path is None:
        return

    if not os.path.exists(pretrained_model_path):
        ValueError("Cannot find pretrained model path: {}".format(
            pretrained_model_path))

    if os.path.isfile(pretrained_model_path):
        path = pretrained_model_path
    elif is_url(pretrained_model_path):
        path = get_weights_path_from_url(pretrained_model_path)
    else:
        assert os.path.exists(
            pretrained_model_path), f"{pretrained_model_path} not exist"
    state_dict = paddle.load(path, return_numpy=True)
    interpolate_pos_embed(model, state_dict)
    model.set_state_dict(state_dict)


def _map_path(url, root_dir):
    # parse path after download under root_dir
    fname = osp.join(url.split("/")[-2], url.split("/")[-1])
    fpath = fname
    return osp.join(root_dir, fpath)


def get_weights_path_from_url(url, md5sum=None):
    """Get weights path from WEIGHT_HOME, if not exists,
    download it from url.
    Args:
        url (str): download url
        md5sum (str): md5 sum of download package

    Returns:
        str: a local path to save downloaded weights.
    Examples:
        .. code-block:: python
            from paddle.utils.download import get_weights_path_from_url
            resnet18_pretrained_weight_url = 'https://paddle-hapi.bj.bcebos.com/models/resnet18.pdparams'
            local_weight_path = get_weights_path_from_url(resnet18_pretrained_weight_url)
    """
    path = get_path_from_url(url, WEIGHTS_HOME, md5sum)
    return path


def get_path_from_url(url, root_dir, md5sum=None, check_exist=True):
    """Download from given url to root_dir.
    if file or directory specified by url is exists under
    root_dir, return the path directly, otherwise download
    from url and decompress it, return the path.
    Args:
        url (str): download url
        root_dir (str): root dir for downloading, it should be
                        WEIGHTS_HOME or DATASET_HOME
        md5sum (str): md5 sum of download package

    Returns:
        str: a local path to save downloaded models & weights & datasets.
    """

    assert is_url(url), "downloading from {} not a url".format(url)
    # parse path after download to decompress under root_dir
    fullpath = _map_path(url, root_dir)

    if osp.exists(fullpath) and check_exist and _md5check(fullpath, md5sum):
        logger.info("Found {}".format(fullpath))
    else:
        fullpath = _download(url, root_dir, md5sum)

    if tarfile.is_tarfile(fullpath) or zipfile.is_zipfile(fullpath):
        fullpath = _decompress(fullpath)

    # model tokenizer config, [file-lock]
    return fullpath


def _download(url, path, md5sum=None):
    """
    Download from url, save to path.
    url (str): download url
    path (str): download to given path
    """
    path = osp.join(path, url.split("/")[-2])
    os.makedirs(path, exist_ok=True)
    fname = osp.join(url.split("/")[-1])
    fullname = osp.join(path, fname)
    retry_cnt = 0

    while not (osp.exists(fullname) and _md5check(fullname, md5sum)):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RuntimeError("Download from {} failed. "
                               "Retry limit reached".format(url))

        logger.info("Downloading {} from {}".format(fname, url))

        req = requests.get(url, stream=True)
        if req.status_code != 200:
            raise RuntimeError("Downloading from {} failed with code "
                               "{}!".format(url, req.status_code))

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get("content-length")
        with open(tmp_fullname, "wb") as f:
            if total_size:
                with tqdm(
                        total=int(total_size),
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024) as pbar:
                    for chunk in req.iter_content(chunk_size=1024):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)

    return fullname

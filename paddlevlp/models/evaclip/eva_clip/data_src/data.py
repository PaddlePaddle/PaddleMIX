import io
import ast
from itertools import islice
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing import Value
import tarfile
import zipfile
import glob
import base64
from functools import partial
import gzip
from matplotlib.pyplot import get
from easydict import EasyDict as edict
from .imagefolder import ImageFolder

# import braceexpand
import torch
import numpy as np
import pandas as pd
import paddle
import paddle.vision.datasets as datasets
import webdataset as wds
from PIL import Image, TarIO, ImageFile
# from paddle.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
# from paddle.utils.data.distributed import DistributedSampler
from paddle.io import DistributedBatchSampler as DistributedSampler
from paddle.io import DataLoader, Dataset, IterableDataset, get_worker_info
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
#from paimon import PaimonBosClient
# from .distributed import is_master
from paddle.vision.transforms import pad
# from eva_clip import tokenize

try:
    import horovod.paddle as hvd
except ImportError:
    hvd = None

import warnings
warnings.filterwarnings("ignore")

# from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # Truncated File Read
Image.MAX_IMAGE_PIXELS = None # DecompressionBombWarning
ImageFile.MAX_IMAGE_PIXELS = None

#bos = PaimonBosClient()


def paddle_worker_info(group=None):
    """Return node and worker info for paddle and some distributed environments."""
    rank = 0
    world_size = 1
    worker = 0
    num_workers = 1
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
        try:
            worker_info = get_worker_info()
            if worker_info is not None:
                worker = worker_info.id
                num_workers = worker_info.num_workers
        except ModuleNotFoundError:
            pass

    return rank, world_size, worker, num_workers



class LaionDataset(IterableDataset):
    def __init__(self,
                 file_list,
                 get_text_emb='',
                 data_world_rank=0,
                 data_world_size=1,
                 buffer_size=1,
                 shuffle_every_n_samples=1000,
                 total_seen_samples=None):

        with open(file_list, 'r', encoding='utf-8') as f:
            self.file_list = f.read().strip().split('\n')
        self.get_text_emb = get_text_emb
        self.buffer_size = buffer_size
        self.shuffle_every_n_samples = shuffle_every_n_samples
        self.min_size = 5
        self.total_seen_samples = total_seen_samples
        self.data_world_rank = data_world_rank
        self.data_world_size = data_world_size

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
                text_emb = paddle.to_tensor(text_emb, dtype=paddle.float32)
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
        return data
        # image = image.astype(paddle.float32)
        # image = data['image']
        
        # if data['text_emb'] is not None:
        #     return image, data['text_emb'], data['text']
        # else:
        #     return image, data['text']

        # if self.get_text_emb:
        #     return image, data['text_emb'], data['text']
        # else:
        #     return image, data['text']

    def __len__(self):
        return self.total_seen_samples

    def sample(self):
        _, _, worker, num_workers = paddle_worker_info()
        total_num_workers = num_workers * self.data_world_size
        global_worker_id = self.data_world_rank * num_workers + worker

        print('[CHECK ME] LaionDataset', global_worker_id, total_num_workers)
        while True:
            # random.shuffle(self.file_list)
            for i in range(len(self.file_list)):
                if i % total_num_workers == global_worker_id:
                    filename = self.file_list[i].strip("\n")

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

class TusouDataset(IterableDataset):

    def __init__(self,
                 file_list,
                 transform,
                 get_text_emb='',
                 buffer_size=1,
                 shuffle_every_n_samples=1,
                 total_seen_samples=None):
        self.preprocess = transform

        with open(file_list, 'r', encoding='utf-8') as f:
            self.file_list = f.read().strip().split('\n')
        self.get_text_emb = get_text_emb
        self.buffer_size = buffer_size
        self.shuffle_every_n_samples = shuffle_every_n_samples
        self.min_size = 5
        self.total_seen_samples = total_seen_samples

    def get_data(self, data):
        w, h = data['image'].size
        if w < self.min_size or h < self.min_size:
            return None

        image = self.preprocess(data['image'])
        image = image.astype(paddle.float16)
        if self.get_text_emb:
            return image, data['text_emb'], data['text']
        else:
            return image, data['text']

    def __len__(self):
        return self.total_seen_samples

    def parse_line(self, line):
        try:
        # if True:
            vec = line.strip().split("\t")

            img_b64 = vec[1]
            text = vec[2]

            image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert('RGB')
            return dict(
                image=image,
                text=text,
                text_emb=None
            )
        except:
        # else:
            print(f'error when parse file {filename}')
            return None

    def sample(self):
        rank, world_size, worker, num_workers = paddle_worker_info()
        total_num_workers = num_workers * world_size
        global_worker_id = rank * num_workers + worker

        print('TusouDataset', global_worker_id, total_num_workers)
        while True:
            random.shuffle(self.file_list)
            for i in range(len(self.file_list)):
                if i % total_num_workers == global_worker_id:
                    filename = self.file_list[i].strip("\n")

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
                            data = self.parse_line(line)

                            if data is None:
                                continue
                            else:
                                data = self.get_data(data)
                                if data is None:
                                    continue
                                yield data

    def shuffle(self, iterator):
        buffer_list = []
        for _ in range(self.buffer_size):
            buffer_list.append(next(iterator))
        i = 0
        while True:
            if i % self.shuffle_every_n_samples == 0:
                random.shuffle(buffer_list)
            yield buffer_list.pop()
            buffer_list.append(next(iterator))
            i += 1

    def __iter__(self):
        return self.shuffle(iter(self.sample()))


class SynthDogDataset(IterableDataset):

    def __init__(self,
                 file_list,
                 transform,
                 buffer_size=1,
                 shuffle_every_n_samples=1,
                 total_seen_samples=None):

        self.preprocess = transform

        with open(file_list, 'r', encoding='utf-8') as f:
            self.file_list = f.read().strip().split('\n')
        self.buffer_size = buffer_size
        self.shuffle_every_n_samples = shuffle_every_n_samples
        self.min_size = 5
        self.total_seen_samples = total_seen_samples

    def parse_line(self, line):
        sample = json.loads(line)
        text = sample['text']
        bos_path = os.path.join('bos://nlp-sr-text2img', sample['bos_path'])
        image = Image.open(io.BytesIO(bos.get_bytes(bos_path)))

        image = self.preprocess(self.pad_to_square(image)).astype(paddle.float16)
        return image, text

    def pad_to_square(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return pad(image, padding, 127, 'constant')


    def __len__(self):
        return self.total_seen_samples

    def sample(self):

        rank, world_size, worker, num_workers = paddle_worker_info()
        total_num_workers = num_workers * world_size
        global_worker_id = rank * num_workers + worker

        print('SynthDogDataset', global_worker_id, total_num_workers)
        while True:
            random.shuffle(self.file_list)
            for i in range(len(self.file_list)):
                if i % total_num_workers == global_worker_id:
                    filename = self.file_list[i].strip("\n")
                    try:
                    #if True:
                        raw_samples = bos.get_bytes(filename).decode(encoding='utf-8').split('\n')
                    #else:
                    except:
                        continue
                    for raw_sample in raw_samples:
                        try:
                        #if True:
                            data = self.parse_line(raw_sample)
                            if data is not None:
                                yield data
                        #else:
                        except:
                            continue

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


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.zfile = self.read_zipfile()
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_size(shards):
    multi_shards = shards.split(";")
    shards_list = []
    for sd in multi_shards:
        shards_list.extend(wds.shardlists.expand_urls(sd))
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084 (we have: 407259192)
        # LAION-2B (english): 2170337258 (we have: 1.7b)
    num_shards = len(shards_list)
    return total_size, num_shards


# def get_imagenet(args, preprocess_fns, split):
#     assert split in ["train", "val", "v2"]
#     is_train = split == "train"
#     preprocess_train, preprocess_val = preprocess_fns

#     if split == "v2":
#         from imagenetv2_pypaddle import ImageNetV2Dataset
#         dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
#     else:
#         if is_train:
#             data_path = args.imagenet_train
#             preprocess_fn = preprocess_train
#         else:
#             data_path = args.imagenet_val
#             preprocess_fn = preprocess_val
#         assert data_path

#         dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

#     if is_train:
#         idxs = np.zeros(len(dataset.targets))
#         target_array = np.array(dataset.targets)
#         k = 50
#         for c in range(1000):
#             m = target_array == c
#             n = len(idxs[m])
#             arr = np.zeros(n)
#             arr[:k] = 1
#             np.random.shuffle(arr)
#             idxs[m] = arr

#         idxs = idxs.astype('int')
#         sampler = SubsetRandomSampler(np.where(idxs)[0])
#     else:
#         sampler = None

#     dataloader = paddle.utils.data.DataLoader(
#         dataset,
#         batch_size=args.batch_size//4,
#         num_workers=args.workers,
#         sampler=sampler,
#     )

#     return DataInfo(dataloader=dataloader, sampler=sampler)

def get_classification(args, preprocess_fns):
    # support classification
    result = {}
    preprocess_train, preprocess_val = preprocess_fns

    preprocess_fn = preprocess_val
    data_paths = args.classification_eval.split(",")
    
    for data_path in data_paths:
        data_path = data_path.rstrip('/')
        logging.info(f"adding classification dataset: {data_path}")
        dataset = datasets.ImageFolder(f"{data_path}/images", transform=preprocess_fn)

        dataset = ImageFolder(f"{data_path}/images", transform=preprocess_fn)

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size, # hard code
            num_workers=args.workers,
        )
        
        classname_filename = f"{data_path}/labels.txt"
        template_filename = f"{data_path}/templates.txt"

        result[f"{os.path.basename(data_path)}"] = edict(dataloader=dataloader, classname_filename=classname_filename, template_filename=template_filename)

    return result

def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample or '0.txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample or '0.png' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pypaddle_worker_seed(increment=0):
    """get dataloader worker seed from pypaddle"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pypaddle dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pypaddle_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pypaddle_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
        search=False
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        multi_urls = urls.split(";")
        urls = []
        for sd in multi_urls:
            urls.extend(wds.shardlists.expand_urls(sd))
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def search_urls(self, urls):
        seach_all = []
        for root in urls:
            parts = os.listdir(root)
            part_paths = [os.path.join(root, part) for part in parts]

            all_paths = []
            for part_path in part_paths:
                temp_paths = glob.glob(os.path.join(part_path, '*.tar'))
                all_paths.extend(temp_paths)
            seach_all.extend(all_paths)
        return seach_all

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pypaddle worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pypaddle_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))

def search_urls(urls):
    multi_urls = urls.split(";")
    urls = []
    for sd in multi_urls:
        urls.extend(wds.shardlists.expand_urls(sd))
    seach_all = []
    for root in urls:
        parts = os.listdir(root)
        part_paths = [os.path.join(root, part) for part in parts]

        all_paths = []
        for part_path in part_paths:
            temp_paths = glob.glob(os.path.join(part_path, '*.tar'))
            all_paths.extend(temp_paths)
        seach_all.extend(all_paths)
    return seach_all

def get_real_wds_size(shards):
    all_wds_tars = wds.shardlists.expand_urls(shards)
    count = 0
    for tar_path in all_wds_tars:
        with tarfile.open(tar_path, 'r') as tf:
            all_names = tf.getnames()
            count += len(all_names)//2
    return count



def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if is_master(args, local=args.log_local):
        logging.info(f"Num of shards in {input_shards}: {str(num_shards)}")
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        multi_datasets = input_shards.split(";")
        all_shards = []
        for ds in multi_datasets:
            all_shards.extend(wds.shardlists.expand_urls(ds))
        pipeline = [wds.SimpleShardList(all_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        if args.extract_features:
            pipeline.append(wds.split_by_node)
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])

    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train),
    ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)

class MixingDataLoader:
    """Mixing different datasets with round-robin or weighted round-robin
    Adpated from https://github.com/mlfoundations/open_clip/pull/107/files
    """
    def __init__(self, args, preprocess_train, epoch, tokenizer, sample_weights=False):
        train_data_list = args.train_data_list.split(';')
        dataset_type_list = args.dataset_type_list.split(';') if args.dataset_type_list else ['webdataset' for _ in range(len(train_data_list))]
        if not args.train_num_samples_list or len(args.train_num_samples_list) != len(train_data_list):
            train_num_samples_list = [args.train_num_samples//len(train_data_list) for _ in range(len(train_data_list))]
        else:
            train_num_samples_list = args.train_num_samples_list

        data_train = []
        for train_data, dataset_type, train_num_samples in zip(train_data_list, dataset_type_list, train_num_samples_list):
            args.train_data = train_data
            args.train_num_samples = train_num_samples
            data_train.append(
                get_dataset_fn(train_data, dataset_type)(
                    args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)
            )

        self.args = args
        self.num_datasets = len(data_train)
        self.dataloaders = [dataset.dataloader for dataset in data_train]
        self.dataiters = [iter(dataloader) for dataloader in self.dataloaders]
        self.datasets = train_data_list
        self.num_batches = sum([dataloader.num_batches for dataloader in self.dataloaders])
        self.num_samples = sum([dataloader.num_samples for dataloader in self.dataloaders])

        # calculate sample weights according to num_samples of multiple datasets
        self.sample_weights = np.array([float(dataloader.num_samples) / self.num_samples for dataloader in self.dataloaders]) if sample_weights else None

        if is_master(args, local=args.log_local):
            # print("List of training datasets in MixingDataLoader: ", train_data_list)
            logging.info("Training datasets with virtual epcoh samples in MixingDataLoader:")
            for dataset, num_samples in zip(train_data_list, train_num_samples_list):
                logging.info(f"\t{num_samples} samples per virtual epoch -> {dataset}")
            logging.info(f"Num of datasets in MixingDataLoader: {self.num_datasets}")
            logging.info(f"Num of samples in MixingDataLoader: {self.num_samples}")
            logging.info(f"Num of batches in MixingDataLoader: {self.num_batches}")
            if self.sample_weights is None:
                logging.info("Disable sample_weights...")
            else:
                logging.info(f"Enable sample_weights: {self.sample_weights}")

        self.count = 0
        self.current_epoch = epoch #0
        self.data_train = data_train
        if self.args.distributed and data_train is not None:
            for data_info in data_train:
                data_info.set_epoch(epoch)
    def __len__(self):
        return self.num_samples

    def __iter__(self):
        while True:
            if self.count == self.num_batches:
                self.current_epoch += 1
                self.count = 0
                if self.args.distributed and self.data_train is not None:
                    for data_info in self.data_train:
                        data_info.set_epoch(self.current_epoch)
                return # end each epoch

            # set random seed for sampling from the same dataset.
            # sample a dataset according to sample_weights
            if self.sample_weights is not None:
                stable_random_seed = int(self.count + self.num_batches * self.current_epoch)
                np.random.seed(stable_random_seed)
                iter_index = np.random.choice(range(self.num_datasets), p=self.sample_weights)
            else:
                iter_index = self.count % self.num_datasets
            # generate training image-text pairs from the sampled dataset.
            try:
                data_iter = self.dataiters[iter_index]
                batch = next(data_iter)
            except StopIteration:
                # refresh dataiter if dataloader is used up.
                self.dataiters[iter_index] = iter(self.dataloaders[iter_index])
                data_iter = self.dataiters[iter_index]
                batch = next(data_iter)

            self.count += 1

            yield batch

def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
		tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

class SyntheticDataset(Dataset):

    def __init__(self, transform=None, image_size=(224, 224), caption="Dummy caption", dataset_size=100, tokenizer=None):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

def get_mixing_dataset_fn(args, preprocess_train, epoch, tokenizer):
    sample_weights = True if args.train_num_samples_list else False
    dataloader = MixingDataLoader(args, preprocess_train, epoch, tokenizer, sample_weights)
    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    return DataInfo(dataloader, shared_epoch)


def laion_collate_fn(batch, tokenizer):
    batch = list(zip(*batch))
    if len(batch) == 3:
        images, text_embs, texts = batch
    elif len(batch) == 2:
        images, texts = batch
        text_embs = None
    else:
        raise ValueError(len(batch))

    images = paddle.stack(images)
    if text_embs is not None:
        text_embs = paddle.stack(text_embs)

    input_ids = tokenizer(texts)
    if not isinstance(input_ids, paddle.Tensor):
        input_ids = input_ids['input_ids']

    labels = input_ids.clone()
    labels[labels == 0] = -100
    labels[:, -1] = -100
    return images, text_embs, input_ids, labels

def laion_dataset_fn(train_data, preprocess_train, epoch, tokenizer, args):
    if tokenizer is None:
        tokenizer = tokenize
    # pipeline = [LaionDataset(train_data, preprocess_train, get_text_emb=args.precomputed_text_emb), wds.batched(args.batch_size, partial=False)]
    # dataset = wds.DataPipeline(*pipeline)
    dataset = LaionDataset(train_data, preprocess_train, get_text_emb=args.precomputed_text_emb)

    num_samples = args.train_num_samples
    round_fn = math.ceil
    global_batch_size = args.batch_size * args.world_size
    num_batches = round_fn(num_samples / global_batch_size) # global batch number
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    # dataset = islice(dataset, num_batches) # FIXME: check this

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size, # FIXME: check this
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
        collate_fn=partial(laion_collate_fn, tokenizer=tokenizer) #if args.coca else None,
    )
    # dataloader = wds.WebLoader(
    #     dataset,
    #     batch_size=None,
    #     shuffle=False,
    #     num_workers=args.workers,
    #     persistent_workers=True,
    #     collate_fn=partial(laion_collate_fn, tokenizer=tokenizer) if args.coca else None,
    # )
    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    dataloader.num_batches = num_batches * args.batch_size
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)

def tusou_collate_fn(batch, tokenizer):
    images, texts = zip(*batch)
    images = paddle.stack(images)

    input_ids = tokenizer(texts)
    if not isinstance(input_ids, paddle.Tensor):
        input_ids = input_ids['input_ids']
    labels = input_ids.clone()
    labels[labels == 0] = -100 #TODO: check this 
    labels[:, -1] = -100
    return images, input_ids, labels

def tusou_dataset_fn(train_data, preprocess_val, tokenizer, args):
    pipeline = [TusouDataset(train_data, preprocess_val, get_text_emb=args.precomputed_text_emb)]
    dataset = wds.DataPipeline(*pipeline)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
        collate_fn=partial(tusou_collate_fn, tokenizer=tokenizer),
    )

    return dataloader

def synthdog_collate_fn(batch, tokenizer):
    images = paddle.stack([sample[0] for sample in batch])
    texts = [sample[1] for sample in batch]

    input_ids = tokenizer(texts)
    if not isinstance(input_ids, paddle.Tensor):
        input_ids = input_ids['input_ids']

    labels = input_ids.clone()
    labels[labels == 0] = -100
    labels[:, -1] = -100
    return images, input_ids, labels

def synthdog_dataset_fn(train_data, preprocess_val, tokenizer, args):

    pipeline = [SynthDogDataset(train_data, preprocess_val)]
    dataset = wds.DataPipeline(*pipeline)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=args.ocr_batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
        collate_fn=partial(synthdog_collate_fn, tokenizer=tokenizer),
    )

    return dataloader


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data_list:
        # sample_weights = True if args.train_num_samples_list else False
        # dataloader = MixingDataLoader(args, preprocess_train, epoch, tokenizer, sample_weights)
        # shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
        # data["train"] = DataInfo(dataloader, shared_epoch)

        data["train"] = get_mixing_dataset_fn(args, preprocess_train, epoch, tokenizer)

    elif args.train_data:
        if args.dataset_type == "laion":
            data["train"] = laion_dataset_fn(args.train_data, preprocess_train, epoch, tokenizer, args)

        elif args.train_data or args.dataset_type == "synthetic":
            data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
                args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    # if args.imagenet_val is not None:
    #     data["eval/classification/imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    # if args.imagenet_v2 is not None:
    #     data["eval/classification/imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    if args.classification_eval is not None:
        tmp = get_classification(args, preprocess_fns)
        for k, v in tmp.items():
            data[f'eval/classification/{k}'] = v

    return data

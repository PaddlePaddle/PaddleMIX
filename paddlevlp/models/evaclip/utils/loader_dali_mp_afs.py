#coding:utf-8
import base64
from cgi import print_arguments
import os
import numpy as np
import argparse
import time
import random
from functools import partial
import queue
from PIL import Image
from io import BytesIO
import cv2
import json

import multiprocessing as mp
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.paddle import DALIGenericIterator
import paddle
from paddle.distributed import fleet

# huggingface
from transformers import AutoTokenizer
from transformers import BertTokenizer
from .WordTokenization import WordTokenizer
from .SimpleTokenization import SimpleTokenizer


class ChineseTokenizer:
    def __init__(self, context_length, truncate_text=True,
                 word_tokenizer=None):
        #tokenizer = AutoTokenizer.from_pretrained(word_tokenizer)
        tokenizer = BertTokenizer.from_pretrained(word_tokenizer)
        #print(tokenizer)
        self.tokenizer = partial(
            tokenizer,
            padding="max_length",
            add_special_tokens=True,
            max_length=context_length,
            truncation=truncate_text)
        self.vocab_size = tokenizer.vocab_size

    def decode(self, tokens):
        if paddle.is_tensor(tokens):
            tokens = paddle.tolist(tokens)

        tokens = [token for token in tokens if token not in (0, )]
        return self.tokenizer.decode(tokens)

    def encode(self, text):
        return paddle.to_tensor(
            self.tokenizer.encode(
                text, add_special_tokens=False))

    def tokenize(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        result = self.tokenizer(texts)
        return result.input_ids, result.attention_mask


def data_prefetch_queue(batch_size, part_list_queue, prefetch_queue, STOP_FLAG,
                        BREAK_FLAG, tokenizer):
    def get_line():
        while not part_list_queue.empty():
            try:
                one_part = part_list_queue.get(timeout=2)
            except:
                raise StopIteration
            with open(one_part, encoding='utf-8') as f:
                for line in f:
                    try:
                        data = line.strip().split('\t')
                        img = data[2]
                        text = data[3]
                        if len(text) == 0:
                            text = ""
                        label = int(random.random() * 100000000)

                    except:
                        continue
                    if len(text) == 0: continue
                    try:
                        img = np.frombuffer(
                            base64.urlsafe_b64decode(img), np.uint8)
                    except:
                        continue
                    yield [img, text, label]

    line_generator = get_line()
    images = []
    descriptions = []
    labels = []

    while True:
        if not BREAK_FLAG.is_set():
            break
        try:
            img, text, label = next(line_generator)
        except StopIteration:
            break
        except Exception as e:
            print('data load error: %s' % e)
        images.append(img)
        labels.append(label)
        descriptions.append(text)
        if len(images) == batch_size:
            tokenized_text, tokenized_type_ids = tokenizer.tokenize(
                descriptions)
            prefetch_queue.put(
                [images, tokenized_text, tokenized_type_ids, labels])
            images = []
            descriptions = []
            labels = []
        elif len(images) < batch_size:
            pass
        else:
            raise BaseException("one_batch error")


class prefetch_worker(object):
    def run(self, n_processes, args):
        random.seed(args.random_seed)
        random.shuffle(args.part_list_all)
        # word_tokenizer = ChineseTokenizer(context_length=args.context_length, truncate_text=True, word_tokenizer=args.huggingface_pretrain)
        # word_tokenizer = WordTokenizer(
        #     vocab_file='./utils/vocab.txt', context_length=args.context_length)
        word_tokenizer = SimpleTokenizer(context_length=args.context_length)
        args.part_list = [
            i for cnt, i in enumerate(args.part_list_all)
            if cnt % args.data_world_size == args.data_world_rank
        ]  # 乘2过于浪费，reload耗时多，+1组足够
        args.part_list += args.part_list[:1]
        for part in args.part_list:
            args.part_list_queue.put(part)
        self.n_processes = n_processes
        args.p_list = []
        for _ in range(self.n_processes):
            p = mp.Process(
                target=data_prefetch_queue,
                args=(
                    args.batch_size,
                    args.part_list_queue,
                    args.global_data_prefetch_queue,
                    args.STOP_FLAG,
                    args.BREAK_FLAG,
                    word_tokenizer, ))
            p.start()
            args.p_list.append(p)

    def queue_reload(self, args):
        while not args.global_data_prefetch_queue.qsize(
        ) == 0:  #.empty(): 当使用empty进行判断时，当queue数量小于20时，直接判别为True,具体原因未知，不可靠
            args.global_data_prefetch_queue.get()
        while not args.part_list_queue.qsize() == 0:  #.empty():
            args.part_list_queue.get()
        for part in args.part_list:
            args.part_list_queue.put(part)
        print("Reloading Data Complete")
        args.BREAK_FLAG.set()
        self.run(self.n_processes, args)


class ExternalInputIterator(object):
    def __init__(self, prefetch_queue):
        self.prefetch_queue = prefetch_queue

    def __iter__(self):
        return self

    def __next__(self):
        mp_rank = fleet.get_hybrid_communicate_group().get_model_parallel_rank(
        )
        if mp_rank == 0:
            raw_data = self.prefetch_queue.get()
            # print("current queue size is %s" % self.prefetch_queue.qsize())
            images, tokenized_text, type_ids, labels = raw_data
            return (images, np.array(tokenized_text), np.array(type_ids),
                    np.array(
                        labels, dtype=np.int64))
        return None

    next = __next__


class ExternalSourcePipeline(Pipeline):
    def __init__(self,
                 num_threads,
                 device_id,
                 gpu_prefetch_queue_depth,
                 batch_size,
                 external_data,
                 image_size=224,
                 context_length=30,
                 seed=-1):
        super(ExternalSourcePipeline, self).__init__(
            num_threads=num_threads,
            device_id=device_id,
            prefetch_queue_depth=gpu_prefetch_queue_depth,
            batch_size=batch_size,
            seed=seed)
        self.input = ops.ExternalSource()
        self.input_text = ops.ExternalSource()
        self.input_type_ids = ops.ExternalSource()
        self.input_labels = ops.ExternalSource()
        self.coin = ops.CoinFlip(probability=0.5)
        self.decode = ops.ImageDecoderRandomCrop(
            device="cpu",
            output_type=types.RGB,
            random_aspect_ratio=[1.0, 1.0],
            random_area=[0.75, 1.0],
            num_attempts=100)
        self.resize = ops.Resize(
            device="cpu",
            resize_x=image_size,
            resize_y=image_size,
            interp_type=types.INTERP_TRIANGULAR)

        self.norm = ops.CropMirrorNormalize(
            std=255., mean=0, device="cpu", output_layout=types.NCHW)
        self.norm2 = ops.CropMirrorNormalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            device="cpu")
        self.external_data = external_data
        self.iterator = iter(self.external_data)
        self.image_size = image_size
        self.context_length = context_length

    def define_graph(self):
        self.images = self.input()
        self.descriptions = self.input_text()
        self.type_ids = self.input_type_ids()
        self.labels = self.input_labels()
        mp_rank = fleet.get_hybrid_communicate_group().get_model_parallel_rank(
        )
        if mp_rank == 0:
            images = self.decode(self.images)
            output = self.resize(images)
            coin_mirror = self.coin()
            output = self.norm2(self.norm(output, mirror=coin_mirror))

            self.descriptions = self.descriptions.gpu()
            self.type_ids = self.type_ids.gpu()
            self.labels = self.labels.gpu()
            return output, self.descriptions, self.type_ids, self.labels
        else:
            output = self.images.gpu()
            self.descriptions = self.descriptions.gpu()
            self.type_ids = self.type_ids.gpu()
            self.labels = self.labels.gpu()
            return output, self.descriptions, self.type_ids, self.labels

    def iter_setup(self):
        mp_rank = fleet.get_hybrid_communicate_group().get_model_parallel_rank(
        )
        if mp_rank == 0:
            (images, descriptions, type_ids, labels) = self.iterator.next()
        else:
            images = np.zeros(
                (self.max_batch_size, 3, self.image_size, self.image_size),
                dtype=np.float32)
            descriptions = np.zeros(
                (self.max_batch_size, self.context_length), dtype=np.int64)
            type_ids = np.zeros(
                (self.max_batch_size, self.context_length), dtype=np.int64)
            labels = np.zeros((self.max_batch_size, ), dtype=np.int64)

        self.feed_input(self.images, images)
        self.feed_input(self.descriptions, descriptions)
        self.feed_input(self.type_ids, type_ids)
        self.feed_input(self.labels, labels)


def get_dali_loader(gpu_prefetch_queue_depth,
                    batch_size,
                    device_id,
                    num_threads,
                    prefetch_queue,
                    size_per_pipeline,
                    image_size=224,
                    context_length=30,
                    seed=-1):
    eii = ExternalInputIterator(prefetch_queue)
    custom_pipeline = ExternalSourcePipeline(
        num_threads=num_threads,
        device_id=device_id,
        gpu_prefetch_queue_depth=gpu_prefetch_queue_depth,
        batch_size=batch_size,
        external_data=eii,
        image_size=image_size,
        context_length=context_length,
        seed=seed)
    custom_pipeline.build()
    paddle_loader = DALIGenericIterator(
        custom_pipeline,
        output_map=["images", "text_token", "type_ids", "labels"],
        size=size_per_pipeline,
        auto_reset=False)
    return paddle_loader


if __name__ == "__main__":
    import multiprocessing as mp
    queue = mp.Queue(maxsize=2048)
    train_loader = get_dali_loader(
        gpu_prefetch_queue_depth=1,
        batch_size=10,
        device_id=0,
        num_threads=4,
        prefetch_queue=queue,
        size_per_pipeline=125000,
        image_size=224)

    a = 1

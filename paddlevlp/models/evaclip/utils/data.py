import os
import cv2
import numpy as np
import json
import copy
import pycocotools
from pycocotools.coco import COCO
from paddle.io import Dataset, DataLoader, DistributedBatchSampler
from paddle.vision import transforms as T
from tqdm import tqdm
from utils.SimpleTokenization import SimpleTokenizer


class CoCoCaptionDataset(Dataset):
    """COCO dataset for clip. 

    Args:
        dataset_dir (str): Root path to the dataset.
        image_dir (str): Path to a directory where images are held.
        anno_path (str): Relative path to the annotation file.
        trainsize (list):[w, h] Image target size
        transform (composed(operators)): A sequence of data transforms.
    """

    def __init__(self,
                 dataset_dir,
                 image_dir,
                 anno_path,
                 trainsize=256,
                 transform=[],
                 tokenizer=SimpleTokenizer()):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.image_dir = image_dir
        self.img_prefix = os.path.join(dataset_dir, image_dir)
        self.anno_path = anno_path
        self.transform = transform
        self.trainsize = trainsize
        self.tokenizer = tokenizer
        self.dataset_name = 'coco'
        self.parse_dataset()

    def __len__(self):
        """Get dataset length."""
        return len(self.db)

    def reset(self):
        pass

    def get_anno(self):
        if self.anno_path is None:
            return
        return os.path.join(self.dataset_dir, self.anno_path)

    def parse_dataset(self):
        self.db = self._load_coco_annotations()

    def _load_coco_annotations(self):
        coco = COCO(self.get_anno())
        img_ids = coco.getImgIds()
        gt_db = []
        for index in img_ids:
            im_ann = coco.loadImgs(index)[0]
            width = im_ann['width']
            height = im_ann['height']
            file_name = im_ann['file_name']
            im_id = int(im_ann["id"])

            annIds = coco.getAnnIds(imgIds=index)
            objs = coco.loadAnns(annIds)
            texts = []
            for obj in objs:
                texts.append(obj['caption'].strip())
            if os.path.exists(os.path.join(self.img_prefix,
                                           file_name)) and len(texts) >= 1:
                rec = {
                    'image_file': os.path.join(self.img_prefix, file_name),
                    'description': texts,
                    'im_id': im_id,
                }
                gt_db.append(rec)
            else:
                print("warning, file not found:{}".format(
                    os.path.join(self.img_prefix, file_name)))

        return gt_db

    def __getitem__(self, idx):
        """Prepare sample for training given the index."""
        records = copy.deepcopy(self.db[idx])
        records['images'] = cv2.imread(records['image_file'],
                                       cv2.IMREAD_COLOR |
                                       cv2.IMREAD_IGNORE_ORIENTATION)
        records['images'] = cv2.cvtColor(records['images'], cv2.COLOR_BGR2RGB)
        records['images'] = self.transform(records['images'])

        # desc = np.random.choice(records['description'], 1)
        desc = records['description'][0]
        tokenized_text, tokenized_type_ids = self.tokenizer.tokenize(desc)
        records['text_token'] = np.array(tokenized_text[0])
        records['type_ids'] = np.array(tokenized_type_ids[0])
        del records['description']
        # print(len(records.keys()), records.keys())
        # print("image shape:{}, token:{}; type_ids:{}; desc:{}".format(records['images'].shape, len(tokenized_text[0]), len(tokenized_type_ids[0]), desc))
        # import pdb;pdb.set_trace()
        return records


def reset_imagenum(args, dataset):
    if args is None:
        return
    args.size_per_pipeline = len(dataset) // (args.dp_degree *
                                              args.sharding_degree)
    args.total_step = args.size_per_pipeline // args.batch_size // args.accum_freq  # num_batches // accum_freq


def get_data(args,
             dataset_dir=None,
             batch_size=16,
             trainsize=224,
             shuffle=False,
             drop_last=False,
             use_shared_memory=True,
             phase='train'):

    if dataset_dir is None:
        dataset_dir = "/home/niuzhibo/PaddleDetection/dataset/coco/"
    images = "{}2017".format(phase)
    anno_path = "annotations/captions_{}2017.json".format(phase)

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    if phase == 'train':
        randomresizecrop = T.RandomResizedCrop(
            trainsize, scale=(0.9, 1.0), interpolation='bicubic')
        normalize = T.Normalize(mean=mean, std=std)
        totensor = T.ToTensor()
        transform = T.Compose([randomresizecrop, totensor, normalize])
    else:
        resize = T.Resize(trainsize, interpolation='bicubic')
        crop = T.CenterCrop(trainsize)
        normalize = T.Normalize(mean=mean, std=std)
        totensor = T.ToTensor()
        transform = T.Compose([resize, crop, totensor, normalize])

    dataset = CoCoCaptionDataset(
        dataset_dir, images, anno_path, transform=transform)
    reset_imagenum(args, dataset)

    batch_sampler = DistributedBatchSampler(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=(phase == 'train'))

    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=4,
        return_list=True,
        use_shared_memory=use_shared_memory)

    return dataloader


if __name__ == '__main__':
    trainloader = get_data(
        None, dataset_dir='/home/niuzhibo/PaddleDetection/dataset/coco/')
    cnt = 0
    for data in trainloader:
        print(data['images'].shape,
              np.array(data['text_token']).shape,
              np.array(data['type_ids']).shape)
        cnt += 1
        if cnt >= 3:
            break

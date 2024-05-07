#!/usr/bin/env python3
import itertools
from typing import List
import numpy as np
import random
import paddle
import paddle.nn as nn
from ppdet.core.workspace import create, load_config, merge_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.engine import Trainer
from tqdm import tqdm
import yolo_world

def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="image file or folder")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="text or .txt file")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    paddle.seed(3407)
    np.random.seed(3407)
    random.seed(3407)

    # load config
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)

    # load text
    if cfg.text.endswith('.txt'):
        with open(cfg.text) as f:
            lines = f.readlines()
        texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
    else:
        texts = [[t.strip()] for t in cfg.text.split(',')] + [[' ']]

    trainer = Trainer(cfg, mode='test')
    trainer.load_weights(cfg.weights)

    for k, m in trainer.model.named_sublayers():
        if isinstance(m, nn.BatchNorm2D):
            m._epsilon = 1e-3  # for amp(fp16)
            m._momentum = 0.97  # 0.03 in pytorch

    trainer.model.eval()

    if not isinstance(cfg.images, List):
        cfg.images = [cfg.images]
    trainer.dataset.set_images(cfg.images)

    loader = create("TestReader")(trainer.dataset, 0)
    for step_id, data in enumerate(tqdm(loader)):
        data["texts"] = [list(itertools.chain.from_iterable(texts))]
        trainer.status["step_id"] = step_id
        # forward
        img_feats, text_feats = trainer.model(data)

        # from reprod_log import ReprodLogger
        # reprod_logger = ReprodLogger()
        # paddle_out = text_feats
        # reprod_logger.add("logits", paddle_out.cpu().detach().numpy())
        # reprod_logger.save("/home/onion/workspace/code/pp/Alignment/backbones/paddle_textfeats.npy")
        # reprod_logger = ReprodLogger()
        # paddle_out = img_feats[0]
        # reprod_logger.add("logits", paddle_out.cpu().detach().numpy())
        # reprod_logger.save("/home/onion/workspace/code/pp/Alignment/backbones/paddle_imgfeats1.npy")
    print(trainer.model.state_dict().keys())

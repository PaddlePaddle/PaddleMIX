## OPEN-MAGVIT2: An Open-source Project Toward Democratizing Auto-Regressive Visual Generation
The Open-MAGVIT2 project produces an open-source replication of Google's MAGVIT-v2 tokenizer, a tokenizer with a super-large codebook, and achieves the state-of-the-art reconstruction performance (1.17 rFID) on ImageNet 256 x 256 .

This project provides the implementation of PaddlePaddle, support training and infer for visual tokenizer、infer reconstruct.

#### Installation
#### GPU
- **Env**: We have tested on `Python 3.10` and `CUDA 11.8` (other versions may also be fine).
- **PaddlePaddle** `Paddle3.0beta1`
- **Dependencies**: `pip install -r requirements.txt`


#### Datasets

We use Imagenet2012 as our dataset. download the dataset from [imagenet](http://www.image-net.org/), and put it into `data` directory:
```
data
└── train/
    ├── n01440764
        ├── n01440764_10026.JPEG
        ├── n01440764_10027.JPEG
        ├── ...
    ├── n01443537
    ├── ...
└── val/
    ├── ...
```

#### Directory structure
```
visual_tokenizer
└── data
    └── train/
    └── val/
└── open-magvit2
    └── train_tokenizer.py
    ├── ...

```



### Stage I: Training of Visual Tokenizer
<!-- * `Stage I Tokenizer Training`: -->
####  Training Scripts
* $128\times 128$ Tokenizer Training
```
#single gpu
python train_tokenizer.py --config configs/gpu/imagenet_lfqgan_128_L.yaml

#multiple gpu
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" train_tokenizer.py  --config configs/gpu/imagenet_lfqgan_128_L.yaml
```

* $256\times 256$ Tokenizer Training
```
#single gpu
python train_tokenizer.py --config configs/gpu/imagenet_lfqgan_256_L.yaml

#multiple gpu
python -u  -m paddle.distributed.launch --gpus "0,1,2,3" train_tokenizer.py  --config configs/gpu/imagenet_lfqgan_256_L.yaml
```

Some important parameter configuration instructions, please refer to  `configs/gpu/imagenet_lfqgan_2256_L.yaml`
```

trainer:
  precision: bfloat16  # float32 or bfloat16
  max_epochs: 1        # max epochs
  max_steps: 10        # max steps
  num_sanity_val_steps: 5      # sanity check
  log_every_n_steps: 5         # log every n steps
  save_checkpoint_steps:       # save checkpoint every n steps
  save_checkpoint_epochs: 1    # save checkpoint every n epochs
  save_path: "checkpoints"     # save checkpoint path

......


ckpt_path: null  # to resume
```

####  Infer Scripts

* $256\times 256$ reconstruct
```
wget https://bj.bcebos.com/v1/paddlenlp/models/community/paddlemix/imagenet_256_L.pdparams
sh scripts/inference/reconstruct.sh
```

## ✏️ Citation
If you found the codebase and our work helpful, please cite it and give us a star :star:.
```
@article{luo2024open,
  title={Open-MAGVIT2: An Open-Source Project Toward Democratizing Auto-regressive Visual Generation},
  author={Luo, Zhuoyan and Shi, Fengyuan and Ge, Yixiao and Yang, Yujiu and Wang, Limin and Shan, Ying},
  journal={arXiv preprint arXiv:2409.04410},
  year={2024}
}

@inproceedings{yu2024language,
  title={Language Model Beats Diffusion - Tokenizer is key to visual generation},
  author={Lijun Yu and Jose Lezama and Nitesh Bharadwaj Gundavarapu and Luca Versari and Kihyuk Sohn and David Minnen and Yong Cheng and Agrim Gupta and Xiuye Gu and Alexander G Hauptmann and Boqing Gong and Ming-Hsuan Yang and Irfan Essa and David A Ross and Lu Jiang},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=gzqrANCF4g}
}
```

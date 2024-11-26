## OPEN-MAGVIT2: An Open-source Project Toward Democratizing Auto-Regressive Visual Generation


#### Datasets

We use Imagenet2012 as our dataset.
```
imagenet
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

### Stage I: Training of Visual Tokenizer
<!-- * `Stage I Tokenizer Training`: -->
#### 🚀 Training Scripts
* $128\times 128$ Tokenizer Training
```
bash scripts/train_tokenizer/run_128_L.sh MASTER_ADDR MASTER_PORT NODE_RANK
```

* $256\times 256$ Tokenizer Training
```
bash scripts/train_tokenizer/run_256_L.sh MASTER_ADDR MASTER_PORT NODE_RANK
```

#### 🚀 Evaluation Scripts
* $128\times 128$ Tokenizer Evaluation
```
bash scripts/evaluation/evaluation_128.sh
```

* $256\times 256$ Tokenizer Evaluation
```
bash scripts/evaluation/evaluation_256.sh
```

#### 🍺 Performance and Models

**Tokenizer** 
| Method | Token Type | #Tokens | Train Data | Codebook Size | rFID | PSNR  | Codebook Utilization | Checkpoint |
|:------:|:----:|:-----:|:-----:|:-------------:|:----:|:----:|:---------------------:|:----:|
|Open-MAGVIT2-20240617| 2D | 16 $\times$ 16 | 256 $\times$ 256 ImageNet | 262144 | 1.53 | 21.53 | 100% | - |
|Open-MAGVIT2-20240617| 2D | 16 $\times$ 16 | 128 $\times$ 128 ImageNet | 262144 | 1.56 | 24.45 | 100% | - |
|Open-MAGVIT2| 2D | 16 $\times$ 16 | 256 $\times$ 256 ImageNet | 262144 | **1.17** | **21.90** | **100%** | [IN256_Large](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/imagenet_256_L.ckpt)|
|Open-MAGVIT2| 2D | 16 $\times$ 16 | 128 $\times$ 128 ImageNet | 262144 | **1.18** | **25.08** | **100%** |[IN128_Large](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/imagenet_128_L.ckpt)|
|Open-MAGVIT2*| 2D | 32 $\times$ 32 | 128 $\times$ 128 ImageNet | 262144 | **0.34** | **26.19** | **100%** |above|

(*) denotes that the results are from the direct inference using the model trained with $128 \times 128$ resolution without fine-tuning.

### Stage II: Training of Auto-Regressive Models

#### 🚀 Training Scripts
Please see in scripts/train_autogressive/run.sh for different model configurations.
```
bash scripts/train_autogressive/run.sh MASTER_ADDR MASTER_PORT NODE_RANK
```

#### 🚀 Sample Scripts
Please see in scripts/train_autogressive/run.sh for different sampling hyper-parameters for different scale of models.
```
bash scripts/evaluation/sample_npu.sh or scripts/evaluation/sample_gpu.sh Your_Total_Rank
```

#### 🍺 Performance and Models
| Method | Params| #Tokens | FID | IS | Checkpoint |
|:------:|:-----:|:-------:|:---:|:--:|:----------:|
|Open-MAGVIT2| 343M | 16 $\times$ 16 | 3.08 | 258.26 | [AR_256_B](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/AR_256_B.ckpt)|
|Open-MAGVIT2| 804M | 16 $\times$ 16 | 2.51 | 271.70 | [AR_256_L](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/AR_256_L.ckpt)|
|Open-MAGVIT2| 1.5B | 16 $\times$ 16 | 2.33 | 271.77 | [AR_256_XL](https://huggingface.co/TencentARC/Open-MAGVIT2/blob/main/AR_256_XL.ckpt)|

## ❤️ Acknowledgement
We thank [Lijun Yu](https://me.lj-y.com/) for his encouraging discussions. We refer a lot from [VQGAN](https://github.com/CompVis/taming-transformers) and [MAGVIT](https://github.com/google-research/magvit). We also refer to [LlamaGen](https://github.com/FoundationVision/LlamaGen), [VAR](https://github.com/FoundationVision/VAR) and [RQVAE](https://github.com/kakaobrain/rq-vae-transformer). Thanks for their wonderful work.

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


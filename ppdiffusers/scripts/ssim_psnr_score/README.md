# SSIM and PSNR

SSIM（Structural Similarity Index）是一种用于衡量两幅图像结构相似度的指标，常用于图像质量评价任务。与像素级别的误差不同，SSIM 模拟人类视觉系统从亮度、对比度和结构等多个维度来评估图像之间的差异。其取值范围为 [-1, 1]，其中 1 表示两张图像完全相同，值越高说明图像质量越接近参考图像。

PSNR（Peak Signal-to-Noise Ratio）是一种基于像素差异的图像质量评估指标，用于衡量图像压缩或生成后与参考图像之间的误差大小。它通过最大像素值与均方误差（MSE）之间的比值计算得出，通常以分贝（dB）为单位。PSNR 值越高表示重建图像与原图越接近，图像失真越小。对于 8-bit 图像，PSNR 值通常大于 30dB 被认为质量良好。



## 依赖
- math
- numpy==1.26.4
- cv2

## 快速使用
计算两个图片数据集的SSIM与PSNR， `path/to/dataset1`/`path/to/dataset2`为图片文件夹

```
python evaluation.py --dataset1 path/to/dataset1 --dataset2 path/to/dataset2
```
图片数据集的结构应如下：
```shell
├── dataset

    ├── 00000.png
    ├── 00001.png
    ......
    ├── 00999.png

```

参数说明
- `num-workers`： 用于加载数据的子进程个数，默认为`min(8, num_cpus)`。
- `resolution`：调整图片的分辨率


## 参考
- [https://github.com/ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache)

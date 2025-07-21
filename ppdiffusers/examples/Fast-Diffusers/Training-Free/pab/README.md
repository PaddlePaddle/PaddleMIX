# PAB

## 快速简介

> **PAB** 是一种 _training‑free_ 的推理加速方法：在空间注意力、时间注意力和交叉注意力机制进行设置了不同的广播范围进行复用缓存，可为Flux模型带来的速度提升。



## 使用方法
```
python text_to_image_generation_pab_flux.py
```

## 参数详解

| 字段                   | 类型   | 意义                                             | 常用取值    |
|------------------------|--------|--------------------------------------------------|-------------|
| `spatial_attention_block_skip_range` | `int`  |     空间注意力复用范围       | `2 – 4`    |
| `temporal_attention_block_skip_range`  | `int`  | 时间注意力复用范围               | `3 - 5`   |
| `cross_attention_block_skip_range`  | `int`  | 交叉注意力复用范围               | `4 - 6`   |
| `spatial_attention_timestep_skip_range`  | `int`  | 注意力时间步复用范围               | `100，800`   |

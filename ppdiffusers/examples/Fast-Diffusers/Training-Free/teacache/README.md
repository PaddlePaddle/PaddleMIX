# TeaCache

## 快速简介

> **TeaCache** 是一种 _training‑free_ 的推理加速方法：通过对于step级别的复用，判断前后step的输出l1距离作为是否可以复用的条件，可为Flux模型带来的速度提升。



## 使用方法
```
python text_to_image_generation_teacache_flux.py
```

## 参数详解

| 字段                   | 类型   | 意义                                             | 常用取值    |
|------------------------|--------|--------------------------------------------------|-------------|
| `enable_teacache` | `bool`  |     是否使用teacache方法       | `True`    |
| `rel_l1_thresh`  | `float`  | l1距离的阈值               | `0.25 - 0.8`   |

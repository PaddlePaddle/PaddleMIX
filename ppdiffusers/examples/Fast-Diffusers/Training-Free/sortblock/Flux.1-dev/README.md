# SortBlockCache

## 快速简介

> **SortBlockCache** 是一种 _training‑free_ 的推理加速方法：对于block相似度进行排序，选择相似度差距大的block进行重计算，利用泰勒展开原理，对于复用的cache就行预测，可为Flux模型带来的速度提升。



## 使用方法
```
python text_to_image_generation_sortblock_flux.py
```

## 参数详解

| 字段                   | 类型   | 意义                                             | 常用取值    |
|------------------------|--------|--------------------------------------------------|-------------|
| `start`            | `int`  |     复用策略开始时间步       | `900 - 950`    |
| `end`  | `int`  | 复用策略结束时间步               | `50 - 100`   |
| `percentage`  | `int`  | 排序百分比初始值              | `1`   |
| `step_Num2`  | `int`  | 复用策略力度               | `4 - 9`   |
| `count`  | `int`  | 时间步计数初始值              | `0`   |
| `beta`  | `float`  | 排序百分比超参数              | `0 - 1`   |

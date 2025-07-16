# taylorseer

## 快速简介

> **taylorseer** 是一种 _training‑free_ 的推理加速方法：利用泰勒展开原理，对于复用的cache就行预测，可为Flux模型带来的速度提升。



## 使用方法
```
python text_to_image_generation_taylorseer_flux.py
```

## 参数详解

| 字段                   | 类型   | 意义                                             | 常用取值    |
|------------------------|--------|--------------------------------------------------|-------------|
| `fresh_threshold`            | `int`  |     复用并预测的步数       | `2 – 6`    |
| `max_order`  | `int`  | 泰勒展开阶数               | `1 - 3`   |

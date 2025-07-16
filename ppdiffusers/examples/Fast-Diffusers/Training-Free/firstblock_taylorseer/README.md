# firstblock_taylorseer

## 快速简介

**firstblock_taylorseer** 是一种 _training‑free_ 的推理加速方法：通过对于step级别的复用，利用teacache的判断方法，判断前后step的第一个block的输出l1距离作为是否可以复用的条件，并使用taylorseer预测下一时间步的值，可为Flux模型带来的速度提升。



## 使用方法
```
python text_to_image_generation_firstblock_taylor_predict_flux.py
```

## 参数详解

| 字段                   | 类型   | 意义                                             | 常用取值    |
|------------------------|--------|--------------------------------------------------|-------------|
| `enable_teacache` | `bool`  |     是否使用teacache方法       | `True`    |
| `residual_diff_threshold`  | `float`  | l1距离的阈值               | `0.25 - 0.8`   |

# BlockDance

## 快速简介

> **BlockDance** 是一种 _training‑free_ 的推理加速方法：通过跳过前面block，直接复用之前step的输出作为之后时间步的step的输入，可为Flux模型带来的速度提升。



## 使用方法
```
python text_to_image_generation_blockdance_flux.py
```

## 参数详解

| 字段                   | 类型   | 意义                                             | 常用取值    |
|------------------------|--------|--------------------------------------------------|-------------|
| `step_start` | `int`  |     复用开始时间步       | `100`    |
| `step_end`  | `int`  | 复用结束时间步               | `900`   |
| `block_step_single`  | `int`  |  单流注意力跳步block的个数   | `26`   |
| `block_step`  | `int`  | 双流注意力跳步block的个数               | `13`   |
| `block_step_N`  | `int`  | 时间步复用范围               | `2 - 4`   |

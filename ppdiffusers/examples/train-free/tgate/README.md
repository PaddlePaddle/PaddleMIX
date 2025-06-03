# T‑GATE

## 快速简介

> **TGATE** 是一种 _training‑free_ 的推理加速方法：在交叉注意力收敛后停止其计算、复用缓存，可为flux带来速度提升。



## 使用方法

- Flux + TGATE
```
python main.py \
--prompt "A cat holding a sign that says hello world" \
--model 'flux' \
--gate_step 25 \
--sp_interval 2 \
--fi_interval 1 \
--warm_up 2 \
--saved_path './generated_tmp/flux/' \
--inference_step 50 \
--seed 42
```



## 参数详解

| 字段                   | 类型   | 意义                                             | 常用取值    |
|------------------------|--------|--------------------------------------------------|-------------|
| `gate_step`            | `int`  | 从第几步开始停止交叉注意力计算                   | `6 – 10`    |
| `num_inference_steps`  | `int`  | 总扩散步数（与原推理保持一致即可）               | `20 – 50`   |
| `reuse_cache`          | `bool` | 是否复用已缓存的注意力（默认 `True`）            | `True`      |
| `enable_self_attn`     | `bool` | 是否对 Self‑Attention 也使用同样的 gating 策略    | `False`     |

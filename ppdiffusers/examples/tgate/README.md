# T‑GATE

## 快速简介

> **TGATE** 是一种 _training‑free_ 的推理加速方法：在交叉注意力收敛后停止其计算、复用缓存，可为 Stable Diffusion / PixArt 等模型带来 **10‑50 %** 的速度提升。



## 使用方法
- SD‑XL + TGATE
```
python main.py \
--prompt 'Astronaut in a jungle, cold color palette, muted colors, detailed, 8k' \
--model 'sdxl' \
--gate_step 10 \
--sp_interval 5 \
--fi_interval 1 \
--warm_up 2 \
--saved_path './generated_tmp/sd_xl/' \
--inference_step 25 \
--seed 42
```

- Pixart‑Alpha + TGATE
```
python main.py \
--prompt 'An alpaca made of colorful building blocks, cyberpunk.' \
--model 'pixart_alpha' \
--gate_step 15 \
--sp_interval 3 \
--fi_interval 1 \
--warm_up 2 \
--saved_path './generated_tmp/pixart_alpha/' \
--inference_step 25 \
```

- LCM‑SDXL + TGATE
```
python main.py \
--prompt 'Self-portrait oil painting, a beautiful cyborg with golden hair, 8k' \
--model 'lcm_sdxl' \
--gate_step 1 \
--sp_interval 1 \
--fi_interval 1 \
--warm_up 0 \
--saved_path './generated_tmp/lcm_sdxl/' \
--inference_step 4 \
```

> 对于 LCM，gate_step 通常设为 1 或 2，inference_step 设为 4。


## 参数详解

| 字段                   | 类型   | 意义                                             | 常用取值    |
|------------------------|--------|--------------------------------------------------|-------------|
| `gate_step`            | `int`  | 从第几步开始停止交叉注意力计算                   | `6 – 10`    |
| `num_inference_steps`  | `int`  | 总扩散步数（与原推理保持一致即可）               | `20 – 50`   |
| `reuse_cache`          | `bool` | 是否复用已缓存的注意力（默认 `True`）            | `True`      |
| `enable_self_attn`     | `bool` | 是否对 Self‑Attention 也使用同样的 gating 策略    | `False`     |

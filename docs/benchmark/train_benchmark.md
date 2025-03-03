
## Train Benchmark

| Model | Stage | Paddle training speed（ips）| Contrast |Pytorch training speed（ips） | Paddle GPU memory uage（G） 
|----|---|---|---|---|---|
| LLaVA1.6 7B            | Pretrain  | 82  | +26%   | 65  | 19/22 |
|                        | SFT       | 52  | +6%    | 49  | 33/49 |
|                        | LoRA      | 56  | +14%   | 49  | 16/17 |
| LLaVA1.6 13B           | Pretrain  | 52  | +18%   | 44  | 33/36 |
|                        | SFT       | 24  | +4%    | 23  | 50/68 |
|                        | LoRA      | 36  | +5%    | 34  | 29/30 |
| Qwen2VL 2B             | SFT       | 41  | +78%   | 23  | - |
| Qwen2VL 7B             | SFT       | 23  | +109%  | 11  | - |
| Stable Diffusion 1.5   | SFT       | 560 | -12%   | 638 | 28/34 |
|                        | LoRA      | 200 | +6%    | 187 | 30/34 |
| Stable Diffusion 3     | SFT       | 34  | 0      | 34  | - |
|                        | LoRA      | 66  | -0.01% | 67  | - |


0.261538462
0.06122449
0.142857143
0.181818182
0.043478261
0.058823529
0.782608696
1.090909091
-0.122257053
0.069518717
0
-0.014925373
---

Notes:
- All models were tested on the H800 (8 * 80G) platform
- For `GPU menory usage`, the table shows `max_memory_allocated/max_memory_reserved`
- Testing config details see blow.

<details>
<summary>Testing config details</summary>

```
# LLaVA and Qwen2VL
N1C8, bf16, O2, stage2, gbz16*8=128; amp_master_grad=True

# Stable Diffusion 1.5
SFT: N1C8, bf16, resolution512, gbz 80
LoRA: N1C8, bf16, resolution512, gbz 96*8

# Stable Diffusion 3 
SFT/LoRA: N1C8, fp16, resolution512, gbz 8
```

</details>


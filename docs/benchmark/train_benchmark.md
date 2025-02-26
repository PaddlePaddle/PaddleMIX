
## Train Benchmark

| Model | Stage | training speed（ips）| GPU memory uage（G）                         |
|----|---|---|---|
| LLaVA1.6 7B            | Pretrain  | 82  | 19/22 |
|                        | SFT       | 52  | 33/49 |
|                        | LoRA      | 56  | 16/17 |
| LLaVA1.6 13B           | Pretrain  | 52  | 33/36 |
|                        | SFT       | 24  | 50/68 |
|                        | LoRA      | 36  | 29/30 |
| Qwen2VL 2B             | SFT       | 41  | - |
| Qwen2VL 7B             | SFT       | 23  | - |
| Stable Diffusion 1.5   | SFT       | 560 | 28/34 |
|                        | LoRA      | 200 | 30/34 |
| Stable Diffusion 3     | SFT       | 34  | - |
|                        | LoRA      | 66  | - |

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


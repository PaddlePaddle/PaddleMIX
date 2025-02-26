
## Inference Benchmark

| Model | Paddle Inference (s/it)  | Note |
|---|---|---|
| LLaVA1.6 7B            | 1.31  | bf16, max token=128
| LLaVA1.6 13B           | 1.65  | bf16, max token=128
| Qwen2VL 2B             | 1.44  | bf16, max token=128
| Qwen2VL 7B             | 1.73  | bf16, max token=128
| Stable Diffusion 1.5   | 0.79  | 512 * 512, 50 steps
| Stable Diffusion 3     | 1.20  | 512 * 512, 50 steps, one card
|                        | 0.87  | 512 * 512, 50 steps, two card


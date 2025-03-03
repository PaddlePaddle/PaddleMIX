
## Inference Benchmark

| Model | Paddle Inference (s/it) | Contrast | torch (s/it) | vllm (s/it) | tensorrt (s/it) | Note |
|---|---|---|---|---|---|---|
| LLaVA1.6 7B            | 1.31 | +51.5% <br>+24.7% | 2.17 | 1.74 | -    | bf16, max token=128
| LLaVA1.6 13B           | 1.65 | +37.0%  | 2.62 | -    | -    | bf16, max token=128
| Qwen2VL 2B             | 1.44 | +38.7% <br>-48.0% | 2.35 | 0.97 | -    | bf16, max token=128
| Qwen2VL 7B             | 1.73 | +60.6% <br>+5.4% | 4.50 | 1.82 | -    | bf16, max token=128
| Stable Diffusion 1.5   | 0.79 | +5.6% | -    | -    | 0.84 | 512 * 512, 50 steps
| Stable Diffusion 3     | 1.20 | -3.4%  | -    | -    | 1.20 | 0.512 * 512, 50 steps, one card

<!-- |                        | 0.87  | -    | -    | 512 * 512, 50 steps, two card -->


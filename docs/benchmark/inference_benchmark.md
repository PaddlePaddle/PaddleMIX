
## Inference Benchmark
<details >
<summary>Fig</summary>

![Figure_1](https://github.com/user-attachments/assets/be63aafa-1528-4b56-8a12-dd766715d08a)

</details>


| Model | Paddle Inference (s/it) | Contrast | torch (s/it) | vllm (s/it) | tensorrt (s/it) | Note |
|---|---|---|---|---|---|---|
| LLaVA1.6 7B            | 1.31 | +51.5% <br>+24.7% | 2.17 | 1.74 | -    | bf16, max token=128
| LLaVA1.6 13B           | 1.65 | +37.0%  | 2.62 | -    | -    | bf16, max token=128
| Qwen2VL 2B             | 1.44 | +38.7% <br>-48.0% | 2.35 | 0.97 | -    | bf16, max token=128
| Qwen2VL 7B             | 1.73 | +60.6% <br>+5.4% | 4.50 | 1.82 | -    | bf16, max token=128
| Stable Diffusion 1.5   | 0.79 | +5.6% | -    | -    | 0.84 | 512 * 512, 50 steps
| Stable Diffusion 3     | 1.20 | -3.4%  | -    | -    | 1.20 | 0.512 * 512, 50 steps, one card

<!-- |                        | 0.87  | -    | -    | 512 * 512, 50 steps, two card -->



<!-- 
```python
import matplotlib.pyplot as plt
import numpy as np

# 数据
models = [
    "LLaVA1.6 7B", 
    "LLaVA1.6 13B", 
    "Qwen2VL 2B", 
    "Qwen2VL 7B", 
    "Stable Diffusion 1.5", 
    "Stable Diffusion 3"
]

paddle_inference = [1.31, 1.65, 1.44, 1.73, 0.79, 1.20]
torch_inference = [2.17, 2.62, 2.35, 4.50, None, None]
vllm_inference = [1.74, None, 0.97, 1.82, None, None]
tensorrt_inference = [0, None, None, None, 0.84, 1.20] # 0 for legend

# contrasts = [
#     ["+51.5%", "+24.7%"],
#     ["+37.0%"],
#     ["+38.7%", "-48.0%"],
#     ["+60.6%", "+5.4%"],
#     ["+5.6%"],
#     ["-3.4%"]
# ]

# 设置图形大小
plt.figure(figsize=(14, 8))

# 设置柱的位置
x_positions = []
x_labels = []
offset = 0

for idx, model in enumerate(models):
    num_bars = 4  # 最多有四种框架
    x = np.arange(num_bars) + offset
    x_positions.append(offset + 1.5)  # 中心位置用于标签
    x_labels.append(model)
    offset += num_bars + 1  # 给不同模型之间增加间隔

    # 绘制柱状图
    plt.bar(x[0], paddle_inference[idx], 0.4, label='Paddle' if idx == 0 else "", color='b')
    if torch_inference[idx] is not None:
        plt.bar(x[1], torch_inference[idx], 0.4, label='Pytorch' if idx == 0 else "", color='r')
    if vllm_inference[idx] is not None:
        plt.bar(x[2], vllm_inference[idx], 0.4, label='vLMM' if idx == 0 else "", color='g')
    if tensorrt_inference[idx] is not None:
        plt.bar(x[3], tensorrt_inference[idx], 0.4, label='TensorRT' if idx == 0 else "", color='y')

    # 在每个柱子上显示速度提升百分比
    # for i, contrast in enumerate(contrasts[idx]):
    #     plt.text(x[i], max(filter(None, [paddle_inference[idx], torch_inference[idx], vllm_inference[idx], tensorrt_inference[idx]])) + 0.05, contrast, ha='center', va='bottom', fontsize=8)
    
    # 在每个柱子上显示数据值和速度提升百分比
    for i, value in enumerate([paddle_inference[idx], torch_inference[idx], vllm_inference[idx], tensorrt_inference[idx]]):
        if value is not None:
            plt.text(x[i], value + 0.05, f'{value:.2f}', ha='center', va='bottom', fontsize=9)

# 添加标签和标题
# plt.xlabel('Model')
plt.ylabel('Inference Time (s/it)')
plt.title('Comparison of Inference Time across Different Frameworks')
plt.xticks(x_positions, x_labels, rotation=30, ha='right')
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()

``` -->
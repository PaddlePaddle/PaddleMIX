
## Train Benchmark

<details >
<summary>Fig</summary>

![Figure_2](https://github.com/user-attachments/assets/5c2fd610-ccef-4ce0-868c-f99bfd900e37)

</details>



| Model | Stage | Paddle training speed（ips）| Contrast |Pytorch training speed（ips） | Paddle GPU memory uage（G） 
|----|---|---|---|---|---|
| LLaVA1.6 7B            | Pretrain  | 82  | +26%   | 65  | 19/22 |
|                        | SFT       | 52  | +6%    | 49  | 33/49 |
|                        | LoRA      | 56  | +14%   | 49  | 16/17 |
| LLaVA1.6 13B           | Pretrain  | 52  | +18%   | 44  | 33/36 |
|                        | SFT       | 24  | +4%    | 23  | 50/68 |
|                        | LoRA      | 36  | +5%    | 34  | 29/30 |
| Qwen2VL 2B             | SFT       | 33  | +43%   | 23  | - |
| Qwen2VL 7B             | SFT       | 13  | +18%   | 11  | - |
| Stable Diffusion 1.5   | SFT       | 560 | -12%   | 638 | 28/34 |
|                        | LoRA      | 200 | +6%    | 187 | 30/34 |
| Stable Diffusion 3     | SFT       | 34  | 0      | 34  | - |
|                        | LoRA      | 66  | -0.01% | 67  | - |

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



<!-- 
```python

import matplotlib.pyplot as plt
import numpy as np

# 数据
model_stages = [
    ("LLaVA1.6 7B", ["Pretrain", "SFT", "LoRA"]),
    ("LLaVA1.6 13B", ["Pretrain", "SFT", "LoRA"]),
    ("Qwen2VL 2B", ["SFT"]),
    ("Qwen2VL 7B", ["SFT"]),
    ("Stable Diffusion 1.5", ["SFT", "LoRA"]),
    ("Stable Diffusion 3", ["SFT", "LoRA"])
]

paddle_speeds = [
    [82, 52, 56],  # LLaVA1.6 7B
    [52, 24, 36],  # LLaVA1.6 13B
    [33],          # Qwen2VL 2B
    [13],          # Qwen2VL 7B
    [560, 200],    # Stable Diffusion 1.5
    [34, 66]       # Stable Diffusion 3
]

pytorch_speeds = [
    [65, 49, 49],  # LLaVA1.6 7B
    [44, 23, 34],  # LLaVA1.6 13B
    [23],          # Qwen2VL 2B
    [11],          # Qwen2VL 7B
    [638, 187],    # Stable Diffusion 1.5
    [34, 67]       # Stable Diffusion 3
]

contrasts = []
for i in range(len(paddle_speeds)):
    contrasts.append([f'{(x-y)/y:.2%}'  for x, y in zip(paddle_speeds[i], pytorch_speeds[i])])
        
# contrasts = [
#     ["+26%", "+6%", "+14%"],
#     ["+18%", "+4%", "+5%"],
#     ["+43%"],
#     ["+18%"],
#     ["-12%", "+6%"],
#     ["0%", "-0.01%"]
# ]

# 设置图形大小
plt.figure(figsize=(14, 8))

# 设置柱的位置
x_positions = []
x_labels = []
offset = 0

for idx, (model, stages) in enumerate(model_stages):
    num_stages = len(stages)
    x = np.arange(num_stages) + offset
    x_positions.extend(x)
    x_labels.extend([f"{model} {stage}" for stage in stages])
    offset += num_stages + 1  # 给不同模型之间增加间隔

    # 绘制柱状图
    plt.bar(x - 0.2, paddle_speeds[idx], 0.4, label='PaddlePaddle' if idx == 0 else "", color='b')
    plt.bar(x + 0.2, pytorch_speeds[idx], 0.4, label='PyTorch' if idx == 0 else "", color='r')

    # 在每个柱子上显示速度提升百分比
    # for i, contrast in enumerate(contrasts[idx]):
    #     plt.text(x[i], max(paddle_speeds[idx][i], pytorch_speeds[idx][i]) + 5, contrast, ha='center', va='bottom', fontsize=8)

    # 在每个柱子上显示数据值和速度提升百分比
    for i, (a, b) in enumerate(zip(paddle_speeds[idx], pytorch_speeds[idx])):
        print(x, a, b)
        plt.text(x[i] - 0.2, a + 0.05, f'{a}', ha='center', va='bottom', fontsize=9)
        plt.text(x[i] + 0.2, b + 0.05, f'{b}', ha='center', va='bottom', fontsize=9)


# 添加标签和标题
# plt.xlabel('Model and Stage')
plt.ylabel('Training Speed (ips)')
plt.title('Comparison of Paddle and PyTorch Training Speeds')
plt.xticks(x_positions, x_labels, rotation=30)
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()

``` -->
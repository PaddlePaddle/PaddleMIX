# PAB

> **PAB（Pyramid Attention Broadcast）** 是一种 **推理加速策略**。它通过在扩散过程中**跳过部分注意力计算并复用缓存**，可带来 ≈25‑35 % 的推理速度提升，同时几乎不降低图像/视频质量。

---

## 1. 工作原理

1. **注意力冗余**：相邻时间步的注意力权重差异呈 *U* 形分布，尤其在交叉注意力层（Cross‑Attn）最小。  
2. **分层跳步**：按「交叉 → 时间 → 空间」重要性递减顺序，给不同注意力层设置 `block_skip_range`/`timestep_skip_range`，冗余高的层跳得更多。  
3. **广播缓存**：跳过的时间步直接复用上一次的注意力输出，无需重新计算 Q/K/V。

> 结果：无需重新训练模型即可加速推理，特别适用于 **DiT 系列视频/图像扩散模型**（如 *CogVideoX*）。

---

## 2. 快速上手
以下示例演示在 CogVideoX‑2B 上启用 PAB（图像模型同理）。
```
import paddle
from ppdiffusers import CogVideoXPipeline, PyramidAttentionBroadcastConfig, apply_pyramid_attention_broadcast
from ppdiffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", paddle_dtype=paddle.bfloat16)

config = PyramidAttentionBroadcastConfig(
    spatial_attention_block_skip_range=2,
    spatial_attention_timestep_skip_range=(100, 800),
    current_timestep_callback=lambda: pipe._current_timestep,
)
apply_pyramid_attention_broadcast(pipe.transformer, config)

prompt = (
    "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
    "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
    "pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
    "casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
    "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
    "atmosphere of this unique musical performance."
)
generator = paddle.Generator().manual_seed(42)
video = pipe(prompt=prompt, guidance_scale=6, num_inference_steps=50, generator=generator).frames[0]
export_to_video(video, "output.mp4", fps=8)
```

## 3. 参数详解
| 字段                                 | 类型              | 作用                         | 常用取值     |
|--------------------------------------|-------------------|------------------------------|--------------|
| `spatial_attention_block_skip_range` | `int`             | 空间自注意力跳步间隔         | `2 ~ 4`      |
| `spatial_attention_timestep_skip_range` | `tuple(int, int)` | 启用 PAB 的时间步区间        | `(100, 800)` |
| `temporal_attention_block_skip_range` | `int`             | 时间自注意力跳步间隔         | `1 ~ 2`      |
| `cross_attention_block_skip_range`   | `int`             | 交叉注意力跳步间隔           | `4 ~ 8`      |
| `current_timestep_callback`          | `Callable`        | 返回当前时间步（回调函数）   | **必填**     |


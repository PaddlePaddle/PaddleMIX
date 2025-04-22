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
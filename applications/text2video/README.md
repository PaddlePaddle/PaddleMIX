### 文本条件的视频生成（Text-to-Video Generation）

```python
from paddlemix.appflow import Appflow
import imageio


prompt = "An astronaut riding a horse."

app = Appflow(app='text_to_video_generation',models=['damo-vilab/text-to-video-ms-1.7b'])
video_frames = app(prompt=prompt,num_inference_steps=25)['result']

imageio.mimsave("text_to_video_generation-synth-result-astronaut_riding_a_horse.gif", video_frames,duration=8)

```

<div align="center">

| Prompt | video |
|:----:|:----:|
| An astronaut riding a horse.|![text_to_video_generation-synth-result-astronaut_riding_a_horse](https://github.com/LokeZhou/PaddleMIX/assets/13300429/21a21062-4ec3-489a-971b-7daa4305106e) |

</div>

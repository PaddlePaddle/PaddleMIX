from ppdiffusers import DiTPipeline, DPMSolverMultistepScheduler, DDIMScheduler
import paddle
from paddlenlp.trainer import set_seed
dtype=paddle.float32
pipe=DiTPipeline.from_pretrained("./DiT_XL_2_256", paddle_dtype=dtype)
#pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

words = ["white shark"]
class_ids = pipe.get_label_ids(words)

set_seed(42)
generator = paddle.Generator().manual_seed(0)
image = pipe(class_labels=class_ids, num_inference_steps=25, generator=generator).images[0]
image.save("white_shark.png")
print(f'\nGPU memory usage: {paddle.device.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB')

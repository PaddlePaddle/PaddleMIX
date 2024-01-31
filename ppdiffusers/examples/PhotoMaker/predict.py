import os
# ignore warning
os.environ["GLOG_minloglevel"] = "2"

import paddle
from ppdiffusers.utils import load_image
from ppdiffusers import EulerDiscreteScheduler
from photomaker import PhotoMakerStableDiffusionXLPipeline

base_model_path = "SG161222/RealVisXL_V3.0"
photomaker_path = "TencentARC/PhotoMaker"
photomaker_ckpt = "photomaker-v1.bin"

### Load base model
pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    base_model_path,  # can change to any base model based on SDXL
    paddle_dtype=paddle.float16,
    use_safetensors=True,
    variant="fp16",
    low_cpu_mem_usage=True
)

### Load PhotoMaker checkpoint
pipe.load_photomaker_adapter(
    photomaker_path,
    weight_name=photomaker_ckpt,
    from_hf_hub=True,
    from_diffusers=True,
    trigger_word="img"  # define the trigger word
)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

pipe.fuse_lora()

### define the input ID images
input_folder_name = './examples/newton_man'
image_basename_list = os.listdir(input_folder_name)
image_path_list = sorted([os.path.join(input_folder_name, basename) for basename in image_basename_list if basename.endswith(".jpg")])

input_id_images = []
for image_path in image_path_list:
    input_id_images.append(load_image(image_path))

# Note that the trigger word `img` must follow the class word for personalization
prompt = "a half-body portrait of a man img wearing the sunglasses in Iron man suit, best quality"
negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth, grayscale"
generator = paddle.Generator().manual_seed(42)
gen_images = pipe(
    prompt=prompt,
    input_id_images=input_id_images,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    num_inference_steps=50,
    start_merge_step=10,
    generator=generator,
).images[0]
gen_images.save('out_photomaker.png')
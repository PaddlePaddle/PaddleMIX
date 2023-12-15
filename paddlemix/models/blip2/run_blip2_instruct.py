import paddle  # # noqa: F401
from PIL import Image

from paddlemix.models.blip2.blip2_opt2_instruct import Blip2OptInstruct
raw_image = Image.open("/home/aistudio/work/PaddleMIX/Confusing-Pictures.jpg").convert("RGB")
model = Blip2OptInstruct(vit_precision="fp32")

from load_model_and_preprocess import load_model_and_preprocess

# loads InstructBLIP model
model, vis_processors, _ = load_model_and_preprocess(model_type="facebook/opt-2.7b", is_eval=True)
# model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True)
# prepare the image
image = vis_processors["eval"](raw_image).unsqueeze(0)
print("-------------------image----------------------")
print(image)

# import numpy as np
# t1 = np.load("image.np.npy")
# raw_image = paddle.to_tensor(t1)

ret = model.generate({"image": image, "prompt": "What is unusual about this image?"})
# ret = model.generate({"image": image})
print("----------------------------------请输出结果ret----------------------------------------")
print(ret)
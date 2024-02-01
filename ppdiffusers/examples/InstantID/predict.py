import paddle
import cv2
import os
import numpy as np
from PIL import Image
from ppdiffusers import ControlNetModel, AutoencoderKL
from ppdiffusers.utils import load_image
from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps


def resize_img(input_image, max_side=1280, min_side=1024, size=None,
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h
                                                                  )], mode)
        w_resize_new = round(ratio * w) // base_pixel_number * base_pixel_number
        h_resize_new = round(ratio * h) // base_pixel_number * base_pixel_number

    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y + h_resize_new, offset_x:offset_x + w_resize_new
            ] = np.array(input_image)
        input_image = Image.fromarray(res)

    return input_image


if __name__ == '__main__':
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    face_adapter = f'./checkpoints/ip-adapter.bin'
    controlnet_path = f'./checkpoints/ControlNetModel'
    controlnet = ControlNetModel.from_pretrained(controlnet_path,
                                                 paddle_dtype=paddle.float16,
                                                 use_safetensors=True,
                                                 from_hf_hub=True,
                                                 from_diffusers=True)
    
    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

    vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae")
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(base_model_path,
                                                    controlnet=controlnet,
                                                    paddle_dtype=paddle.float16,
                                                    variant="fp16",
                                                    low_cpu_mem_usage=True)
    pipe.vae = vae
    pipe.load_ip_adapter_instantid(face_adapter, 
                                   weight_name=os.path.basename("face_adapter"),
                                   from_diffusers=True)
    
    prompt = (
        'analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, masterpiece, best quality'
    )
    n_prompt = (
        '(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured'
    )
    face_image = load_image('./examples/yann-lecun_resize.jpg')
    face_image = resize_img(face_image)
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * x['bbox'][3] - x['bbox'][1])[-1]
    face_emb = face_info['embedding']
    face_kps = draw_kps(face_image, face_info['kps'])
    image = pipe(prompt=prompt, negative_prompt=n_prompt, image_embeds=face_emb, image=face_kps, controlnet_conditioning_scale=0.8,
                 ip_adapter_scale=0.8, num_inference_steps=30, guidance_scale=5).images[0]
    image.save('result.jpg')
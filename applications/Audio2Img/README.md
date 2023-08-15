# Audio To Image

## 1. 应用简介

*****

Generate image from audio(w/ prompt or image) with [ImageBind](https://facebookresearch.github.io/ImageBind/paper)'s unified latent space and stable-diffusion-2-1-unclip.

- No training is need.
- Integration with 🤗  [ppdiffusers](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers).

----

**Support Tasks**

- [Audio To Image]
  - [Audio to Image]
  - [Audio+Text to Image]
  - [Audio+Image to Image]

----

**Update**

[2023/8/15]: 
- [v0.0]: Support fusing audio, text(prompt) and imnage in ImageBind latent space.


## 2. 运行
*****

example: Use audio generate image across modalities (e.g. Image, Text and Audio) with the model of ImageBind and StableUnCLIPImg2ImgPipeline.

```bash
cd applications/Audio2Img

python audio2img_imagebind.py \
--model_name_or_path The dir name of imagebind checkpoint. \
--stable_unclip_model_name_or_path The dir name of StableUnCLIPImg2ImgPipeline pretrained checkpoint. \
--input_audio an audio file.  \
```
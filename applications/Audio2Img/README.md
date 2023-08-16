# Audio To Image

## 1. 应用简介

*****

Generate image from audio(w/ prompt or image) with [ImageBind](https://facebookresearch.github.io/ImageBind/paper)'s unified latent space and stable-diffusion-2-1-unclip.

- No training is need.
- Integration with 🤗  [ppdiffusers](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers).

----

**Support Tasks**

- [Audio To Image](#audio-to-image)
  - [1. 应用简介](#1-应用简介)
  - [2. 运行](#2-运行)
  - [3. 可视化](#3-可视化)
    - [Audio to Image](#audio-to-image-1)
      - [3.1.1 命令](#311-命令)
      - [3.1.2 效果](#312-效果)
    - [Audio+Text to Image](#audiotext-to-image)
      - [3.2.1 命令](#321-命令)
      - [3.2.2 效果](#322-效果)
    - [Audio+Image to Image](#audioimage-to-image)
      - [3.3.1 命令](#331-命令)
      - [3.3.2 效果](#332-效果)

----

**Update**

[2023/8/15]: 
- [v0.0]: Support fusing audio, text(prompt) and imnage in ImageBind latent space.


## 2. 运行
*****

example: Use audio generate image across modalities (e.g. Image, Text and Audio) with the model of ImageBind and StableUnCLIPImg2ImgPipeline.

```python
cd applications/Audio2Img

python audio2img_imagebind.py \
--model_name_or_path The dir name of imagebind checkpoint. \
--stable_unclip_model_name_or_path The dir name of StableUnCLIPImg2ImgPipeline pretrained checkpoint. \
--input_audio an audio file.  \
```

----
## 3. 可视化
----

### Audio to Image
#### 3.1.1 命令

```python
cd applications/Audio2Img

python audio2img_imagebind.py \
--model_name_or_path The dir name of imagebind checkpoint. \
--stable_unclip_model_name_or_path The dir name of StableUnCLIPImg2ImgPipeline pretrained checkpoint. \
--input_audio assets/bird_audio.wav  \
```
#### 3.1.2 效果
![](../Audio2Img/vis_audio2img/audio2img_output_bird.jpg)


### Audio+Text to Image
#### 3.2.1 命令
```python
cd applications/Audio2Img

python audio2img_imagebind.py \
--model_name_or_path The dir name of imagebind checkpoint. \
--stable_unclip_model_name_or_path The dir name of StableUnCLIPImg2ImgPipeline pretrained checkpoint. \
--input_audio assets/bird_audio.wav  \
--input_text 'A photo.' \
```
#### 3.2.2 效果
![图片](../Audio2Img/vis_audio2img/audio_text_to_img_output_bird_a_photo.jpg)


### Audio+Image to Image
#### 3.3.1 命令
```python
cd applications/Audio2Img

python audio2img_imagebind.py \
--model_name_or_path The dir name of imagebind checkpoint. \
--stable_unclip_model_name_or_path The dir name of StableUnCLIPImg2ImgPipeline pretrained checkpoint. \
--input_audio assets/wave.wav \
--input_image assets/dog_image.jpg \
```

#### 3.3.2 效果
![图片](../Audio2Img/vis_audio2img/audio_img_to_img_output_wave_dog.jpg)

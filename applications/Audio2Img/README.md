### 音频生成图像（Audio-to-Image Generation）

#### 1. Application introduction

*****

Generate image from audio(w/ prompt or image) with [ImageBind](https://facebookresearch.github.io/ImageBind/paper)'s unified latent space and stable-diffusion-2-1-unclip.

- No training is need.
- Integration with [ppdiffusers](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers).

----

**Support Tasks**

- [Audio To Image](#audio-to-image)
  - [1. Application Introduction](#1-Application)
  - [2. Run](#2-Run)
  - [3. Visualization](#3-Visualization)
    - [Audio to Image](#audio-to-image-1)
      - [3.1.1 Instruction](#311-Instruction)
      - [3.1.2 Result](#312-Result)
    - [Audio+Text to Image](#audiotext-to-image)
      - [3.2.1 Instruction](#321-Instruction)
      - [3.2.2 Result](#322-Result)
    - [Audio+Image to Image](#audioimage-to-image)
      - [3.3.1 Instruction](#331-Instruction)
      - [3.3.2 Result](#332-Result)

----

**Update**

[2023/8/15]:
- [v0.0]: Support fusing audio, text(prompt) and imnage in ImageBind latent space.


#### 2. Run
*****

example: Use audio generate image across modalities (e.g. Image, Text and Audio) with the model of ImageBind and StableUnCLIPImg2ImgPipeline.

```python
cd applications/Audio2Img

python audio2img_imagebind.py \
--model_name_or_path imagebind-1.2b/ \
--stable_unclip_model_name_or_path stabilityai/stable-diffusion-2-1-unclip \
--input_audio https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/bird_audio.wav \
```

----
#### 3. Visualization
----

#### Audio to Image
#### 3.1.1 Instruction

```python
cd applications/Audio2Img

python audio2img_imagebind.py \
--model_name_or_path imagebind-1.2b/ \
--stable_unclip_model_name_or_path stabilityai/stable-diffusion-2-1-unclip \
--input_audio https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/bird_audio.wav  \
```
#### 3.1.2 Result
|  Input Audio | Output Image |
| --- | --- |
|[bird_audio.wav](https://github.com/luyao-cv/file_download/blob/main/assets/bird_audio.wav)| ![audio2img_output_bird](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/audio2img_output_bird.jpg)  |


#### Audio+Text to Image
#### 3.2.1 Instruction
```python
cd applications/Audio2Img

python audio2img_imagebind.py \
--model_name_or_path imagebind-1.2b/ \
--stable_unclip_model_name_or_path stabilityai/stable-diffusion-2-1-unclip \
--input_audio https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/bird_audio.wav  \
--input_text 'A photo.' \
```
#### 3.2.2 Result
|  Input Audio | Input Text | Output Image |
| --- | --- |  --- |
|[bird_audio.wav](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/bird_audio.wav) | 'A photo.' | ![audio_text_to_img_output_bird_a_photo](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/audio_text_to_img_output_bird_a_photo.jpg)


#### Audio+Image to Image
#### 3.3.1 Instruction
```python
cd applications/Audio2Img

python audio2img_imagebind.py \
--model_name_or_path imagebind-1.2b/ \
--stable_unclip_model_name_or_path stabilityai/stable-diffusion-2-1-unclip \
--input_audio https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/wave.wav \
--input_image https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/dog_image.jpg \
```

#### 3.3.2 Result
|  Input Audio | Input Image | Output Image |
| --- | --- |  --- |
|[wave.wav](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/wave.wav) | ![input_dog_image](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/dog_image.jpg) | ![audio_img_to_img_output_wave_dog](https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/audio_img_to_img_output_wave_dog.jpg)

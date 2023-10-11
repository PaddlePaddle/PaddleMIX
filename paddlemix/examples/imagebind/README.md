# ImageBind

## 1. 模型简介

*****

Paddle implementation of [ImageBind](https://facebookresearch.github.io/ImageBind/paper).

To appear at CVPR 2023 (*Highlighted paper*)

> ImageBind learns a joint embedding across six different modalities - images, text, audio, depth, thermal, and IMU data. It enables novel emergent applications ‘out-of-the-box’ including cross-modal retrieval, composing modalities with arithmetic, cross-modal detection and generation.

## 2. Demo
*****

example: Extract and compare features across modalities (e.g. Image, Text and Audio).
```bash
cd paddlemix/examples/imagebind/

python run_predict.py \
--model_name_or_path imagebind-1.2b/ \
--input_text "A dog." \
--input_image https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/dog_image.jpg \
--input_audio https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/wave.wav \
```

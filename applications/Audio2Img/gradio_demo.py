# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from types import SimpleNamespace

import gradio as gr
import paddle

from paddlemix import ImageBindModel, ImageBindProcessor
from paddlemix.utils.log import logger
from ppdiffusers import StableUnCLIPImg2ImgPipeline

ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    AUDIO="audio",
    THERMAL="thermal",
    DEPTH="depth",
    IMU="imu",
)


class Predictor:
    def __init__(self, model_args):
        self.processor = ImageBindProcessor.from_pretrained(model_args.model_name_or_path)
        self.predictor = ImageBindModel.from_pretrained(model_args.model_name_or_path)
        self.predictor.eval()

    def run(self, inputs):
        with paddle.no_grad():
            embeddings = self.predictor(inputs)
        return embeddings


def model_init(model_args):
    predictor = Predictor(model_args)
    return predictor


def infer(input_image, input_audio, input_text):

    global predictor
    image_pil = input_image

    encoding = predictor.processor(images=image_pil, text="", audios=input_audio, return_tensors="pd")
    inputs = {}

    if image_pil is not None:
        image_processor = encoding["pixel_values"]
        inputs.update({ModalityType.VISION: image_processor})

    if input_audio is not None:
        audio_processor = encoding["audio_values"]
        inputs.update({ModalityType.AUDIO: audio_processor})
    else:
        pass

    embeddings = predictor.run(inputs)
    image_proj_embeds = embeddings[ModalityType.AUDIO]

    if image_pil is not None:
        logger.info("Generate vision embedding: {}".format(embeddings[ModalityType.VISION]))
        image_proj_embeds += embeddings[ModalityType.VISION]

    logger.info("Generate audio embedding: {}".format(embeddings[ModalityType.AUDIO]))

    if input_text is not None:
        prompt = input_text
    else:
        prompt = ""

    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(model_args.stable_unclip_model_name_or_path)
    pipe.set_progress_bar_config(disable=None)
    output = pipe(image_embeds=image_proj_embeds, prompt=prompt)

    return output.images[0]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="imagebind-1.2b/",
        help="Path to pretrained model or model identifier",
    )
    parser.add_argument(
        "--stable_unclip_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-unclip",
        help="Path to pretrained model or model identifier in stable_unclip_model_name_or_path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        choices=["CPU", "GPU", "XPU"],
        help="Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU.",
    )
    return parser.parse_args()


with gr.Blocks() as demo:
    gr.Markdown("音频生成图像（Audio-to-Image Generation）")
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(label="input audio", type="filepath")
            with gr.Tab(label="input text（可选）") as txttab:
                input_text = gr.Textbox(label="input text")
            with gr.Tab(label="input image（可选）") as imgtab:
                input_image = gr.Image(label="input image")
            infer_button = gr.Button("推理")
        output_image = gr.Image(label="result")
        txttab.select(fn=lambda: None, outputs=input_image)
        imgtab.select(fn=lambda: None, outputs=input_text)
        infer_button.click(fn=infer, inputs=[input_image, input_audio, input_text], outputs=[output_image])
if __name__ == "__main__":

    model_args = parse_arguments()
    assert model_args.device in ["CPU", "GPU", "XPU", "NPU"], "device should be CPU, GPU, XPU or NPU"
    predictor = model_init(model_args)

    demo.launch()

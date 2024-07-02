# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import sys

sys.path.append("../")
import asyncio
import random
import time

import gradio as gr
import paddle
from infer import create_pipe, parse_arguments

lock = asyncio.Lock()

args = parse_arguments()
pipe = create_pipe(args)

infer_op = "zero_copy_infer"
infer_op_dict = {
    "vae_encoder": infer_op,
    "vae_decoder": infer_op,
    "text_encoder": infer_op,
    "unet": infer_op,
}


async def predict(init_image, prompt, strength, steps=4, seed=1231231):
    if seed < 0:
        seed = random.randint(0, 2**32)
    if init_image is not None:
        async with lock:
            generator = paddle.Generator().manual_seed(seed)
        last_time = time.time()
        if steps == 1:
            strength = 1.0
        results = pipe.img2img(
            prompt=prompt,
            image=init_image,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=1.0,
            strength=strength,
            width=512,
            height=512,
            parse_prompt_type="lpw",
            infer_op_dict=infer_op_dict,
        )
    else:
        async with lock:
            generator = paddle.Generator().manual_seed(seed)
        last_time = time.time()
        results = pipe.text2img(
            prompt=prompt,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=1.0,
            width=512,
            height=512,
            parse_prompt_type="lpw",
            infer_op_dict=infer_op_dict,
        )
    print(f"Pipe took {time.time() - last_time} seconds")
    return results.images[0]


css = """
#container{
    margin: 0 auto;
    max-width: 80rem;
}
#intro{
    max-width: 100%;
    text-align: center;
    margin: 0 auto;
}
"""
with gr.Blocks(css=css) as demo:
    init_image_state = gr.State()
    with gr.Column(elem_id="container"):
        gr.Markdown(
            """# LCM Image/Text to Image With FastDeploy
            LCM model can generate high quality images in 4 steps, read more on [LCM Post](https://latent-consistency-models.github.io/).
            **Model**: https://huggingface.co/latent-consistency/lcm-lora-sdv1-5
            """,
            elem_id="intro",
        )
        with gr.Row():
            prompt = gr.Textbox(
                placeholder="Insert your prompt here:",
                scale=5,
                container=False,
            )
            generate_bt = gr.Button("Generate", scale=1)
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    sources=["upload", "webcam", "clipboard"],
                    label="Webcam",
                    type="pil",
                )
            with gr.Column():
                image = gr.Image(type="filepath")
                with gr.Accordion("Advanced options", open=False):
                    strength = gr.Slider(
                        label="Strength",
                        value=0.7,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.001,
                    )
                    steps = gr.Slider(label="Steps", value=4, minimum=1, maximum=50, step=1)
                    seed = gr.Slider(
                        randomize=True,
                        minimum=-1,
                        maximum=12013012031030,
                        label="Seed",
                        step=1,
                    )

        inputs = [image_input, prompt, strength, steps, seed]
        generate_bt.click(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        prompt.input(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        steps.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        seed.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        strength.change(fn=predict, inputs=inputs, outputs=image, show_progress=False)
        image_input.change(
            fn=lambda x: x,
            inputs=image_input,
            outputs=init_image_state,
            show_progress=False,
            queue=False,
        )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=8654)

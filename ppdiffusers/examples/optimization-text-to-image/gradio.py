import gradio as gr
import os
image_files = [f'image{i}.gif' for i in range(0, 6)]

for file in image_files:
    if not os.path.exists(file):
        print(f"error")

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Image(value=image_files[0] if os.path.exists(image_files[0]) else None, show_label=False)
            gr.Image(value=image_files[3] if os.path.exists(image_files[3]) else None, show_label=False)
        with gr.Column():
            gr.Image(value=image_files[1] if os.path.exists(image_files[1]) else None, show_label=False)
            gr.Image(value=image_files[4] if os.path.exists(image_files[4]) else None, show_label=False)
        with gr.Column():
            gr.Image(value=image_files[2] if os.path.exists(image_files[2]) else None, show_label=False)
            gr.Image(value=image_files[5] if os.path.exists(image_files[5]) else None, show_label=False)

demo.launch()

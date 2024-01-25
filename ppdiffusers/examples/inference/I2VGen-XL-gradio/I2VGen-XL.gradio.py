import gradio as gr
import paddle
from ppdiffusers import ImgToVideoSDPipeline, VideoToVideoModelscopePipeline
from ppdiffusers.utils import export_to_video, load_image

image_to_video_pipe = ImgToVideoSDPipeline.from_pretrained(
    "Yang-Changhui/img-to-video-paddle", paddle_dtype=paddle.float32
)

video_to_video_pipe = VideoToVideoModelscopePipeline.from_pretrained("Yang-Changhui/video-to-video-paddle")


def upload_file(file):
    return file.name


def image_to_video(image_in):
    if image_in is None:
        raise gr.Error('请上传图片或等待图片上传完成')
    image_in = load_image(image_in)
    output_video_frames = image_to_video_pipe(image_in).frames
    output_video_path = export_to_video(output_video_frames, "img2video_test.mp4")
    print(output_video_path)
    return output_video_path


def video_to_video(video_in, text_in):
    output_video_frames = video_to_video_pipe(prompt=text_in, video_path=video_in).frames
    output_video_path = export_to_video(output_video_frames, "video2video_test.mp4")
    print(output_video_path)
    return output_video_path


with gr.Blocks() as demo:
    gr.Markdown(
        """<center><font size=7>I2VGen-XL</center>
        <left><font size=3>I2VGen-XL可以根据用户输入的静态图像和文本生成目标接近、语义相同的视频，生成的视频具高清(1280 * 720)、宽屏(16:9)、时序连贯、质感好等特点。</left>

        <left><font size=3>I2VGen-XL can generate videos with similar contents and semantics based on user input static images and text. The generated videos have characteristics such as high-definition (1280 * 720), widescreen (16:9), coherent timing, and good texture.</left>
        """
    )
    with gr.Blocks():
        gr.Markdown(
            """<left><font size=3>步骤1：选择合适的图片进行上传 (建议图片比例为1：1)，然后点击“生成视频”，得到满意的视频后进行下一步。”</left>

        <left><font size=3>Step 1:Select the image to upload (it is recommended that the image ratio is 1:1), and then click on “Generate Video” to obtain a generated video before proceeding to the next step.</left>"""
        )
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(label="图片输入", type="filepath", interactive=False, elem_id="image-in", height=300)
                with gr.Row():
                    upload_image = gr.UploadButton("上传图片", file_types=["image"], file_count="single")
                    image_submit = gr.Button("生成视频🎬")
            with gr.Column():
                video_out_1 = gr.Video(label='生成的视频', elem_id='video-out_1', interactive=False, height=300)
        gr.Markdown(
            """<left><font size=3>步骤2：补充对视频内容的英文文本描述，然后点击“生成高分辨率视频”，视频生成大致需要2分钟。”</left>

        <left><font size=3>Step 1:Add the English text description of the video you want to generate, and then click on “Generate high-resolution video”. The video generation will take about 2 minutes..</left>"""
        )
        with gr.Row():
            with gr.Column():
                text_in = gr.Textbox(label="文本描述", lines=2, elem_id="text-in")
                video_submit = gr.Button("生成高分辨率视频🎬")
            with gr.Column():
                paddle.device.cuda.empty_cache()
                video_out_2 = gr.Video(label='生成的视频', elem_id='video-out_2', interactive=False, height=300)
    gr.Markdown("<left><font size=2>注：如果生成的视频无法播放，请尝试升级浏览器或使用chrome浏览器。</left>")

    upload_image.upload(upload_file, upload_image, image_in, queue=False)
    image_submit.click(fn=image_to_video, inputs=[image_in], outputs=[video_out_1])

    video_submit.click(fn=video_to_video, inputs=[video_out_1, text_in], outputs=[video_out_2])

demo.queue(status_update_rate=1, api_open=False).launch(share=False, show_error=True)

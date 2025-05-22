# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import hashlib
import os
import os.path
import sys
import tempfile
import time
from datetime import datetime

import gradio as gr
import numpy as np
import paddle
from PIL import Image
import shutil

# 设置使用的GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 模型配置
model_path = "PaddleMIX/PPDocBee2-3B"
dtype = "bfloat16"  # V100请改成float16
attn_implementation = "flash_attention_2" # V100请改成'eager'

# 全局变量定义
model = None
processor = None

#显卡资源充足可以去掉这个
min_pixels = 256 * 28 * 28  # 最小像素数
max_pixels = 48 * 48 * 28 * 28  # 最大像素数

SERVER_NAME = "localhost"
SERVER_PORR = 8080


def check_and_install_paddlemix():
    try:
        from paddlemix.models.ppdocbee2 import PPDocBee2ForConditionalGeneration

        print("Required PPDocBee2 model successfully installed")
    except ImportError:
        print("Failed to install required PPDocBee2 model even after running the script")
        sys.exit(1)


# 在继续之前检查所需模型
check_and_install_paddlemix()

from paddlemix.models.qwen2_5_vl import MIXQwen2_5_Tokenizer
from paddlemix.models.ppdocbee2 import PPDocBee2ForConditionalGeneration
from paddlemix.processors.qwen2_5_vl_processing import (
    Qwen2_5_VLImageProcessor,
    Qwen2_5_VLProcessor,
    process_vision_info,
)

# 示例使用HTTP链接
EXAMPLES = [
    [
        "维修保养、其他注意事项的注意点中，电池需为什么型号的？",
        "paddlemix/demo_images/shuomingshu_20.png",
    ],
    [
        "产品期限是多久？",
        "paddlemix/demo_images/shuomingshu_39.png",
    ],
]


class ImageCache:
    """图片缓存管理类"""

    def __init__(self):
        """初始化图片缓存"""
        self.temp_dir = tempfile.mkdtemp()
        self.current_image = None
        self.is_example = False  # 标记当前图片是否为示例图片
        print(f"Created temporary directory for image cache: {self.temp_dir}")

    def cleanup_previous(self):
        """清理之前的缓存图片"""
        if self.current_image and os.path.exists(self.current_image) and not self.is_example:
            try:
                os.unlink(self.current_image)
                print(f"Cleaned up previous image: {self.current_image}")
            except Exception as e:
                print(f"Error cleaning up previous image: {e}")

    def cache_image(self, image_path, is_example=False):
        """
        缓存图片并返回缓存路径
        Args:
            image_path: 图片文件路径
            is_example: 是否为示例图片
        Returns:
            缓存后的图片路径
        """
        if not image_path:
            return None

        try:
            # 如果是示例图片且已经在使用中，直接返回
            if is_example and self.current_image == image_path and self.is_example:
                return self.current_image

            # 创建安全的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
            _, ext = os.path.splitext(image_path)
            if not ext:
                ext = ".jpg"  # 默认扩展名
            new_filename = f"image_{timestamp}_{file_hash}{ext}"

            # 在临时目录中创建新路径
            new_path = os.path.join(self.temp_dir, new_filename) if not is_example else image_path

            if not is_example:
                # 处理上传的图片文件
                shutil.copy2(image_path, new_path)
                self.cleanup_previous()

            self.current_image = new_path
            self.is_example = is_example

            return new_path

        except Exception as e:
            print(f"Error caching image: {e}")
            return image_path


# 创建全局图片缓存管理器
image_cache = ImageCache()


def load_model():
    """加载模型并进行内存优化"""
    global model, processor

    if model is None:
        # 加载模型和处理器

        paddle.set_default_dtype(dtype)
        model = PPDocBee2ForConditionalGeneration.from_pretrained(
            model_path, 
            dtype=dtype, 
            attn_implementation=attn_implementation)

        image_processor = Qwen2_5_VLImageProcessor()
        tokenizer = MIXQwen2_5_Tokenizer.from_pretrained(model_path)
        # min_pixels = 256*28*28 # 200704
        # max_pixels = 1280*28*28 # 1003520
        processor = Qwen2_5_VLProcessor(image_processor, tokenizer, min_pixels=min_pixels, max_pixels=max_pixels)

        # 设置为评估模式
        model.eval()
    del tokenizer
    return model, processor


def clear_cache():
    """清理GPU缓存"""
    if paddle.device.cuda.memory_allocated() > 0:
        paddle.device.cuda.empty_cache()
        import gc

        gc.collect()


def multimodal_understanding(image, question, seed=42, top_p=0.95, temperature=0.1):
    """
    多模态理解主函数
    Args:
        image: 输入图片
        question: 问题文本
        seed: 随机种子
        top_p: 采样参数
        temperature: 温度参数
    Yields:
        处理状态和结果
    """
    # 输入验证
    if not image:
        yield "⚠️ 请上传图片后再开始对话。"
        return
    if not question or question.strip() == "":
        yield "⚠️ 请输入您的问题后再开始对话。"
        return

    try:
        start_time = time.time()
        yield "🔄 正在处理您的请求，请稍候..."

        # 检查超时
        if time.time() - start_time > 200:
            yield "⏳ 系统当前用户繁多，请等待10分钟后再次尝试。感谢您的理解！"
            return

        clear_cache()

        # 设置随机种子
        paddle.seed(seed)
        np.random.seed(seed)

        # 处理图片缓存
        is_example = any(image == example[1] for example in EXAMPLES)
        cached_image = image_cache.cache_image(image, is_example=is_example)
        if not cached_image:
            return "图片处理失败，请检查图片格式是否正确。"

        # 构建提示文本
        prompts = question + "\n请用图片中完整出现的内容回答，可以是单词、短语或句子，针对问题回答尽可能详细和完整，并保持格式、单位、符号和标点都与图片中的文字内容完全一致。"

        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": cached_image,
                    },
                    {"type": "text", "text": prompts},
                ],
            }
        ]

        yield "模型正在分析图片内容..."

        # 处理视觉信息
        image_inputs, video_inputs = process_vision_info(messages)
        image_pad_token = "<|vision_start|><|image_pad|><|vision_end|>"
        text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{image_pad_token}{prompts}<|im_end|>\n<|im_start|>assistant\n"

        # 生成回答
        with paddle.no_grad():
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pd"
            )

            yield "正在生成回答..."

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                top_p=top_p,
                temperature=temperature,
                num_beams=1,
                do_sample=True,
                use_cache=True,
            )

            output_text = processor.batch_decode(
                generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        # 清理内存
        del inputs, generated_ids
        clear_cache()

        yield output_text

    except Exception as e:
        error_message = f"处理过程中出现错误: {str(e)}\n请重试或在评论区留下你的问题。"
        return error_message


def process_example(question, image):
    """处理示例图片的包装函数"""
    cached_path = image_cache.cache_image(image, is_example=True)
    return multimodal_understanding(cached_path, question)


def handle_image_upload(image):
    """处理图片上传"""
    if image is None:
        return None
    try:
        cached_path = image_cache.cache_image(image, is_example=False)
        return cached_path
    except Exception as e:
        print(f"Error handling image upload: {e}")
        return None


# model, processor = load_model()
# # image = "/home/aistudio/work/doc-lark/PaddleMIX/paddlemix/demo_images/examples_image1.jpg"
# print(multimodal_understanding(EXAMPLES[1][1],EXAMPLES[1][0]))

# Gradio界面配置
with gr.Blocks() as demo:
    gr.Markdown(
        value="""
    # 🤖 PP-DocBee-V2(3B): Multimodal Document Understanding Demo

    📚 原始模型来自 [PaddleMIX](https://github.com/PaddlePaddle/PaddleMIX)  （🌟 一个基于飞桨PaddlePaddle框架构建的多模态大模型套件）
    """
    )
    with gr.Row():
        image_input = gr.Image(type="filepath", label="📷 Upload Image or Input URL")
        with gr.Column():
            question_input = gr.Textbox(label="💭 Question", placeholder="Enter your question here...")
            und_seed_input = gr.Number(label="🎲 Seed", precision=0, value=42)
            top_p = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.05, label="📊 Top P")
            temperature = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.05, label="🌡️ Temperature")

    image_input.upload(fn=handle_image_upload, inputs=[image_input], outputs=[image_input])

    understanding_button = gr.Button("💬 Chat", variant="primary")
    understanding_output = gr.Textbox(label="🤖 Response", interactive=False)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[question_input, image_input],
        outputs=understanding_output,
        fn=process_example,
        cache_examples=True,
        run_on_click=True,
    )

    # 加载模型
    clear_cache()
    model, processor = load_model()
    clear_cache()

    understanding_button.click(
        fn=multimodal_understanding,
        inputs=[image_input, question_input, und_seed_input, top_p, temperature],
        outputs=understanding_output,
        api_name="chat",
    )

if __name__ == "__main__":
    # 创建队列
    demo.queue()
    demo.launch(server_name=SERVER_NAME, server_port=SERVER_PORR, share=True, ssr_mode=False, max_threads=1)  # 限制并发请求数

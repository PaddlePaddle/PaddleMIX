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

import os
import subprocess

import gradio as gr
from model import ImageChatModel
from ollama import chat

image_chat_model = ImageChatModel()


def start_ollama_service():
    try:
        subprocess.Popen(["ollama", "serve"])
        print("Ollama service started successfully")
        os.system("ollama ls")
    except Exception as e:
        print(f"Error starting Ollama service: {e}")


def analyze_image(image):
    if not image:
        return "请先上传图片"
    prompt = "请描述这个动漫图片，需要1. 推测动漫是哪一部；2. 给出图片的整体风格；3.描述图像中的细节，并推测可能的背景故事。"
    for analysis in image_chat_model.generate_description(image, prompt):
        yield analysis  # 返回中间状态消息
        if "请稍等，正在分析图片..." not in analysis:
            return analysis


def analyze_face(image):
    if not image:
        return "请先上传图片"
    """分析面容特征"""
    image_prompt = "请详细描述此人的性别，面相特征，包括美貌长相、五官、表情、配饰等细节，输出为JSON格式,中文。"
    for analysis in image_chat_model.generate_description(image, image_prompt):
        yield analysis  # 返回中间状态消息
        if "请稍等，正在分析图片..." not in analysis:
            return analysis


def analyze_fortune(
    image, image_analysis, birthday, mbti_type, analysis_type, custom_question, progress=gr.Progress()
):
    """分析运势"""
    if not image:
        return "请先上传照片"
    if not image_analysis:
        return "请先等待图片分析结果"
    # progress(0, desc="正在启动 AI 命理师...")
    yield "分析中..."

    # 生成命理分析
    # progress(0.4, desc="🎯 正在解读命理...")
    prompt = f"""
    你是一位专业的AI命理师，擅长将现代心理学与东方玄学相结合。
    ## 图像分析
    {image_analysis}
    ## 用户信息
    - 生日：{birthday}
    - MBTI：{mbti_type}
    - 分析类型：{analysis_type}
    - 特定问题：{custom_question if custom_question else "无"}

    请根据以上信息进行分析：
    1. 结合性别、面相特征和MBTI给出性格解读
    2. 基于生日和当前时间给出运势预测
    3. 针对用户选择的分析类型给出具体建议
    4. 如果有特定问题，请特别关注相关方面

    注意：保持专业性的同时要适当融入趣味性，最后注明"本结果仅供娱乐"。
    """

    # progress(0.6, desc="✨ 正在生成个性化解读...")
    stream = chat(
        model="deepseek-r1:32b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        stream=True,
    )

    result = ""
    for chunk in stream:
        result += chunk["message"]["content"]
        # progress(0.8, desc="📝 正在润色结果...")
        yield result + "\n\n⌛ 正在生成中，请稍候..."

    # progress(1.0, desc="✅ 生成完成！")
    yield result


def analyze_traditional_texts(image):
    """识别图片中的繁体字"""
    if not image:
        return "请先上传图片"

    prompt = "请识别图片中的繁体字，并转换为简体中文输出。格式要求和原文格式一致。输出简体字。"
    for analysis in image_chat_model.generate_description(image, prompt):
        yield analysis  # 返回中间状态消息
        if "请稍等，正在分析图片..." not in analysis:
            return analysis


def anime_creation(
    image, image_analysis, creation_type, poem_type, story_type, style, custom_prompt, progress=gr.Progress()
):
    """生成创作内容"""
    if not image:
        return "请先上传图片"

    progress(0.2, desc="🎨 正在构思创意...")
    if creation_type == "诗歌类":
        req = f"请创作一首{poem_type}, 需要取诗歌的名字"
    else:
        req = f"请创作{style}风格的{story_type}，需要取章节名"

    prompt = f"""
    你是一个了解动漫，富有才情的作家，能根据图片描述和创作要求进行创作
    ## 图片描述
    {image_analysis}
    ## 创作要求
    1. {req}
    2. 内容上贴合图片描述，创作风格贴合图片的风格，尽可能推断出这个动漫是什么，人物有哪些
    3. 如果有自定义需求：{custom_prompt}，需要满足；没有不需要。
    """
    progress(0.4, desc="✍️ 正在创作中...")
    stream = chat(
        model="deepseek-r1:32b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        stream=True,
    )

    result = ""
    for chunk in stream:
        result += chunk["message"]["content"]
        yield result + "\n\n⌛ 创作火力全开中，请稍候..."

    yield result


def chat_with_texts(message, history, text_content, history_flag=True):
    """与文本内容进行对话

    Args:
        message: 当前用户消息
        history: 对话历史记录
        text_content: 文档内容

    Yields:
        str: 模型响应内容
    """
    # 输入验证
    if not text_content:
        yield "请先上传图片!"
        return

    try:
        # 构建系统提示词
        system_prompt = f"""你是一个专业的文献解读专家。
        ## 文档内容
        {text_content}

        请基于以上文档内容和历史聊天记录回答用户问题。如果问题超出范围，请明确指出。
        """

        # 构建消息历史
        messages = [{"role": "system", "content": system_prompt}]

        # 添加历史对话
        if history_flag and len(history) > 0:
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})

        # 添加当前问题
        messages.append({"role": "user", "content": message})

        # 调用模型进行对话
        stream = chat(model="deepseek-r1:32b", messages=messages, stream=True)

        # 处理响应流
        result = ""
        for chunk in stream:
            if not chunk["message"]["content"]:
                continue

            content = chunk["message"]["content"]
            if content == "<think>":
                result += "🤔思考中..."
                yield result
                continue
            if content == "</think>":
                result += "✨思考完成!"
                yield result
                continue

            result += content
            yield result + "\n\n⌛ 正在生成回答，请稍候..."

        yield result

    except Exception as e:
        yield f"对话出错: {str(e)}"


def setup_events(
    image_input,
    image_analysis,
    creation_type,
    story_group,
    poem_group,
    generate_btn,
    style_type_poem,
    style_type_story,
    style,
    custom_prompt,
    output_text,
):
    """设置UI组件的事件处理"""

    def update_groups(choice):
        if choice == "诗歌类":
            return gr.update(visible=False), gr.update(visible=True)
        else:
            return gr.update(visible=True), gr.update(visible=False)

    # 当图片上传时，自动分析图片
    image_input.change(fn=analyze_image, inputs=[image_input], outputs=[image_analysis])

    # 当创作类型改变时，更新显示的选项组
    creation_type.change(fn=update_groups, inputs=[creation_type], outputs=[story_group, poem_group])


# tabs
def create_anime_creation_tab():
    """创建动漫二创标签页"""
    with gr.Tab("动漫二创"):
        gr.Markdown("# 🎨 高能回忆杀！为你喜欢的动漫画面二创🚀")
        gr.Markdown(
            """
        📖 本项目基于PaddleMIX和DeepSeek-R1实现！[✨PaddleMIX✨](https://github.com/PaddlePaddle/PaddleMIX) 让我们能够开箱即用许多SOTA模型，快来看看如何快速整合 Qwen2.5-VL 和 DeepSeek-R1为我们喜欢的动漫场景进行二创吧～

        💡 **使用方法：** <br>
        1.上传图片 （或点击应用下方Examples）<br>
        2.选择创作类型（诗歌/故事）<br>
        3.输入补充信息（比如是哪一部动漫，角色是哪些，期望的剧情等）<br>
        4.点击"开始创作"

        🖌️ DeepSeek-R1凭借其强大的推理能力能为我们的创作提供更多思路，快来体验一下吧～
        """
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="🖼️ Step 1: 上传动漫海报")
                image_analysis = gr.Textbox(label="图片描述", interactive=False)

                with gr.Group():
                    creation_type = gr.Radio(choices=["诗歌类", "故事类"], label="📝 Step 2: 选择创作类型", value="诗歌类")

                with gr.Group() as poem_group:
                    style_type_poem = gr.Radio(choices=["五言绝句", "七言律诗", "现代诗"], label="✨ Step 3: 选择诗歌类型", value="现代诗")

                with gr.Group(visible=False) as story_group:
                    style_type_story = gr.Radio(choices=["微小说", "剧本大纲", "分镜脚本"], label="✨ Step 3: 选择故事类型", value="微小说")
                    style = gr.Radio(
                        choices=["热血", "治愈", "悬疑", "古风", "科幻", "日常"], label="🎨 Step 4: 选择创作风格", value="治愈"
                    )

            with gr.Column():
                custom_prompt = gr.Textbox(label="💭 Step 4: 创作补充信息（选填）", placeholder="输入额外的创作要求（动漫名称、任务、情节补充）")
                generate_btn = gr.Button("🚀 Step 5: 开始创作")

                output_text = gr.Textbox(label="创作结果", interactive=False)

        # 设置事件处理
        setup_events(
            image_input,
            image_analysis,
            creation_type,
            story_group,
            poem_group,
            generate_btn,
            style_type_poem,
            style_type_story,
            style,
            custom_prompt,
            output_text,
        )

        generate_btn.click(
            fn=anime_creation,
            inputs=[
                image_input,
                image_analysis,
                creation_type,
                style_type_poem,
                style_type_story,
                style,
                custom_prompt,
            ],
            outputs=[output_text],
        )


def create_fortune_tab():
    """创建AI命理师标签页"""
    with gr.Tab("AI命理师"):
        gr.Markdown("# 🔮 AI解命大师")
        gr.Markdown(
            """
        📖 本项目基于PaddleMIX和DeepSeek-R1 实现！[✨PaddleMIX✨](https://github.com/PaddlePaddle/PaddleMIX) 让我们能够开箱即用许多SOTA模型,
        快来体验 Qwen2.5-VL 的图像解析能力和 DeepSeek-R1 的推理能力，为你的人生解密吧～

        💡 **使用方法：** <br>
        1. 上传一张清晰的自拍照（建议半身照）<br>
        2. 填写您的生日和MBTI类型(选填)<br>
        3. 选择想要了解的运势类型(选填)<br>
        4. 可以输入具体想问的问题(选填)<br>
        5. 点击"开始解析"获取个性化解读

        🎯 DeepSeek-R1凭借其强大的推理能力，结合现代心理学与东方玄学，为你提供独特的解读～

        ⚠️ 本功能仅供娱乐，请理性对待分析结果
        """
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="🤳 Step 1: 上传自拍照片")
                image_analysis = gr.Textbox(label="面容分析", interactive=False)
                birthday = gr.Textbox(label="📅 Step 2: 输入生日(选填)", placeholder="格式：YYYY-MM-DD", value="")
                mbti_type = gr.Dropdown(
                    choices=[
                        "无",
                        "INTJ",
                        "INTP",
                        "ENTJ",
                        "ENTP",
                        "INFJ",
                        "INFP",
                        "ENFJ",
                        "ENFP",
                        "ISTJ",
                        "ISFJ",
                        "ESTJ",
                        "ESFJ",
                        "ISTP",
                        "ISFP",
                        "ESTP",
                        "ESFP",
                    ],
                    label="🎭 Step 3: 选择MBTI类型(选填)",
                    value="无",
                )
                analysis_type = gr.Radio(
                    choices=["整体运势", "感情运势", "事业财运", "健康运势"], label="🔮 Step 4: 选择分析类型", value="整体运势"
                )
                custom_question = gr.Textbox(label="❓ Step 5: 输入特定问题(选填)", placeholder="有什么特别想了解的问题吗？")

            with gr.Column():
                generate_btn = gr.Button("✨ Step 6: 开始解析")
                output_text = gr.Textbox(label="创作结果", interactive=True)

        # 设置事件处理
        image_input.change(fn=analyze_face, inputs=[image_input], outputs=[image_analysis])

        generate_btn.click(
            fn=analyze_fortune,
            inputs=[image_input, image_analysis, birthday, mbti_type, analysis_type, custom_question],
            outputs=[output_text],
        )


def create_traditional_qa_tab():
    """创建繁体字识别问答标签页"""
    with gr.Tab("繁体文献问答"):
        gr.Markdown("# 📚 繁体文献智能问答助手")
        gr.Markdown(
            """
        📖 本项目基于PaddleMIX和DeepSeek-R1 实现！[✨PaddleMIX✨](https://github.com/PaddlePaddle/PaddleMIX) 让我们能够开箱即用许多SOTA模型,
        快来体验 Qwen2.5-VL 的图像解析能力和 DeepSeek-R1 的推理能力，快来体验一下吧～

        💡 **功能说明：**
        1. 上传含有繁体字的图片（或从下方选择示例）
        2. 本助手将自动识别繁体字并转换为简体中文
        3. 然后你可以针对文献内容进行提问

        PS: 支持多轮问答
        """
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="📚 Step 1: 上传繁体文献图片")
                text_content = gr.Textbox(label="📝 Step 2: 识别结果（简体中文）", interactive=True, lines=10)

            with gr.Column():
                gr.Markdown("💬 Step 3: 开始提问")
                gr.ChatInterface(
                    chat_with_texts,
                    additional_inputs=[text_content],
                    type="messages",
                    chatbot=gr.Chatbot(height=500),
                    theme="ocean",
                    cache_examples=True,
                )

        # 设置事件处理
        image_input.change(fn=analyze_traditional_texts, inputs=[image_input], outputs=[text_content])


def create_interface():
    """创建主界面"""
    with gr.Blocks(title="🎨 PaddleMIX 多模态大模型创意工坊") as interface:
        gr.Markdown("# 🎨 PaddleMIX 多模态大模型创意工坊")

        with gr.Tabs():
            create_traditional_qa_tab()
            create_anime_creation_tab()
            create_fortune_tab()

    return interface


def main():
    """主函数"""
    interface = create_interface()
    interface.launch(server_name="10.67.188.11", server_port=8101, share=True)


if __name__ == "__main__":
    start_ollama_service()
    main()

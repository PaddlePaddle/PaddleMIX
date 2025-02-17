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

import os
import re
import json


class PPInfinityDocData(object):

    def __init__(self, llm_obj):
        self.llm = llm_obj
        
        
    def generate_chart(self, file_path, llm=None):
        pass

    def generate_doc(self, image_layout_info: str, llm=None):
        template = '''
        ```json
        [
            {
                "from": "human",
                "value": "xxx"
            },
            {
                "from": "gpt",
                "value":  "xxx"
            },
            {
                "from": "human",
                "value": "xxx"
            },
            {
                "from": "gpt",
                "value":  "xxx"
            }
        ]
        ```
        '''
        llm = llm if llm else self.llm

        def generate_qa_pairs(layout_info):            
            try:
                system = "你是一个文档数据视觉问答对话生成系统，主要任务是基于提供的文档图片的ocr版面信息，设计指令和对应答案，以便得到的多模态数据用于模型训练时，模型能够充分学习多模态文档理解能力。"
                question = f'''这里是从一个研报文档图片中提取的ocr版面信息：{layout_info}，请根据其中的内容想象你正在看对应的图片而不是简单的json字符串（文本相关字符是按行识别的，请完整拼接后进行理解），然后结合图片中的信息从专业研报阅读者关心的角度设计指令和回答。
                在生成这些对话时，请确保遵循以下指导原则：
                # 这些指令必须满足以下的要求：
                ```
                1. **指令重点关注文档图片中的信息抽取能力，答案可以直接从图中观察得到。**
                2. **指令和回答必须要与图片高度相关，指令无法脱离图片进行回答。如果指令中提供了太多图片中的信息，使得可以脱离图片对指令进行回答，那是坚决不允许的。**
                3. **指令必须根据图片中的信息生成，且可以结合图片内容得到准确的答案**
                4. **提供的ocr版面信息中的文本字符包含了回答的确切答案，请严格确保回答正确，否则不要生成该指令和回答**
                5. **对于每个版面区域类型（印刷文字、表格、图表、印刷公式、印章），如果文档中包含该类型的信息，那么请生成基于这个类型内容的指令，否则不需要生成**
                6. **指令的回答尽可能简洁且准确，不要重复问题，直接回复答案，同时保证回答直接从图片原文中获取，不要进行总结概括**
                7. **指令与问答中不要出现有关版面结构的信息。**
                8. **指令直接给出问题，不要出现‘请问'、'请回答'、'在文档中'等字眼**
                9.**指令的回答请用图片中完整出现的内容回答，可以是单词、短语或句子，针对问题回答尽可能详细和完整，并保持格式、单位、符号和标点都与图片中的文字内容完全一致。**
                ```
                请生成至少5个中文指令和回答。你需要通过JSON格式提供生成内容，请确保输出中包含```json ```，可以参考下面的样例组织你的输出：{template}，如果文档中的内容全部为英文，则在生成的结果中添加一个键值对："language":"en"。
                '''
                assistant_reply = llm.predict(system+question)
                
            except Exception as e:
                print(f'{e}')
                return
            
            try:
                json_spec_blocks = re.findall(r'```(.*?)```', assistant_reply, re.DOTALL)
                if json_spec_blocks:
                    json_spec = json_spec_blocks[0].strip()
                else:
                    json_spec = assistant_reply
                json_spec = json_spec.lstrip('json')
                qa_pairs = json.loads(json_spec)
                return qa_pairs
            except Exception as e:
                print(f"Error processing {e}")
                return None

        return generate_qa_pairs(image_layout_info)


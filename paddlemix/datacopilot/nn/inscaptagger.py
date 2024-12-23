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
import paddle
from paddlenlp.generation import TextStreamer
from paddlemix.models.llava.language_model.llava_llama import (
    LlavaConfig,
    LlavaLlamaForCausalLM,
)
from paddlemix.models.llava.language_model.tokenizer import LLavaTokenizer
from paddlemix.processors import LlavaProcessor
from paddlemix.models.llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)
from paddlemix.models.llava.conversation import conv_templates
from paddlemix.models.llava.mm_utils import load_image
from paddlenlp.transformers import CLIPImageProcessor

class PPInsCapTagger(object):
    def __init__(self, model_name_or_path, max_new_tokens = 4096, dtype='float16') -> None:
        self.dtype = dtype
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.init_model(model_name_or_path, max_new_tokens, dtype)


    def init_model(self, model_name_or_path, max_new_tokens, dtype):
        tokenizer = LLavaTokenizer.from_pretrained(model_name_or_path)
        model_config = LlavaConfig.from_pretrained(model_name_or_path)
        model = LlavaLlamaForCausalLM.from_pretrained(model_name_or_path, dtype=dtype)
        model.eval()
        name_or_path = (os.path.join(model_name_or_path, "processor", "eval"))
        image_processor = CLIPImageProcessor.from_pretrained(name_or_path)
        processor = LlavaProcessor(
            image_processor, 
            tokenizer,
            max_length=max_new_tokens, 
            image_aspect_ratio=model_config.image_aspect_ratio
            )
        
        model.resize_token_embeddings(len(tokenizer))
        vision_tower = model.get_vision_tower()

        vision_tower.load_model()

        self.model = model
        self.model_config = model_config
        self.processor = processor
        self.tokenizer = tokenizer


    def __call__(self, item):
        return self.inference(item)
        

    def inference(self, item):

        model, model_config, processor, tokenizer = self.model, self.model_config, self.processor, self.tokenizer

        image_file = item["image"]

        conversations = item['conversations']

        conversations = [''.join(sublist) for sublist in conversations]
        instructions = '\n\n'.join(conversations)

        instructions = 'Label this piece of data based on the image and the following conversations:/n/n' +  instructions.replace("\n<image>", "").replace("<image>\n", "")

        n = self.max_new_tokens - 1
        if len(instructions) >= n:
            instructions = instructions[:n-1]

        temperature = 0.0
        
        conv = conv_templates['llava_v1'].copy()

        first_message = True
        inp = instructions
        
        if image_file is not None and first_message:
            if model_config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
            conv.append_message(conv.roles[0], inp)
            first_message = False
        else:
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        record = {"image": image_file, "conversations": prompt}
        image_size = load_image(image_file).size
        data_dict = processor(record=record, image_aspect_ratio=model_config.image_aspect_ratio)

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        try:
            with paddle.no_grad():
                output_ids = model.generate(
                    input_ids=data_dict["input_ids"],
                    images=paddle.cast(data_dict["images"], self.dtype),
                    image_sizes=[image_size],
                    decode_strategy="sampling" if temperature > 0 else "greedy_search",
                    temperature=temperature,
                    max_new_tokens=130,
                    streamer=streamer,
                    use_cache=True,
                )

            outputs = tokenizer.decode(output_ids[0][0]).strip()

            out_item = {
                'image':item["image"],
                'conversations':item['conversations'],
                'tag':outputs[:-4]
            }
            # tag = outputs[:-4]
        except:
            # tag = None
            print(item)
            out_item = {
                'image':item["image"],
                'conversations':item['conversations'],
                'tag':None
            }

        return out_item



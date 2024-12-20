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

import argparse
import copy

import paddle
from paddlenlp.transformers import Qwen2Tokenizer
from PIL import Image

from paddlemix.models.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from paddlemix.models.llava.conversation import conv_templates
from paddlemix.models.llava.language_model.llava_qwen import LlavaQwenForCausalLM
from paddlemix.models.llava.mm_utils import process_images
from paddlemix.models.llava.multimodal_encoder.siglip_encoder import (
    SigLipImageProcessor,
)
from paddlemix.models.llava.train_utils import tokenizer_image_token
from paddlemix.utils.log import logger

def main(args):
    compute_dtype = "float16" if args.fp16 else "bfloat16"
    if "npu" in paddle.get_device():
        is_bfloat16_supported = True
    else:
        is_bfloat16_supported = paddle.amp.is_bfloat16_supported()
    if compute_dtype == "bfloat16" and not is_bfloat16_supported:
        logger.warning("bfloat16 is not supported on your device,change to float32")
        compute_dtype = "float32"

    logger.info(f"compute_dtype: {compute_dtype}")

    model = LlavaQwenForCausalLM.from_pretrained(args.model_path, dtype=compute_dtype).eval()
    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    image_processor = SigLipImageProcessor()

    image = Image.open(args.image_file)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.cast(compute_dtype) for _image in image_tensor]

    question = DEFAULT_IMAGE_TOKEN + "\n" + args.critic_prompt
    conv = copy.deepcopy(conv_templates[args.conv_mode])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pd").unsqueeze(0)
    image_sizes = [image.size]

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    text_outputs = tokenizer.batch_decode(cont[0], skip_special_tokens=True)
    print("output:\n", text_outputs[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="lmms-lab/llava-critic-7b")
    parser.add_argument("--critic_prompt", type=str, default="Given an image and a corresponding question, please serve as an unbiased and fair judge to evaluate the quality of the answers provided by a Large Multimodal Model (LMM). Determine which answer is better and explain your reasoning with specific details. Your task is provided as follows:\nQuestion: [What this image presents?]\nThe first response: [The image is a black and white sketch of a line that appears to be in the shape of a cross. The line is a simple and straightforward representation of the cross shape, with two straight lines intersecting at a point.]\nThe second response: [This is a handwritten number seven.]\nASSISTANT:\n")
    parser.add_argument("--image_file", type=str, default="paddlemix/demo_images/critic_img_seven.png")
    parser.add_argument("--conv_mode", type=str, default="qwen_1_5")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)

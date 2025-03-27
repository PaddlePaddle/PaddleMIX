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
import os

import paddle
from paddlenlp.generation import TextStreamer
from paddlenlp.transformers import CLIPImageProcessor

from paddlemix.models.llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)
from paddlemix.models.llava.conversation import conv_templates
from paddlemix.models.llava.language_model.llava_llama import (
    LlavaConfig,
    LlavaLlamaForCausalLM,
)
from paddlemix.models.llava.language_model.tokenizer import LLavaTokenizer
from paddlemix.models.llava.mm_utils import get_model_name_from_path, load_image
from paddlemix.processors import LlavaProcessor
from paddlemix.utils.log import logger


def main(args):
    paddle.seed(seed=0)
    compute_dtype = "float16" if args.fp16 else "bfloat16"
    if "npu" in paddle.get_device():
        is_bfloat16_supported = True
    else:
        is_bfloat16_supported = paddle.amp.is_bfloat16_supported()
    if compute_dtype == "bfloat16" and not is_bfloat16_supported:
        logger.warning("bfloat16 is not supported on your device,change to float32")
        compute_dtype = "float32"

    model_name = get_model_name_from_path(args.model_path)

    model_name_or_path = args.model_path
    tokenizer = LLavaTokenizer.from_pretrained(model_name_or_path)
    model_config = LlavaConfig.from_pretrained(model_name_or_path)
    model = LlavaLlamaForCausalLM.from_pretrained(model_name_or_path, dtype=compute_dtype)
    model.eval()
    name_or_path = os.path.join(model_name_or_path, "processor", "eval")
    image_processor = CLIPImageProcessor.from_pretrained(name_or_path)
    processor = LlavaProcessor(
        image_processor, tokenizer, max_length=args.max_new_tokens, image_aspect_ratio=model_config.image_aspect_ratio
    )

    model.resize_token_embeddings(len(tokenizer))
    vision_tower = model.get_vision_tower()

    vision_tower.load_model()

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode
    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = "user", "assistant"
    else:
        roles = conv.roles

    first_message = True
    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break
        print(f"{roles[1]}: ", end="")
        if args.image_file is not None and first_message:
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
        record = {"image": args.image_file, "conversations": prompt}
        image_size = load_image(args.image_file).size
        data_dict = processor(record=record, image_aspect_ratio=model_config.image_aspect_ratio)

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with paddle.no_grad():
            output_ids = model.generate(
                input_ids=data_dict["input_ids"],
                images=paddle.cast(data_dict["images"], compute_dtype),
                image_sizes=[image_size],
                decode_strategy="sampling" if args.temperature > 0 else "greedy_search",
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
            )

        outputs = tokenizer.decode(output_ids[0][0]).strip()
        conv.messages[-1][-1] = outputs
        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.6-vicuna-7b")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)

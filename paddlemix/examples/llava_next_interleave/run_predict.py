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

import paddle
from paddlenlp.generation import TextStreamer
from paddlenlp.transformers import Qwen2Tokenizer

from paddlemix.models.llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from paddlemix.models.llava.conversation import conv_templates
from paddlemix.models.llava.language_model.llava_qwen import (
    LlavaQwenConfig,
    LlavaQwenForCausalLM,
)
from paddlemix.models.llava.mm_utils import (
    get_model_name_from_path,
    is_valid_video_filename,
    load_image,
    sample_frames,
    tokenizer_image_token,
)
from paddlemix.utils.log import logger


def main(args):
    paddle.seed(seed=0)
    compute_dtype = "float16" if args.fp16 else "bfloat16"
    if compute_dtype == "bfloat16" and not paddle.amp.is_bfloat16_supported():
        logger.warning("bfloat16 is not supported on your device,change to float32")
        compute_dtype = "float32"

    model_name = get_model_name_from_path(args.model_path)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    # TO DO: add image token to tokenizer paddle 和 torch的对不齐，要手动自己设置
    tokenizer.added_tokens_decoder = {151643: "<|endoftext|>", 151644: "<|im_start|>", 151645: "<|im_end|>"}
    tokenizer.added_tokens_encoder = {"<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645}

    model_config = LlavaQwenConfig.from_pretrained(args.model_path)
    model = LlavaQwenForCausalLM.from_pretrained(args.model_path, dtype=compute_dtype)
    model.eval()

    model.resize_token_embeddings(len(tokenizer))
    vision_tower = model.get_vision_tower()

    vision_tower.load_model()

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    elif "qwen" in model_name.lower():
        conv_mode = "qwen_1_5"
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

    first_message = True

    num_new_images = 0
    image_list = []
    for f in args.image_file:
        if is_valid_video_filename(f):
            num_new_images += args.num_frames
            image_list += sample_frames(f, args.num_frames)
        else:
            num_new_images += 1
            image_list.append(load_image(f))

    image_tensor = [
        paddle.to_tensor(
            vision_tower.image_processor.preprocess(f, return_tensors="pd")["pixel_values"][0], dtype=compute_dtype
        ).cuda()
        for f in image_list
    ]

    image_tensor = paddle.stack(image_tensor)

    inp = args.question
    if args.image_file is not None and first_message:
        if model_config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN * num_new_images + DEFAULT_IM_END_TOKEN + "\n" + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN * num_new_images + "\n" + inp
        conv.append_message(conv.roles[0], inp)
        first_message = False
    else:
        conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_size = load_image(args.image_file[0]).size
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pd").unsqueeze(0)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with paddle.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            decode_strategy="sampling" if args.temperature > 0 else "greedy_search",
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            use_cache=True,
            do_sample=True,
        )

    outputs = tokenizer.decode(output_ids[0][0]).strip().split("</s>")[0]
    print("outputs:\n", outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="lmms-lab/llava-next-interleave-qwen-0.5b")
    parser.add_argument("--question", type=str, default="What is shown in this image?")
    parser.add_argument(
        "--image-file", type=str, nargs="+", required=True, help="Path to an image file or a list of image files."
    )
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_frames", type=int, default=16)
    args = parser.parse_args()
    main(args)

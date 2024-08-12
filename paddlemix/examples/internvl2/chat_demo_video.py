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
import paddle.vision.transforms as T
from PIL import Image
import numpy as np
from decord import VideoReader, cpu
from paddlemix.models.internvl2.internlm2 import InternLM2Tokenizer
from paddlenlp.transformers import AutoTokenizer, Qwen2Tokenizer, LlamaTokenizer, Llama3Tokenizer

from paddlemix.models.internvl2.internvl_chat import InternVLChatModel

paddle.set_grad_enabled(False)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        # T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation='bicubic'),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = paddle.stack(pixel_values)
    return pixel_values


# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = paddle.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = paddle.concat(pixel_values_list)
    return pixel_values, num_patches_list


def load_tokenizer(model_size, model_path):
    if model_size in ['1B']:
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        # TODO:
        tokenizer.added_tokens_encoder =  {'<|endoftext|>': 151643, '<|im_start|>': 151644, '<|im_end|>': 151645, '<img>': 151646, '</img>': 151647, '<IMG_CONTEXT>': 151648, '<quad>': 151649, '</quad>': 151650, '<ref>': 151651, '</ref>': 151652, '<box>': 151653, '</box>': 151654}
        tokenizer.added_tokens_decoder = {v: k for k, v in tokenizer.added_tokens_encoder.items()}

    elif model_size in ['2B', '8B', '26B']:
        tokenizer = InternLM2Tokenizer.from_pretrained(model_path)
        # TODO:
        tokenizer.added_tokens_encoder = {'<unk>': 0, '<s>': 1, '</s>': 2, '<|plugin|>': 92538, '<|interpreter|>': 92539, '<|action_end|>': 92540, '<|action_start|>': 92541, '<|im_end|>': 92542, '<|im_start|>': 92543, '<img>': 92544, '</img>': 92545, '<IMG_CONTEXT>': 92546, '<quad>': 92547, '</quad>': 92548, '<ref>': 92549, '</ref>': 92550, '<box>': 92551, '</box>': 92552}
        tokenizer.added_tokens_decoder = {v: k for k, v in tokenizer.added_tokens_encoder.items()}

    elif model_size in ['4B']:
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        # TODO:
        tokenizer.added_tokens_encoder = {'<unk>': 0, '<s>': 1, '</s>': 2, '<|endoftext|>': 32000, '<|assistant|>': 32001, '<|placeholder1|>': 32002, '<|placeholder2|>': 32003, '<|placeholder3|>': 32004, '<|placeholder4|>': 32005, '<|system|>': 32006, '<|end|>': 32007, '<|placeholder5|>': 32008, '<|placeholder6|>': 32009, '<|user|>': 32010, '<img>': 32011, '</img>': 32012, '<IMG_CONTEXT>': 32013, '<quad>': 32014, '</quad>': 32015, '<ref>': 32016, '</ref>': 32017, '<box>': 32018, '</box>': 32019}
        tokenizer.added_tokens_decoder = {v: k for k, v in tokenizer.added_tokens_encoder.items()}

    elif model_size in ['40B']:
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        # TODO:
        tokenizer.added_tokens_encoder = {'<unk>': 0, '<|startoftext|>': 1, '<|endoftext|>': 2, '<|im_start|>': 6, '<|im_end|>': 7, '<img>': 68, '</img>': 70, '<IMG_CONTEXT>': 64000, '<quad>': 64001, '</quad>': 64002, '<ref>': 64003, '</ref>': 64004, '<box>': 64005, '</box>': 64006}
        tokenizer.added_tokens_decoder = {v: k for k, v in tokenizer.added_tokens_encoder.items()}

    elif model_size in ['76B']:
        tokenizer = Llama3Tokenizer.from_pretrained(model_path)
        # TODO:
        tokenizer.added_tokens_encoder = {'<img>': 128256, '</img>': 128257, '<IMG_CONTEXT>': 128258, '<quad>': 128259, '</quad>': 128260, '<ref>': 128261, '</ref>': 128262, '<box>': 128263, '</box>': 128264}
        tokenizer.added_tokens_decoder = {v: k for k, v in tokenizer.added_tokens_encoder.items()}

    else:
        raise ValueError

    return tokenizer


def main(args):
    if args.video_path is not None and args.video_path != 'None':
        pixel_values, num_patches_list = load_video(args.video_path, num_segments=8, max_num=1)
        pixel_values = pixel_values.to(paddle.bfloat16)
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        args.text = video_prefix + args.text
        # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}

    else:
        pixel_values = None

    # init model and tokenizer
    MODEL_PATH = args.model_name_or_path
    model_size = MODEL_PATH.split('-')[-1]
    print(f'model size: {model_size}')
    tokenizer = load_tokenizer(model_size, MODEL_PATH)
    print('tokenizer:\n', tokenizer)
    print('len(tokenizer): ', len(tokenizer))

    model = InternVLChatModel.from_pretrained(MODEL_PATH).eval()

    generation_config = dict(max_new_tokens=1024, do_sample=False)

    with paddle.no_grad():
        # video multi-round conversation (视频多轮对话)
        response, history = model.chat(tokenizer, pixel_values, args.text, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=True)
        print(f'User: {args.text}\nAssistant: {response}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="OpenGVLab/InternVL2-8B",
        help="pretrained ckpt and tokenizer",
    )
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--text", type=str, default='Please describe the video shortly.', required=True)
    args = parser.parse_args()
    main(args)

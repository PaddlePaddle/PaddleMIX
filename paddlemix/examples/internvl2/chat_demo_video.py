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

import numpy as np
import paddle
import paddle.vision.transforms as T
from decord import VideoReader, cpu
from paddlenlp.transformers import Llama3Tokenizer, LlamaTokenizer, Qwen2Tokenizer
from PIL import Image

from paddlemix.datasets.internvl_dataset import dynamic_preprocess
from paddlemix.models.internvl2.internlm2 import InternLM2Tokenizer
from paddlemix.models.internvl2.internvl_chat import InternVLChatModel
from paddlemix.models.qwen2_vl import MIXQwen2Tokenizer

paddle.set_grad_enabled(False)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def check_dtype_compatibility():
    """
    检查当前环境下可用的数据类型
    返回最优的可用数据类型
    """
    if not paddle.is_compiled_with_cuda():
        print("CUDA not available, falling back to float32")
        return paddle.float32

    # 获取GPU计算能力
    gpu_arch = paddle.device.cuda.get_device_capability()
    if gpu_arch is None:
        print("Unable to determine GPU architecture, falling back to float32")
        return paddle.float32

    major, minor = gpu_arch
    compute_capability = major + minor / 10
    print(f"GPU compute capability: {compute_capability}")

    try:
        # 测试bfloat16兼容性
        if compute_capability >= 8.0:  # Ampere及更新架构
            test_tensor = paddle.zeros([2, 2], dtype="bfloat16")
            test_op = paddle.matmul(test_tensor, test_tensor)
            print("bfloat16 is supported and working")
            return paddle.bfloat16
    except Exception as e:
        print(f"bfloat16 test failed: {str(e)}")

    try:
        # 测试float16兼容性
        if compute_capability >= 5.3:  # Maxwell及更新架构
            test_tensor = paddle.zeros([2, 2], dtype="float16")
            test_op = paddle.matmul(test_tensor, test_tensor)
            print("float16 is supported and working")
            return paddle.float16
    except Exception as e:
        print(f"float16 test failed: {str(e)}")

    print("Falling back to float32 due to compatibility issues")
    return paddle.float32


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            # T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation="bicubic"),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
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
    frame_indices = np.array(
        [int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)]
    )
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = paddle.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = paddle.concat(pixel_values_list)
    return pixel_values, num_patches_list


def load_tokenizer(model_path):
    import re

    match = re.search(r"\d+B", model_path)
    model_v2_5 = "InternVL2_5" in model_path
    model_v3 = "InternVL3" in model_path
    if match:
        model_size = match.group()
    else:
        model_size = "2B"

    if not model_v2_5 and not model_v3:
        # InternVL2，暂不支持4B的phi3
        if model_size in ["1B"]:
            tokenizer = MIXQwen2Tokenizer.from_pretrained(model_path)
        elif model_size in ["2B", "8B", "26B"]:
            tokenizer = InternLM2Tokenizer.from_pretrained(model_path)
        elif model_size in ["40B"]:
            tokenizer = LlamaTokenizer.from_pretrained(model_path)
        elif model_size in ["76B"]:
            tokenizer = Llama3Tokenizer.from_pretrained(model_path)
        else:
            raise ValueError
        return tokenizer

    if model_v2_5:
        if model_size in ["1B", "4B", "38B", "78B"]:
            tokenizer = MIXQwen2Tokenizer.from_pretrained(model_path)
        elif model_size in ["2B", "8B", "26B"]:
            tokenizer = InternLM2Tokenizer.from_pretrained(model_path)
        else:
            raise ValueError
        return tokenizer

    if model_v3:
        if model_size in ["1B", "2B", "8B", "14B", "38B", "78B"]:
            tokenizer = MIXQwen2Tokenizer.from_pretrained(model_path)
        elif model_size in ["9B"]:
            tokenizer = InternLM2Tokenizer.from_pretrained(model_path)
        else:
            raise ValueError
        return tokenizer


def main(args):
    if args.video_path is not None and args.video_path != "None":
        pixel_values, num_patches_list = load_video(args.video_path, num_segments=8, max_num=1)
        pixel_values = pixel_values.to(paddle.bfloat16)
        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
        args.text = video_prefix + args.text
        # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}

    else:
        pixel_values = None

    # init model and tokenizer
    MODEL_PATH = args.model_name_or_path
    tokenizer = load_tokenizer(MODEL_PATH)
    print("tokenizer:\n", tokenizer)
    print("len(tokenizer): ", len(tokenizer))

    model = InternVLChatModel.from_pretrained(MODEL_PATH, dtype=args.dtype).eval()
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    with paddle.no_grad():
        # video multi-round conversation (视频多轮对话)
        response, history = model.chat(
            tokenizer,
            pixel_values,
            args.text,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True,
        )
        print(f"User: {args.text}\nAssistant: {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="OpenGVLab/InternVL2-8B",
        help="pretrained ckpt and tokenizer",
    )
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--text", type=str, default="Please describe the video shortly.", required=True)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"], help="Model dtype"
    )
    args = parser.parse_args()

    if args.dtype == "bfloat16":
        args.dtype = paddle.bfloat16
    elif args.dtype == "float16":
        args.dtype = paddle.float16
    else:
        args.dtype = paddle.float32

    # 检查环境支持的dtype并设置
    available_dtype = check_dtype_compatibility()

    # 如果用户指定了dtype，尝试使用用户指定的类型
    if args.dtype == "bfloat16":
        desired_dtype = paddle.bfloat16
    elif args.dtype == "float16":
        desired_dtype = paddle.float16
    else:
        desired_dtype = paddle.float32

    # 如果用户指定的dtype不可用，使用检测到的可用dtype
    if desired_dtype != available_dtype:
        print(f"Warning: Requested dtype {args.dtype} is not available, using {available_dtype}")
        args.dtype = available_dtype
    else:
        args.dtype = desired_dtype

    print(f"Using dtype: {args.dtype}")

    main(args)

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["FLAGS_use_cuda_managed_memory"] = "true"
import requests
from PIL import Image

from paddlemix import MiniGPT4ForConditionalGeneration, MiniGPT4Processor


def predict(args):
    # load MiniGPT4 moel and processor
    model = MiniGPT4ForConditionalGeneration.from_pretrained(args.pretrained_name_or_path)
    model.eval()
    processor = MiniGPT4Processor.from_pretrained(args.pretrained_name_or_path)
    print("load processor and model done!")

    # prepare model inputs for MiniGPT4
    if args.image_path.startswith("http"):
        image = Image.open(requests.get(args.image_path, stream=True).raw)
    else:
        image = Image.open(args.image_path)

    text = "describe this image"
    prompt = "Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.###Human: <Img><ImageHere></Img> <TextHere>###Assistant:"
    inputs = processor([image], text, prompt)

    # generate with MiniGPT4
    generate_kwargs = {
        "max_length": args.max_length,
        "num_beams": args.num_beams,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "temperature": args.temperature,
        "decode_strategy": args.decode_strategy,
        "eos_token_id": [[835], [2277, 29937]],
    }
    outputs = model.generate(**inputs, **generate_kwargs)
    msg = processor.batch_decode(outputs[0])
    print("Inference result: ", msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_name_or_path",
        default="your minigpt4 path",
        type=str,
        help="The dir name of minigpt4 checkpoint.",
    )
    parser.add_argument(
        "--image_path",
        default="https://paddlenlp.bj.bcebos.com/data/images/mugs.png",
        type=str,
        help="The image path, you can input a url or a local path.",
    )
    parser.add_argument(
        "--decode_strategy",
        default="greedy_search",
        type=str,
        help="The decoding strategy in generation. Currently, there are three decoding strategies supported: greedy_search, sampling and beam_search. Default to greedy_search.",
    )
    parser.add_argument(
        "--max_length",
        default=300,
        type=int,
        help="The maximum length of the sequence to be generated. Default to 300.",
    )
    parser.add_argument(
        "--num_beams", default=1, type=int, help="The number of beams in the beam_search strategy. Default to 1."
    )
    parser.add_argument(
        "--top_p",
        default=1.0,
        type=float,
        help="The cumulative probability for top-p-filtering in the sampling strategy. The value should satisfy :math:`0 <= top\_p < 1`. Default to 1.0, which means no effect.",
    )
    parser.add_argument(
        "--top_k",
        default=0,
        type=int,
        help="The number of highest probability tokens to keep for top-k-filtering in the sampling strategy. Default to 0, which means no effect.",
    )
    parser.add_argument(
        "--repetition_penalty",
        default=1.0,
        type=float,
        help="The parameter for repetition penalty. 1.0 means no penalty. See `this paper <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details. Defaults to 0.",
    )
    parser.add_argument(
        "--length_penalty",
        default=0.0,
        type=float,
        help="The exponential penalty to the sequence length in the beam_search strategy. The larger this param is, the more that the model would generate shorter sequences. Default to 0.0, which means no penalty.",
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        help="The value used to module the next token probabilities in the sampling strategy. Default to 1.0, which means no effect.",
    )

    args = parser.parse_args()

    predict(args)

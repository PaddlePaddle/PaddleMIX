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
from typing import List, Union

import paddle
from paddlenlp.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
)
from PIL import Image

from paddlemix.models.aria.model import AriaForConditionalGeneration
from paddlemix.processors import AriaProcessor

set_dtype = "bfloat16"
paddle.set_default_dtype(set_dtype)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Aria Inference Script")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="bfloat16",
        help="DType to use in inference.",
    )
    parser.add_argument("--base_model_path", required=True, help="Path to the base model")
    parser.add_argument("--tokenizer_path", required=True, help="Path to the tokenizer")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--prompt", required=True, help="Text prompt for the model")
    parser.add_argument(
        "--max_image_size",
        type=int,
        help="Maximum size of the image to be processed",
        default=980,
    )
    parser.add_argument(
        "--split_image",
        help="Whether to split the image into patches",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def load_model(base_model_path, args):
    model = AriaForConditionalGeneration.from_pretrained(
        base_model_path, dtype=args.dtype, ignore_mismatched_sizes=True
    )
    model = model.astype(dtype=args.dtype)

    return model


def prepare_input(image_path, prompt, processor: AriaProcessor, max_image_size, split_image):
    image = Image.open(image_path)

    text = "<|im_start|>user\n<fim_prefix><|img|><fim_suffix>" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    print(text)
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pd",
        max_image_size=max_image_size,
        split_image=split_image,
    )
    return inputs


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_token_ids):
        """
        Args:
            stop_token_ids (List[int]): 用于停止生成的token ids列表
            input_ids (paddle.Tensor): 输入序列的token ids
        """
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(self, output_ids: paddle.Tensor, scores: paddle.Tensor, **kwargs) -> bool:
        # 检查最后生成的token是否在停止token列表中
        last_token = output_ids[0, -1].item()
        if last_token in self.stop_token_ids:
            return True

        return False


class EosTokenCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the "end-of-sequence" token is generated.
    By default, it uses the `model.generation_config.eos_token_id`.

    Args:
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
    """

    def __init__(self, eos_token_id: Union[int, List[int], paddle.Tensor]):
        if not isinstance(eos_token_id, paddle.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = paddle.to_tensor(eos_token_id)
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor, **kwargs) -> paddle.Tensor:
        self.eos_token_id = self.eos_token_id.to(input_ids.place)
        # is_done = isin_mps_friendly(input_ids[:, -1], self.eos_token_id)
        is_done = paddle.isin(input_ids[:, -1], self.eos_token_id)
        return is_done


def inference(
    image_path,
    prompt,
    model: AriaForConditionalGeneration,
    processor: AriaProcessor,
    max_image_size,
    split_image,
    dtype_,
):
    inputs = prepare_input(image_path, prompt, processor, max_image_size, split_image)
    inputs["pixel_values"] = inputs["pixel_values"].to(model._dtype)
    print("dtype", model._dtype)
    inputs = {k: v.to(model.parameters()[0].place) for k, v in inputs.items()}

    stopping_criteria = StoppingCriteriaList(
        [
            EosTokenCriteria(2),
            KeywordsStoppingCriteria([93519]),
        ]
    )
    with paddle.no_grad(), paddle.amp.auto_cast(dtype=dtype_):
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            stopping_criteria=stopping_criteria,
            tokenizer=processor.tokenizer,
            do_sample=True,
            temperature=0.9,
        )
    for i in range(tuple(inputs["input_ids"].shape)[0]):
        output_text = processor.tokenizer.decode(output[i][0], skip_special_tokens=True).replace("<|im_end|>", "")

    return output_text


def main():
    args = parse_arguments()
    processor = AriaProcessor.from_pretrained(args.base_model_path, tokenizer_path=args.tokenizer_path)
    model = load_model(args.base_model_path, args)
    result = inference(
        args.image_path,
        args.prompt,
        model,
        processor,
        args.max_image_size,
        args.split_image,
        args.dtype,
    )
    print(result)


if __name__ == "__main__":
    main()

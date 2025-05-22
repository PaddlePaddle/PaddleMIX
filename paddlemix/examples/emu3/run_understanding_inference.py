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

from paddlenlp.generation import (
    GenerationConfig,
    MaxLengthCriteria,
    StoppingCriteriaList,
)
from PIL import Image

from paddlemix.models.emu3 import Emu3ForCausalLM, Emu3VisionVQModel
from paddlemix.models.emu3.tokenization_emu3 import Emu3Tokenizer
from paddlemix.models.emu3.utils_emu3 import EosTokenCriteria
from paddlemix.processors import Emu3Processor, Emu3VisionVQImageProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="BAAI/Emu3-Chat")
parser.add_argument("--vq_model_path", type=str, default="BAAI/Emu3-VisionTokenizer")
parser.add_argument("--image_path", type=str, default="paddlemix/demo_images/emu3_demo.png")
parser.add_argument("--question", type=str, default="Please describe the image")
parser.add_argument("--max_new_tokens", type=int, default=1024)

parser.add_argument("--dtype", type=str, default="bfloat16")

args = parser.parse_args()


# prepare model and processor
model = Emu3ForCausalLM.from_pretrained(args.model_path, dtype=args.dtype).eval()
tokenizer = Emu3Tokenizer.from_pretrained(args.model_path, padding_side="left")

image_processor = Emu3VisionVQImageProcessor.from_pretrained(args.vq_model_path)
image_tokenizer = Emu3VisionVQModel.from_pretrained(args.vq_model_path).eval()

processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

# prepare input
text = [args.question]
image = Image.open(args.image_path)
image = [image]

inputs = processor(
    text=text,
    image=image,
    mode="U",
    padding_image=True,
    padding="longest",
    return_tensors="pd",
)
# [1, 4188]
# PretrainedTokenizer(name_or_path='', vocab_size=184622,
# model_max_len=1000000000000000019884624838656, padding_side='left', truncation_side='right',
# special_tokens={'bos_token': '<|extra_203|>', 'eos_token': '<|extra_204|>', 'pad_token': '<|endoftext|>'})

# prepare hyper parameters
GENERATION_CONFIG = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id
)

stopping_criteria = StoppingCriteriaList(
    [
        MaxLengthCriteria(args.max_new_tokens + inputs.input_ids.shape[1]),
        EosTokenCriteria(int(model.config.eos_token_id)),
    ]
)

# generate
outputs = model.generate(
    inputs.input_ids, GENERATION_CONFIG, max_new_tokens=args.max_new_tokens, stopping_criteria=stopping_criteria
)
# print('outputs', outputs)
answers = processor.batch_decode(outputs[0], skip_special_tokens=True)
for ans in answers:
    print(ans)

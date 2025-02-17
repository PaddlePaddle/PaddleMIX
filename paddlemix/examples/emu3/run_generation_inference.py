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
from paddlenlp.generation.logits_process import (
    LogitsProcessorList,
    PrefixConstrainedLogitsProcessor,
)
from PIL import Image

from paddlemix.models.emu3 import Emu3ForCausalLM, Emu3VisionVQModel
from paddlemix.models.emu3.tokenization_emu3 import Emu3Tokenizer
from paddlemix.models.emu3.utils_emu3 import (
    EosTokenCriteria,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from paddlemix.processors import Emu3Processor, Emu3VisionVQImageProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="BAAI/Emu3-Gen")
parser.add_argument("--vq_model_path", type=str, default="BAAI/Emu3-VisionTokenizer")
parser.add_argument("--prompt", type=str, default="a portrait of young girl.")
parser.add_argument("--ratio", type=str, default="1:1")
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--width", type=int, default=720)
parser.add_argument("--dtype", type=str, default="bfloat16")

args = parser.parse_args()

# prepare model and processor
model = Emu3ForCausalLM.from_pretrained(args.model_path, dtype=args.dtype).eval()
tokenizer = Emu3Tokenizer.from_pretrained(args.model_path, padding_side="left")

image_processor = Emu3VisionVQImageProcessor.from_pretrained(args.vq_model_path)
image_tokenizer = Emu3VisionVQModel.from_pretrained(args.vq_model_path).eval()

processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

# prepare input
POSITIVE_PROMPT = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."

classifier_free_guidance = 3.0
prompt = [args.prompt]
prompt = [p + POSITIVE_PROMPT for p in prompt]

kwargs = dict(
    mode="G",
    prompt=prompt,
    ratio=[args.ratio],
    image_area=args.height * args.width,
    return_tensors="pd",
    padding="longest",
)
pos_inputs = processor(text=prompt, **kwargs)
neg_inputs = processor(text=[NEGATIVE_PROMPT] * len(prompt), **kwargs)

# prepare hyper parameters
h = pos_inputs.image_size[:, 0]
w = pos_inputs.image_size[:, 1]
constrained_fn = processor.build_prefix_constrained_fn(h, w)
logits_processor = LogitsProcessorList(
    [
        UnbatchedClassifierFreeGuidanceLogitsProcessor(
            classifier_free_guidance,
            model,
            unconditional_ids=neg_inputs.input_ids,
        ),
        PrefixConstrainedLogitsProcessor(
            constrained_fn,
            num_beams=1,
        ),
    ]
)

max_new_tokens = 40960
stopping_criteria = StoppingCriteriaList(
    [
        MaxLengthCriteria(max_new_tokens + pos_inputs.input_ids.shape[1]),
        EosTokenCriteria(int(model.config.eos_token_id)),
    ]
)
print(f"eos_token_id:{model.config.eos_token_id}")
print(f"boi_token_id:{model.config.boi_token_id}")

GENERATION_CONFIG = GenerationConfig(
    use_cache=True,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.pad_token_id,
    max_new_tokens=max_new_tokens,
    do_sample=True,
    top_k=2048,
    trunc_input=False,
)

# generate
outputs = model.generate(
    pos_inputs.input_ids, GENERATION_CONFIG, logits_processors=logits_processor, stopping_criteria=stopping_criteria
)
print("outputs", outputs)
for idx_i, out in enumerate(outputs[0]):
    mm_list = processor.decode(out)
    for idx_j, im in enumerate(mm_list):
        if not isinstance(im, Image.Image):
            continue
        im.save(f"result_{idx_i}_{idx_j}.png")

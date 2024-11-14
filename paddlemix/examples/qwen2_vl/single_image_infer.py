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

import datetime

import paddle
from paddlenlp.transformers import Qwen2Tokenizer

from paddlemix.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from paddlemix.processors.qwen2_vl_processing import (
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    process_vision_info,
)

benchmark = True
warm_up = 3

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_NAME, dtype="bfloat16")

image_processor = Qwen2VLImageProcessor()
tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_NAME)
processor = Qwen2VLProcessor(image_processor, tokenizer)

# min_pixels = 256*28*28 # 200704
# max_pixels = 1280*28*28 # 1003520
# processor = Qwen2VLProcessor(image_processor, tokenizer, min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                # "image": "paddlemix/demo_images/examples_image1.jpg",
                "image": "/root/paddlejob/workspace/env_run/output/changwenbin/PaddleMIX/paddlemix/demo_images/examples_image1.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
image_inputs, video_inputs = process_vision_info(messages)

question = "Describe this image."
image_pad_token = "<|vision_start|><|image_pad|><|vision_end|>"
text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{image_pad_token}{question}<|im_end|>\n<|im_start|>assistant\n"

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pd",
)

if warm_up > 0:
    for _ in range(warm_up):
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)  # already trimmed in paddle
if benchmark:
    repeat_times = 10
    sumtime = 0.0
    for i in range(repeat_times):
        paddle.device.synchronize()
        starttime = datetime.datetime.now()

        paddle.device.synchronize()
        import nvtx

        generate_nvtx = nvtx.start_range(message="generate", color="green")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)  # already trimmed in paddle

        paddle.device.synchronize()
        nvtx.end_range(generate_nvtx)

        paddle.device.synchronize()
        endtime = datetime.datetime.now()

        duringtime = endtime - starttime
        duringtime = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
        sumtime += duringtime
        print(f"Single {MODEL_NAME} end to end time : ", duringtime, "ms")

        paddle.device.cuda.empty_cache()
        inference_global_mem = paddle.device.cuda.memory_reserved() / (1024**3)
        print(f"Inference used CUDA memory : {inference_global_mem:.3f} GiB")

    print(f"Single {MODEL_NAME} ave end to end time : ", sumtime / repeat_times, "ms")

    paddle.device.cuda.empty_cache()
    inference_global_mem = paddle.device.cuda.memory_reserved() / (1024**3)
    print(f"Inference used CUDA memory : {inference_global_mem:.3f} GiB")
    cuda_mem_after_used = paddle.device.cuda.max_memory_allocated() / (1024**3)
    print(f"Max used CUDA memory : {cuda_mem_after_used:.3f} GiB")
else:
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)  # already trimmed in paddle

output_text = processor.batch_decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("output_text:\n", output_text[0])

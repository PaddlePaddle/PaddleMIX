# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from datasets import load_dataset
from math_verify import parse, verify

# training
from paddlenlp.trl import ModelConfig
from paddlenlp.trl.utils import ScriptArguments
from paddlenlp.utils.import_utils import import_module

sys.path.append("paddlemix/examples/r1_mllm")
from r1_mllm.dataset.qwen2_vl_dataset import Qwen2VLDataCollatorForSeq2Seq
from r1_mllm.trainer import GRPOConfig, Qwen2VLGRPOTrainer
from r1_mllm.utils.args import TrlParser
from r1_mllm.utils.constant import TEMPLATE_MAPPING
from r1_mllm.utils.tokenizer import get_processor


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(default=12845056, metadata={"help": "Maximum number of pixels for the image"})
    min_pixels: Optional[int] = field(default=3136, metadata={"help": "Minimum number of pixels for the image"})
    dataset_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset configuration name. Corresponds to the `name` argument of the `datasets.load_dataset` "
            "function."
        },
    )


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass
        if reward == 0.0:
            try:
                sol_match = re.search("<answer>(.*?)</answer>", sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                content_match = re.search("<answer>(.*?)</answer>", content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = "<think>.*?</think>\\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [(1.0 if match else 0.0) for match in matches]


reward_funcs_registry = {"accuracy": accuracy_reward, "format": format_reward}
SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ]
        }

    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=example["problem"]),
                        },
                    ],
                }
            ]
        }

    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)
    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    processor, tokenizer = get_processor(
        os.path.basename(model_args.model_name_or_path), model_args.model_name_or_path
    )
    model_name = os.path.basename(model_args.model_name_or_path)
    template_name = TEMPLATE_MAPPING.get(model_name, "qwen2_5_vl")
    TEMPLATES = import_module(f"paddlemix.models.{template_name}.template.TEMPLATES")
    data_collator = Qwen2VLDataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        template_name=template_name,
        template=TEMPLATES[template_name],
        processor=processor,
        label_pad_token_id=-100,
    )
    trainer_cls = Qwen2VLGRPOTrainer
    print("using: ", trainer_cls)
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        data_collator=data_collator,
        eval_dataset=None,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        dtype="bfloat16" if training_args.bf16 else "float16",
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

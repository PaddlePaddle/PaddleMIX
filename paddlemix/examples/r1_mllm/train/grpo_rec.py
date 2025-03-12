import ast
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import os
import random
import re
import sys
from typing import Optional, Tuple, Union, Iterable

import paddle
import paddlenlp
import yaml
from math_verify import parse, verify
from PIL import Image

# training
from paddlenlp.trl import ModelConfig
from paddlenlp.trl.utils import ScriptArguments
from paddlenlp.utils.import_utils import import_module

sys.path.append('paddlemix/examples/r1_mllm')
from r1_mllm.trainer import GRPOConfig, Qwen2VLGRPOTrainer
from r1_mllm.dataset.qwen2_vl_dataset import Qwen2VLRECDataset
from r1_mllm.utils.tokenizer import get_processor
from r1_mllm.utils.args import TrlParser
from r1_mllm.utils.constant import TEMPLATE_MAPPING

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
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format'"
        },
    )
    max_pixels: Optional[int] = field(
        default=12845056, metadata={"help": "Maximum number of pixels for the image"}
    )
    min_pixels: Optional[int] = field(
        default=3136, metadata={"help": "Minimum number of pixels for the image"}
    )
    image_root: Optional[str] = field(
        default=None, metadata={"help": "Root directory of the image"}
    )
    attn_implementation: Optional[str] = field(
        default='flash_attention_2', metadata={"help": "Attention type"}
    )

SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"


"""
    If the iou of the bbox predicted by the model and the ground truth is greater than 0.5, the reward is 1.0, otherwise 0.0 .
    This is a hard reward, maybe the soft reward is better and could be used in the future .
"""


def iou_reward(completions, solution, **kwargs):
    def iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    # bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
    for content, sol in zip(contents, solution):
        reward = 0.0
        
        # convert str to list
        if isinstance(sol,str):
            sol = ast.literal_eval(sol)

        # Try symbolic verification first
        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                bbox_match = re.search(bbox_pattern, content_answer)
                if bbox_match:
                    bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                    if iou(bbox, sol) > 0.5:
                        reward = 1.0
        except Exception as e:
            print(e)  # Continue to next verification method if this fails
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- rank:{kwargs.get('rank',0)} ------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

def float_iou_reward(completions, solution, **kwargs):
    def iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2] - 1, box2[2] - 1)
        inter_y2 = min(box1[3] - 1, box2[3] - 1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
        else:
            inter = 0
        union = (
            (box1[2] - box1[0]) * (box1[3] - box1[1])
            + (box2[2] - box2[0]) * (box2[3] - box2[1])
            - inter
        )
        return float(inter) / union

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = "<answer>(.*?)</answer>"
    
    #bbox_pattern = "\\[(\\d+),\\s*(\\d+),\\s*(\\d+),\\s*(\\d+)]"
    bbox_pattern = r"\[([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+)\]"
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                bbox_match = re.search(bbox_pattern, content_answer)
                if bbox_match:
                    bbox = [
                        float(bbox_match.group(1)),
                        float(bbox_match.group(2)),
                        float(bbox_match.group(3)),
                        float(bbox_match.group(4)),
                    ]
                    if iou(bbox, sol) > 0.5:
                        reward = 1.0
        except Exception as e:
            print(e)
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(
                    f"------------- {current_time} Accuracy reward: {reward} -------------\n"
                )
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"

    rewards = []
    # kwargs['prompts']
    for completion in completions:
        reward = 0
        completion_contents = completion[0]["content"]
        match = re.fullmatch(pattern, completion_contents, re.DOTALL)
        if match:
            reward = 1
        rewards.append(reward)
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(
                    f"------------- {current_time} Format reward: {reward} -------------\n"
                )
                f.write(f"Content: {completion_contents}\n")

    return rewards

reward_funcs_registry = {"accuracy": iou_reward, "format": format_reward}


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)
    model_path = model_args.model_name_or_path
    model_name = os.path.basename(model_path)

    processor,tokenizer = get_processor(os.path.basename(model_args.model_name_or_path),model_args.model_name_or_path)
    template_name = TEMPLATE_MAPPING.get(model_name, "qwen2_5_vl")

    TEMPLATES = import_module(f"paddlemix.models.{template_name}.template.TEMPLATES")
    dataset = Qwen2VLRECDataset(script_args.dataset_name, script_args,training_args,model_args,tokenizer,processor,template=TEMPLATES[template_name])

    trainer_cls = Qwen2VLGRPOTrainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        attn_implementation=script_args.attn_implementation,
        dtype="bfloat16" if training_args.bf16 else "float16"
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

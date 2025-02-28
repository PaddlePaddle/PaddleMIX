import json
import math
import os
import random
import re
import sys

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple, Union, Iterable

import paddle
import paddlenlp
import yaml
from math_verify import parse, verify
from PIL import Image
from paddlenlp.trl import ModelConfig
from paddlenlp.trl.utils import ScriptArguments
from paddlenlp.trainer.argparser import PdArgumentParser,DataClassType,DataClass

# training
from paddlemix.models.qwen2_5_vl.supervised import _encode_supervised_example
from paddlemix.models.qwen2_5_vl import MIXQwen2_5_Tokenizer
from paddlemix.processors.qwen2_5_vl_processing import Qwen2_5_VLImageProcessor, Qwen2_5_VLProcessor
from paddlemix.models.qwen2_5_vl.template import TEMPLATES


class TrlParser(PdArgumentParser):
    """
    A subclass of [`transformers.HfArgumentParser`] designed for parsing command-line arguments with dataclass-backed
    configurations, while also supporting configuration file loading and environment variable management.

    Args:
        dataclass_types (`Union[DataClassType, Iterable[DataClassType]]` or `None`, *optional*, defaults to `None`):
            Dataclass types to use for argument parsing.
        **kwargs:
            Additional keyword arguments passed to the [`transformers.HfArgumentParser`] constructor.

    Examples:

    ```yaml
    # config.yaml
    env:
        VAR1: value1
    arg1: 23
    ```

    ```python
    # main.py
    import os
    from dataclasses import dataclass
    from trl import TrlParser

    @dataclass
    class MyArguments:
        arg1: int
        arg2: str = "alpha"

    parser = TrlParser(dataclass_types=[MyArguments])
    training_args = parser.parse_args_and_config()

    print(training_args, os.environ.get("VAR1"))
    ```

    ```bash
    $ python main.py --config config.yaml
    (MyArguments(arg1=23, arg2='alpha'),) value1

    $ python main.py --arg1 5 --arg2 beta
    (MyArguments(arg1=5, arg2='beta'),) None
    ```
    """

    def __init__(
        self,
        dataclass_types: Optional[Union[DataClassType, Iterable[DataClassType]]] = None,
        **kwargs,
    ):
        # Make sure dataclass_types is an iterable
        if dataclass_types is None:
            dataclass_types = []
        elif not isinstance(dataclass_types, Iterable):
            dataclass_types = [dataclass_types]
        

        # TODO: 
        # Check that none of the dataclasses have the "config" field
        # for dataclass_type in dataclass_types:
        #     if "config" in dataclass_type.__dataclass_fields__:
        #         raise ValueError(
        #             f"Dataclass {dataclass_type.__name__} has a field named 'config'. This field is reserved for the "
        #             f"config file path and should not be used in the dataclass."
        #         )

        super().__init__(dataclass_types=dataclass_types, **kwargs)

    def parse_args_and_config(
        self, args: Optional[Iterable[str]] = None, return_remaining_strings: bool = False
    ) -> tuple[DataClass, ...]:
        """
        Parse command-line args and config file into instances of the specified dataclass types.

        This method wraps [`transformers.HfArgumentParser.parse_args_into_dataclasses`] and also parses the config file
        specified with the `--config` flag. The config file (in YAML format) provides argument values that replace the
        default values in the dataclasses. Command line arguments can override values set by the config file. The
        method also sets any environment variables specified in the `env` field of the config file.
        """
        args = list(args) if args is not None else sys.argv[1:]
        if "--config" in args:
            # Get the config file path from
            config_index = args.index("--config")
            args.pop(config_index)  # remove the --config flag
            config_path = args.pop(config_index)  # get the path to the config file
            with open(config_path) as yaml_file:
                config = yaml.safe_load(yaml_file)

            # Set the environment variables specified in the config file
            if "env" in config:
                env_vars = config.pop("env", {})
                if not isinstance(env_vars, dict):
                    raise ValueError("`env` field should be a dict in the YAML file.")
                for key, value in env_vars.items():
                    os.environ[key] = str(value)

            # Set the defaults from the config values
            config_remaining_strings = self.set_defaults_with_config(**config)
        else:
            config_remaining_strings = []

        # Parse the arguments from the command line
        output = self.parse_args_into_dataclasses(args=args, return_remaining_strings=return_remaining_strings)

        # Merge remaining strings from the config file with the remaining strings from the command line
        if return_remaining_strings:
            args_remaining_strings = output[-1]
            return output[:-1] + (config_remaining_strings + args_remaining_strings,)
        else:
            return output

    def set_defaults_with_config(self, **kwargs) -> list[str]:
        """
        Overrides the parser's default values with those provided via keyword arguments.

        Any argument with an updated default will also be marked as not required
        if it was previously required.

        Returns a list of strings that were not consumed by the parser.
        """
        # If an argument is in the kwargs, update its default and set it as not required
        for action in self._actions:
            if action.dest in kwargs:
                action.default = kwargs.pop(action.dest)
                action.required = False
        remaining_strings = [item for key, value in kwargs.items() for item in [f"--{key}", str(value)]]
        return remaining_strings



print(__file__)
sys.path.append('paddlemix/examples/vlm_r1/open-r1-multimodal/src')
from open_r1.trainer import GRPOConfig, Qwen2VLGRPOTrainer

# from paddlemix.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_flashatt
# TrlParser get_peft_config

# def custom_forward(
#     self,
#     hidden_states: paddle.Tensor,
#     cu_seqlens: paddle.Tensor,
#     rotary_pos_emb: Optional[paddle.Tensor] = None,
#     position_embeddings: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
# ) -> paddle.Tensor:
#     seq_length = tuple(hidden_states.shape)[0]
#     q, k, v = (
#         self.qkv(hidden_states)
#         .reshape([seq_length, 3, self.num_heads, -1])
#         .transpose(perm=[1, 0, 2, 3])
#         .unbind(axis=0)
#     )
#     if position_embeddings is None:
#         logger.warning_once(
#             "The attention layers in this model are transitioning from computing the RoPE embeddings internally through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed `position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be removed and `position_embeddings` will be mandatory."
#         )
#         emb = paddle.concat(x=(rotary_pos_emb, rotary_pos_emb), axis=-1)
#         cos = emb.cos().astype(dtype="float32")
#         sin = emb.sin().astype(dtype="float32")
#     else:
#         cos, sin = position_embeddings
#         cos = cos.astype("float32")
#         sin = sin.astype("float32")
#     (
#         q,
#         k,
#     ) = apply_rotary_pos_emb_flashatt(
#         q.unsqueeze(axis=0), k.unsqueeze(axis=0), cos, sin
#     )
#     q = q.squeeze(axis=0)
#     k = k.squeeze(axis=0)
#     max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
#     attn_output = (
#         transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.flash_attn_varlen_func(
#             q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
#         ).reshape(seq_length, -1)
#     )
#     attn_output = self.proj(attn_output)
#     return attn_output


# (
#     transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLVisionFlashAttention2.forward
# ) = custom_forward


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


SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"


class LazySupervisedDataset(paddle.io.Dataset):
    def __init__(self,
        data_path: str,
        script_args: GRPOScriptArguments,
        training_args: GRPOConfig,
        tokenizer,processor,template
    ):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []
        self.tokenizer = tokenizer
        self.processor = processor
        self.template = template
        self.max_seq_length = training_args.max_prompt_length
        self.max_image_size = 512 # TODO
        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None
                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")
                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(
                            ":"
                        )
                        if "%" in sampling_number:
                            sampling_number = math.ceil(
                                int(sampling_number.split("%")[0])
                                * len(cur_data_dict)
                                / 100
                            )
                        else:
                            sampling_number = int(sampling_number)
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def _preprocess_image(self, image):
        r"""
        Pre-processes a single image.
        """
        image_resolution = self.max_image_size
        if max(image.width, image.height) > image_resolution:
            resize_factor = image_resolution / max(image.width, image.height)
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")

        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)

        return image

    def get_image_path(self, image_path):
        # image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        return self.processor.image_processor

    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()
        
        # Ensure the first conversation contains an image placeholder
        if "<image>" not in data_item["messages"][0]["content"]:
            data_item["messages"][0]["content"] = "<image>\n" + data_item["messages"][0]["content"]

        # Merge the image path
        # image_path = self.get_image_path(data_item["image_path"][0])  # TODO: now only single image

        messages = data_item["messages"]

        input_ids, labels = _encode_supervised_example(
            messages=messages,
            system="",
            tools="",
            images=[data_item['image']],
            videos=[],
            template=self.template,
            tokenizer=self.tokenizer,
            processor=self.processor,
            cutoff_len=self.max_seq_length,
            train_on_prompt=False,
            mask_history=False,
        )
        attention_mask = [1] * len(input_ids)

        # Create the final return dictionary
        ret = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            images=[data_item['image']],
        )

        return ret

    def pure_text_get_item(self, data_item):
        messages = data_item["messages"]

        input_ids, labels = _encode_supervised_example(
            messages=messages,
            system="",
            tools="",
            images=[],
            videos=[],
            template=self.template,
            tokenizer=self.tokenizer,
            processor=self.processor,
            cutoff_len=self.max_seq_length,
            train_on_prompt=False,
            mask_history=False,
        )
        attention_mask = [1] * len(input_ids)

        # Create the final return dictionary
        ret = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            images=[],
        )
        
        return ret

    def __getitem__(self, i):
        
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ]
            }

        QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."

        def make_conversation_image(example):
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": QUESTION_TEMPLATE.format(Question=example["problem"])
                    },
                    {
                        "role": "assistant",
                        "content": str(example['solution'])
                    }
                ]
            }

        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        if "image" in example:
            image_path = os.path.join(image_root, example["image"])
            while not os.path.exists(image_path):
                print(
                    f"Warning: Image {image_path} not found, randomly selecting another image"
                )
                new_index = random.randint(0, len(self.list_data_dict) - 1)
                example = self.list_data_dict[new_index]
                image_path = os.path.join(image_root, example["image"])
            image = self._preprocess_image(Image.open(image_path).convert("RGB"))
        else:
            image = None
        
        data_item =  {
            "image": image,
            "image_path": example['image'],
            # "problem": example["problem"],
            # "label": example["solution"],
            "messages": make_conversation_image(example)["messages"]
            if "image" in example
            else make_conversation(example)["prompt"],
        }
        return self.multi_modal_get_item(data_item)
        #     try:
        #         data_item = self.raw_data[i]
        #         if "images" in data_item and len(data_item["images"]) != 0:
        #             ret = self.multi_modal_get_item(data_item)  # TODO: 暂时都是单图
        #         else:
        #             ret = self.pure_text_get_item(data_item)  # TODO: 纯文
        #         break
        #     except Exception as e:
        #         print(e, self.ds_name, flush=True)
        #         if not isinstance(e, UnidentifiedImageError):
        #             traceback.print_exc()
        #         data_item = self.raw_data[i]
        #         if "images" in data_item:
        #             if type(data_item["images"]) == list:
        #                 images = [item for item in data_item["images"]]
        #                 print(f"Failed to load image: {images}, the dataset is: {self.ds_name}")
        #             else:
        #                 data_path = data_item["images"]
        #                 print(f"Failed to load image: {data_path}, the dataset is: {self.ds_name}")
        #         elif "video" in data_item:
        #             data_path = data_item["video"]
        #             print(f"Failed to load video: {data_path}, the dataset is: {self.ds_name}")
        #         i = random.randint(0, len(self.raw_data) - 1)
        # return ret

"""
    If the iou of the bbox predicted by the model and the ground truth is greater than 0.5, the reward is 1.0, otherwise 0.0 .
    This is a hard reward, maybe the soft reward is better and could be used in the future .
"""


def iou_reward(completions, solution, **kwargs):
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


# def format_reward(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""
#     pattern = "<think>.*?</think>\\s*<answer>.*?</answer>"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
#     import pdb;pdb.set_trace()
#     return [(1.0 if match else 0.0) for match in matches]
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"\s*<think>.*?</think>\s*<answer>.*?</answer>"
    rewards = []
    for completion in completions:
        reward = 0
        completion_contents = completion[0]["content"]
        match = re.fullmatch(pattern, completion_contents, re.DOTALL)
        if match:
            reward = 1
        # import pdb;pdb.set_trace()
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

    image_processor = Qwen2_5_VLImageProcessor()
    tokenizer = MIXQwen2_5_Tokenizer.from_pretrained(model_path, padding_side="left")
    processor = Qwen2_5_VLProcessor(image_processor, tokenizer)
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args,training_args,tokenizer,processor,template=TEMPLATES['qwen2_5_vl'])

    trainer_cls = Qwen2VLGRPOTrainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

import copy
import os
import random
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sequence, Dict
from dataclasses import dataclass, field
from paddlemix.utils.log import logger

import numpy as np
import paddle
import paddle.nn.functional as F
import paddlenlp
from paddlenlp.data import DataCollatorForSeq2Seq
import datasets
import PIL.Image
from packaging import version
from paddle.io import Sampler
from paddlenlp.trainer import Trainer,TrainerCallback
from paddlenlp.transformers.model_utils import (
    PretrainedModel,
    _add_variant,
    load_sharded_checkpoint,
    unwrap_model,
)
from paddlenlp.transformers.tokenizer_utils_base import PretrainedTokenizerBase
from paddlemix.models.qwen2_vl import Qwen2VLForConditionalGeneration
from paddlemix.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from paddlemix.models.aria.model.modeling_aria import AriaForConditionalGeneration
from paddlemix.models.qwen2_vl.template import TEMPLATES

from .grpo_config import GRPOConfig
from ..utils.tokenizer import get_tokenizer,get_processor
from ..utils.models import create_reference_model,freeze_params,get_model
from ..utils.data import apply_chat_template,is_conversational,maybe_apply_chat_template
from ..utils.distributed import all_gather


if paddlenlp.trainer.integrations.is_wandb_available():
    import wandb

RewardFunc = Union[
    str, paddlenlp.transformers.PretrainedModel, Callable[[list, list], list[float]]
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        indexes,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed:int = None,
    ):
        self.indexes = indexes
        self.num_samples = len(indexes)
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count

    def __iter__(self):
        # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
        indexes = self.indexes
        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield [index]

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class Qwen2VLGRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, paddlenlp.transformers.PretrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[
            Union[datasets.Dataset, datasets.IterableDataset]
        ] = None,
        eval_dataset: Optional[
            Union[
                datasets.Dataset,
                datasets.IterableDataset,
                dict[str, Union[datasets.Dataset, datasets.IterableDataset]],
            ]
        ] = None,
        processing_class: PretrainedTokenizerBase = None,
        reward_processing_classes: Optional[
            Union[
                PretrainedTokenizerBase,
                list[PretrainedTokenizerBase],
            ]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[paddle.optimizer.Optimizer],
            Optional[paddle.optimizer.lr.LambdaDecay],
        ] = (None, None),
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation = None,
        dtype: str = "bfloat16",
        data_collator=None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        model_init_kwargs = args.model_init_kwargs or {}
        if attn_implementation:
            model_init_kwargs["attn_implementation"] = attn_implementation
            print("attn_implementation:",attn_implementation)

        if model_init_kwargs.get("dtype") is None:
            model_init_kwargs["dtype"] = dtype
        
        
        if isinstance(model, str):
            model_id = model
            dtype = model_init_kwargs.get("dtype")
            if (
                isinstance(dtype, paddle.dtype)
                or dtype == "auto"
                or dtype is None
            ):
                pass
            elif isinstance(dtype, str):
                # dtype = getattr(paddle, dtype)
                model_init_kwargs["dtype"] = dtype
            else:
                raise ValueError(
                    f"Invalid `dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing a `paddle.dtype` (e.g., 'float32'), but got {dtype}."
                )
            model_init_kwargs["use_cache"] = (
                False
                if args.recompute
                else model_init_kwargs.get("use_cache")
            )
        processor_kwargs = {
            "max_pixels": max_pixels,
            "min_pixels": min_pixels,
        }
        model_path = model_id
        model_name = os.path.basename(model_path)
        model = get_model(model_name,model_path,**model_init_kwargs) # model_id: Qwen/Qwen2.5-VL-3B-Instruct
        if args.freeze_vision:
            freeze_params(model.visual)
        
        self.ref_model = create_reference_model(model)

        if processing_class is None:
            processor,tokenizer = get_processor(model_name,model_path,**processor_kwargs)
            processing_class = processor
        
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = paddlenlp.transformers.AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        elif len(reward_processing_classes) != len(reward_funcs):
            raise ValueError(
                "The number of reward processing classes must match the number of reward functions."
            )
        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(reward_func, paddlenlp.transformers.PretrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = (
                        paddlenlp.transformers.AutoTokenizer.from_pretrained(
                            reward_func.config._name_or_path
                        )
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes
        self.processing_class = processing_class
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.num_iterations = args.num_iterations
        self.generation_config = paddlenlp.generation.GenerationConfig(
            use_cache = True,
            max_new_tokens=self.max_completion_length,
            decode_strategy="sampling",
            do_sample=True,
            top_p=1,
            top_k=50,
            eos_token_id=[tokenizer.pad_token_id,tokenizer.eos_token_id],
            pad_token_id=tokenizer.pad_token_id,
            temperature=1.0,
        )
        
        # Bug: will cause same llm sampling results if set to same seed for every process
        self.beta = args.beta
        self.epsilon = args.epsilon
        self._metrics = defaultdict(list)
        self._step = 0
        self._buffered_inputs = [None] * args.gradient_accumulation_steps
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        self.model_accepts_loss_kwargs = False

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.args.dataset_world_size
        global_batch_size = self.args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
            
        set_seed(self.args.seed) # for data indexes
        self.indexes = paddle.randperm(len(self.train_dataset)).tolist()
        training_seed = self.args.seed + self.args.dataset_rank
        set_seed(training_seed) # for generate sampling

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_per_token_logps(
        self, model, input_ids, attention_mask, pixel_values, image_grid_thw
    ):
        logits = model(input_ids,attention_mask=attention_mask,pixel_values=pixel_values,image_grid_thw=image_grid_thw,).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = F.log_softmax(logits_row,axis=-1)
            token_log_prob = paddle.take_along_axis(
                arr=log_probs,
                axis=1,
                indices=input_ids_row.unsqueeze(axis=1),
                broadcast=False,
            ).squeeze(axis=1)
            per_token_logps.append(token_log_prob)
        return paddle.stack(x=per_token_logps)

    def _prepare_inputs(
        self, inputs: dict[str, Union[paddle.Tensor, Any]],
    ) -> dict[str, Union[paddle.Tensor, Any]]:
        return inputs


    def _generate_and_score_completions(
        self,
        inputs: dict[str, Union[paddle.Tensor, Any]],
        model
    ) -> dict[str,Union[paddle.Tensor,Any]]:
        device = inputs['pixel_values'].place
        input_ids = inputs["input_ids"]
        prompts = self.processing_class.batch_decode(input_ids, skip_special_tokens=True)
        prompt_ids, prompt_mask = inputs["input_ids"], inputs["attention_mask"]

        # process visual input
        _, pixel_seq_len,pixel_dim = inputs["pixel_values"].shape
        inputs["pixel_values"] = inputs["pixel_values"].reshape([-1,pixel_dim])

        _, _,g2 = inputs["image_grid_thw"].shape
        inputs["image_grid_thw"] = inputs["image_grid_thw"].reshape([-1,g2])

        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Regular generation path
        if paddle.distributed.is_initialized():
            with paddle.no_grad():
                completion_ids = unwrap_model(model).generate(**inputs,generation_config=self.generation_config)[0]
        else:
            with paddle.no_grad():
                completion_ids = model.generate(**inputs,generation_config=self.generation_config)[0]

        prompt_length = prompt_ids.shape[1]
        
        prompt_completion_ids = paddle.concat([prompt_ids,completion_ids],axis=1)
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = paddle.full(shape=(is_eos.shape[0],), fill_value=is_eos.shape[1], dtype="int64")
        eos_idx[is_eos.astype("bool").any(axis=1)] = is_eos.astype(dtype="int64").argmax(axis=1)[is_eos.astype("bool").any(axis=1)]
        sequence_indices = paddle.arange(end=is_eos.shape[1]).expand(
            shape=[is_eos.shape[0], -1]
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(axis=1)).astype(
            dtype="int64"
        )
        attention_mask = paddle.concat(x=[prompt_mask, completion_mask], axis=1)



        with paddle.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
                )
                old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
                )
            else:
                # TODO: lora
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
                    )
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
        
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions = [
            [{"role": "assistant", "content": completion}]
            for completion in completions
        ]

        rewards_per_func = paddle.zeros(shape=[len(prompts), len(self.reward_funcs)])

        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PretrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pd", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with paddle.no_grad():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                solution = self.processing_class.batch_decode(inputs['labels'])
                reward_kwargs = {
                    "prompts":prompts,
                    "solution":solution
                }
                if paddle.distributed.is_initialized():
                    reward_kwargs['rank'] = paddle.distributed.get_rank()

                output_reward_func = reward_func(
                    completions=completions, **reward_kwargs
                )
                rewards_per_func[:,i] = paddle.to_tensor(
                    data=output_reward_func, dtype="float32", place=device
                )

        # Gather rewards across processes
        rewards_per_func = paddle.concat(all_gather(rewards_per_func),axis=0)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(axis=1)
        # Compute grouped-wise rewards
        # Each group consists of num_generations completions for the same prompt
        mean_grouped_rewards = rewards.reshape([-1, self.num_generations]).mean(axis=1)
        std_grouped_rewards = rewards.reshape([-1, self.num_generations]).std(axis=1)
        
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, axis=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, axis=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        # Get only the local slice of advantages
        process_slice = slice(
            self.args.dataset_rank * len(prompts),
            (self.args.dataset_rank + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        completion_length = paddle.concat(all_gather(completion_mask.astype("float32").sum(1))).mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = paddle.concat(all_gather(rewards_per_func)).mean(axis=0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PretrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(paddle.concat(all_gather(rewards)).mean().item())
        self._metrics["reward_std"].append(paddle.concat(all_gather(std_grouped_rewards)).mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw
        }
    
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # Check if we need to generate new completions or use buffered ones
        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs, model)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1


        # Get the prepared inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        
        # Concatenate for full sequence
        input_ids = paddle.concat(x=[prompt_ids, completion_ids], axis=1)
        attention_mask = paddle.concat(x=[prompt_mask, completion_mask], axis=1)

        # Get the current policy's log probabilities
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, pixel_values, image_grid_thw)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_ids.shape[1] - 1:]

        # Get the advantages from inputs
        advantages = inputs["advantages"]

        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        # Compute the policy ratio and clipped version
        coef_1 = paddle.exp(per_token_logps - old_per_token_logps)
        coef_2 = paddle.clip(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -paddle.min(paddle.stack([per_token_loss1, per_token_loss2]),axis=0)
        completion_mask = completion_mask.astype("float32")
        # Add KL penalty if beta > 0
        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = paddle.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl

            # Log KL divergence
            mean_kl = ((per_token_kl * completion_mask).sum(axis=1) / completion_mask.sum(axis=1)).mean()
            self._metrics["kl"].append(paddle.to_tensor(all_gather(mean_kl)).mean().item())

        # Compute final loss
        loss = ((per_token_loss * completion_mask).sum(axis=1) / completion_mask.sum(axis=1)).mean()

        # Log clip ratio
        is_clipped = (per_token_loss1 < per_token_loss2).astype("float32")
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics["clip_ratio"].append(paddle.to_tensor(all_gather(clip_ratio)).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None,**kwargs) -> None:
        metrics = {key: (sum(val) / len(val)) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return
        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(
            self.model.config._name_or_path
        ):
            base_model = self.model.config._name_or_path
        else:
            base_model = None
        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]
        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")
        citation = textwrap.dedent(
            """            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )
        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url()
            if paddlenlp.trainer.integrations.is_wandb_available and wandb.run is not None
            else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )
        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self) -> Sampler:
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.dataset_world_size
            * self.args.gradient_accumulation_steps
        )
        return RepeatRandomSampler(
            indexes=self.indexes,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )
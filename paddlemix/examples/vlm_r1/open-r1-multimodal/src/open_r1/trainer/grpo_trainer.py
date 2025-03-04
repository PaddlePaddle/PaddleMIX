import ast
import copy
import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sequence, Dict
from dataclasses import dataclass, field
from paddlemix.utils.log import logger

import paddle
import paddle.nn.functional as F
import paddlenlp
from paddlenlp.data import DataCollatorForSeq2Seq
import datasets
import PIL.Image
from packaging import version

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

@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    r"""
    Data collator that supports VLMs.

    Features should contain input_ids, attention_mask, labels, and optionally contain images and videos.
    """

    template: Optional["TEMPLATES"] = None
    processor: Optional["ProcessorMixin"] = None

    def __post_init__(self):
        if self.template is None:
            raise ValueError("Template is required for MultiModalDataCollator.")

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "paddle.Tensor"]:
        batch_images, batch_videos, batch_imglens, batch_vidlens, batch_input_ids = [], [], [], [], []
        for feature in features:
            images = feature.pop("images", None) or []
            videos = feature.pop("videos", None) or []
            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_imglens.append(len(images))
            batch_vidlens.append(len(videos))
            batch_input_ids.append(feature["input_ids"])                
        

        if (
            self.processor is not None and sum(batch_imglens) == 0 and sum(batch_vidlens) == 0
        ):  
            fake_messages = [{"role": "user", "content": IMAGE_PLACEHOLDER}]
            fake_images = [Image.new("RGB", (64, 64), (255, 255, 255))]
            fake_messages = self.template.mm_plugin.process_messages(fake_messages, fake_images, [], self.processor)
            fake_input_ids = self.tokenizer.encode(fake_messages[0]["content"], add_special_tokens=False)
            fake_input_ids, _ = self.template.mm_plugin.process_token_ids(
                fake_input_ids, None, fake_images, [], self.tokenizer, self.processor
            )

            if len(fake_input_ids) != 0:
                if self.tokenizer.padding_side == "right":
                    features[0]["input_ids"] = features[0]["input_ids"]+ fake_input_ids["input_ids"]
                    features[0]["attention_mask"] = features[0]["attention_mask"] + [0] * len(fake_input_ids["input_ids"])
                    features[0]["labels"] = features[0]["labels"] + [IGNORE_INDEX] * len(fake_input_ids["input_ids"])
                else:
                    features[0]["input_ids"] = fake_input_ids["input_ids"] + features[0]["input_ids"]
                    features[0]["attention_mask"] = [0] * len(fake_input_ids["input_ids"]) + features[0]["attention_mask"]
                    features[0]["labels"] = [IGNORE_INDEX] * len(fake_input_ids["input_ids"]) + features[0]["labels"]

            batch_images = fake_images
            batch_imglens[0] = 1
            batch_input_ids[0] = features[0]["input_ids"]

        mm_inputs = self.template.mm_plugin.get_mm_inputs(
            batch_images, batch_videos, batch_imglens, batch_vidlens, batch_input_ids, self.processor
        )
        if "token_type_ids" in mm_inputs:
            token_type_ids = mm_inputs.pop("token_type_ids")
            for i, feature in enumerate(features):
                feature["token_type_ids"] = token_type_ids[i]

        features: Dict[str, "paddle.Tensor"] = super().__call__(features)

        if self.model is not None and hasattr(self.model, "get_rope_index"):  # for qwen2vl mrope
            features["position_ids"], features["rope_deltas"] = self.model.get_rope_index(
                input_ids=features["input_ids"],
                image_grid_thw=mm_inputs.get("image_grid_thw", None),
                video_grid_thw=mm_inputs.get("video_grid_thw", None),
                attention_mask=features["attention_mask"],
            )

        if "cross_attention_mask" in mm_inputs:  # for mllama inputs when pad_to_multiple_of is enabled
            cross_attention_mask = mm_inputs.pop("cross_attention_mask")
            seq_len = features["input_ids"].size(1)
            orig_len = cross_attention_mask.size(1)
            mm_inputs["cross_attention_mask"] = F.pad(cross_attention_mask, (0, 0, 0, 0, 0, seq_len - orig_len))

        features.update(mm_inputs)
        if isinstance(features.get("pixel_values"), list):  # for pixtral inputs
            features = features.data  # use default_collate() instead of BatchEncoding.to()

        if "image_bound" in features:  # for minicpmv inputs
            bsz, seq_length = features["input_ids"].shape
            features["position_ids"] = paddle.arange(seq_length).long().repeat(bsz, 1)
            return {"data": features, "input_ids": features["input_ids"], "labels": features["labels"]}

        return features


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
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        dtype: str = "bfloat16",
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation

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
        self.generation_config = paddlenlp.generation.GenerationConfig(
            use_cache=model_init_kwargs['use_cache'],
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            num_return_sequences=self.num_generations,
            temperature=1.0,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
        )
        self.beta = args.beta
        self.epsilon = args.epsilon
        self._metrics = defaultdict(list)
        data_collator = None

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

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, paddlenlp.transformers.PretrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(
                    reward_func, evaluation_mode=True
                )

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_per_token_logps(
        self, model, input_ids, attention_mask, pixel_values, image_grid_thw
    ):
        logits = model(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        ).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            # log_probs = logits_row.log_softmax(dim=-1)
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
        self, inputs: dict[str, Union[paddle.Tensor, Any]]
    ) -> dict[str, Union[paddle.Tensor, Any]]:
        return inputs

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        device = inputs['pixel_values'].place

        # images = []
        # for x in inputs:
        #     if "image" in x:
        #         img = x["image"]
        #     else:
        #         img = PIL.Image.open(x["image_path"])
        #     w, h = img.size
        #     if w < 28 or h < 28:
        #         if w < h:
        #             new_w = 28
        #             new_h = int(h * (28 / w))
        #         else:
        #             new_h = 28
        #             new_w = int(w * (28 / h))
        #         img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
        #     images.append(img)

        prompt_ids, prompt_mask = (
            inputs["input_ids"],
            inputs["attention_mask"],
        )
        # repeat
        return_seq_length = self.generation_config.num_return_sequences

        # TODO
        _, pixel_seq_len,pixel_dim = inputs["pixel_values"].shape
        inputs["pixel_values"] = inputs["pixel_values"].reshape([-1,pixel_dim])

        _, _,g2 = inputs["image_grid_thw"].shape
        inputs["image_grid_thw"] = inputs["image_grid_thw"].reshape([-1,g2])

        inputs["pixel_values"] = paddle.repeat_interleave(inputs["pixel_values"],repeats=return_seq_length,axis=0)
        inputs["image_grid_thw"] = paddle.repeat_interleave(inputs["image_grid_thw"],repeats=return_seq_length,axis=0)
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if paddle.distributed.is_initialized():
            with paddle.no_grad():
                completion_ids = unwrap_model(model).generate(**inputs,generation_config=self.generation_config)[0]
        else:
            with paddle.no_grad():
                completion_ids = model.generate(**inputs,generation_config=self.generation_config)[0]
        prompt_length = prompt_ids.shape[1]
        prompt_mask = prompt_mask.repeat_interleave(
            repeats=self.num_generations, axis=0
        )
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = paddle.full(shape=(is_eos.shape[0],), fill_value=is_eos.shape[1], dtype="int64")

        # logger.info(completion_ids)
        # if paddle.distributed.is_initialized() and paddle.distributed.get_rank()==0:
        #     import pdb;pdb.set_trace()
        # paddle.distributed.barrier()
        
        eos_idx[is_eos.astype("bool").any(axis=1)] = is_eos.astype(dtype="int64").argmax(axis=1)[is_eos.astype("bool").any(axis=1)]

        sequence_indices = paddle.arange(end=is_eos.shape[1]).expand(
            shape=[is_eos.shape[0], -1]
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(axis=1)).astype(
            dtype="int64"
        )
        attention_mask = paddle.concat(x=[prompt_mask, completion_mask], axis=1)
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        prompt_completion_ids = paddle.concat([paddle.repeat_interleave(prompt_ids,repeats=self.num_generations,axis=0),completion_ids],axis=1)
        per_token_logps = self._get_per_token_logps(
            model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
        )
        per_token_logps = per_token_logps[:, prompt_length - 1 :]
        with paddle.no_grad():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    pixel_values,
                    image_grid_thw,
                )
            else:
                # TODO
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model,
                        prompt_completion_ids,
                        attention_mask,
                        pixel_values,
                        image_grid_thw,
                    )
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
        per_token_kl = (
            paddle.exp(x=ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )
        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # apply chat template
        completions = [
            [{"role": "assistant", "content": completion}]
            for completion in completions
        ]
        rewards_per_func = paddle.zeros(shape=[len(prompt_ids)*self.num_generations, len(self.reward_funcs)])
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            prompts = self.processing_class.batch_decode(prompt_ids,skip_special_tokens=True)
            solution = [ast.literal_eval(self.processing_class.batch_decode(inputs['labels'])[0])]
             # self.processing_class.batch_decode(prompt_ids,skip_special_tokens=True)
            if isinstance(reward_func, paddlenlp.transformers.PretrainedModel):
                if is_conversational(inputs[0]):
                    messages = [
                        {"messages": p + c} for p, c in zip(prompts, completions)
                    ]
                    texts = [
                        apply_chat_template(x, reward_processing_class)["text"]
                        for x in messages
                    ]
                else:
                    texts = [(p + c) for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with paddle.no_grad():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                reward_kwargs = {
                    "prompts":prompts*self.num_generations,
                    "solution":solution* self.num_generations
                }
                output_reward_func = reward_func(
                    completions=completions, **reward_kwargs
                )
                rewards_per_func[:,i] = paddle.to_tensor(
                    data=output_reward_func, dtype="float32", place=device
                )
        
        rewards = rewards_per_func.sum(axis=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view([-1, self.num_generations]).mean(axis=1)
        std_grouped_rewards = rewards.view([-1, self.num_generations]).std(axis=1)


        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            repeats=self.num_generations, axis=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            repeats=self.num_generations, axis=0
        )
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 0.0001)

        coef_1 = paddle.exp(x=per_token_logps - per_token_logps.detach())
        coef_2 = paddle.clip(x=coef_1, min=1 - self.epsilon, max=1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(axis=1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(axis=1)
        per_token_loss = -paddle.min(paddle.stack([per_token_loss1, per_token_loss2]),axis=0)
        if self.beta > 0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (
            (per_token_loss * completion_mask.astype('float32')).sum(axis=1) / completion_mask.sum(axis=1).astype('float32')
        ).mean()

        global_completion_length_list = all_gather(completion_mask.sum(axis=1))
        completion_length = (
            paddle.to_tensor(global_completion_length_list)
            .astype(dtype="float32")
            .mean()
            .item()
        )
        self._metrics["completion_length"].append(completion_length)

        global_rewards_per_func_list = all_gather(rewards_per_func)
        reward_per_func  = (
            paddle.to_tensor(global_rewards_per_func_list)
            .astype(dtype="float32")
            .mean(axis=0)
        )
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, paddlenlp.transformers.PretrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(
                reward_per_func[i].mean().item()
            )

        global_reward = all_gather(rewards)
        self._metrics["reward"].append(
           paddle.to_tensor(global_reward).mean().item()
        )

        global_reward_std = all_gather(std_grouped_rewards)
        self._metrics["reward_std"].append(
           paddle.to_tensor(global_reward_std).mean().item()
        )
        mean_kl = (
            (per_token_kl * completion_mask.astype("float32")).sum(axis=1) / completion_mask.astype("float32").sum(axis=1)
        ).mean()

        global_kl = all_gather(mean_kl)
        self._metrics["kl"].append(
            paddle.to_tensor(global_kl).mean().item()
        )
        is_clipped = (per_token_loss1 < per_token_loss2).astype(dtype="float32")
        clip_ratio = (is_clipped * completion_mask.astype("float32")).sum() / completion_mask.astype("float32").sum()
        
        global_clip_ratio = all_gather(clip_ratio)
        self._metrics["clip_ratio"].append(
            paddle.to_tensor(global_clip_ratio).mean().item()
        )
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

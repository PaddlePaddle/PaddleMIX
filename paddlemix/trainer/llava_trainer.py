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


import inspect
from typing import List, Optional

import paddle
import paddle.nn as nn
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
    DygraphShardingOptimizer,
)
from paddlenlp.trainer.trainer import Trainer, has_length, TrainerCallback
from paddlenlp.trainer.trainer_utils import ShardingOption
from paddlenlp.utils.log import logger

"""utils for log"""
import json
import os
import time

import numpy as np
import paddle
import paddle.distributed.fleet as fleet

from paddlenlp.trainer.plugins.timer import get_timers


PADDLE_EMA_WEIGHTS_NAME = "ema_state.pdparams"


def get_memory_info():
    """get_memory_info"""
    divisor = 2**30
    return (
        paddle.device.cuda.memory_allocated() / divisor,
        paddle.device.cuda.max_memory_allocated() / divisor,
        paddle.device.cuda.memory_reserved() / divisor,
        paddle.device.cuda.max_memory_reserved() / divisor,
    )


class Statistical(object):
    """Statistical
    """
    def __init__(self, buffer_size, skip_step, approximate=False):
        self.step = 0
        self.local_step = 0
        self.is_first_period = True

        assert skip_step < buffer_size
        self.buffer_size = buffer_size
        self.skip_step = skip_step
        self.approximate = approximate

        self.local_tokens = np.zeros([buffer_size], dtype=np.int64)
        self.local_samples = np.zeros([buffer_size], dtype=np.int64)
        self.efficient_tokens = np.zeros([buffer_size], dtype=np.int64)
        self.tokens = np.zeros([buffer_size], dtype=np.int64)
        self.samples = np.zeros([buffer_size], dtype=np.int64)
        self.durations = np.zeros([buffer_size], dtype=np.float64)
        if int(os.getenv("PADDLE_TRAINERS_NUM", "1")) > 1:
            assert (
                paddle.distributed.is_initialized()
            ), "please call fleet.init() before"
            if hasattr(fleet.fleet, "_hcg"):
                hcg = fleet.get_hybrid_communicate_group()
    
                dp_size = hcg.get_data_parallel_world_size() 
                sharding_size = hcg.get_sharding_parallel_world_size()
                self.dp_size = dp_size * sharding_size 
                self.world_size = paddle.distributed.get_world_size()
                self.groups = []
                if dp_size > 1:
                    self.groups.append(hcg.get_data_parallel_group())
                if sharding_size > 1:
                    self.groups.append(hcg.get_sharding_parallel_group())
            else:
                self.dp_size = paddle.distributed.get_world_size()
                self.world_size = self.dp_size
                self.groups = []

        else:
            self.dp_size = 1
            self.world_size = 1
            self.groups = []

    def reset(self):
        """reset
        """
        if self.local_step > 0:
            self.is_first_period = False
        self.local_step = 0
        self.local_tokens.fill(0)
        self.local_samples.fill(0)
        self.efficient_tokens.fill(0)
        self.tokens.fill(0)
        self.samples.fill(0)
        self.durations.fill(0.0)

    def _get_global(self, tokens_num, batch_size):
        if self.dp_size == 1 or self.approximate:
            return tokens_num * self.dp_size, batch_size * self.dp_size

        timers = get_timers()
        if timers:
            timers("all-reduce-token-bz").start()
        x = paddle.to_tensor([tokens_num, batch_size], dtype=paddle.int64)
        for g in self.groups:  
            paddle.distributed.stream.all_reduce(x, group=g, sync_op=True, use_calc_stream=True)
        if timers:
            timers("all-reduce-token-bz").stop()
        return x.numpy().tolist() 

    def add(self, start_time, efficient_tokens_num, tokens_num, batch_size):
        """add """
        assert (
            self.local_step < self.buffer_size
        ), "the step number exceeds the ckpt saving interval"
        global_tokens_num, global_batch_size = self._get_global(tokens_num, batch_size)
        global_efficient_tokens_num, _ = self._get_global(efficient_tokens_num, batch_size)

        duration = time.time() - start_time
        self.durations[self.local_step] = duration
        self.local_tokens[self.local_step] = tokens_num
        self.local_samples[self.local_step] = batch_size
        self.efficient_tokens[self.local_step] = global_efficient_tokens_num
        self.tokens[self.local_step] = global_tokens_num
        self.samples[self.local_step] = global_batch_size
        self.step += 1
        self.local_step += 1
        return global_tokens_num, global_batch_size

    def get_tokens_per_sec_per_card(self):
        """get_tokens_per_sec_per_card"""
        return (
            self.tokens[self.local_step - 1]
            / self.durations[self.local_step - 1]
            / self.world_size
        )
        
    def get_efficient_tokens_per_sec_per_card(self):
        """get_efficient_tokens_per_sec_per_card"""
        return (
            self.efficient_tokens[self.local_step - 1]
            / self.durations[self.local_step - 1]
            / self.world_size
        )

    def get_avg_tokens_per_sec_per_card(self):
        """get_avg_tokens_per_sec_per_card"""
        if self.step <= self.skip_step:
            return self.get_tokens_per_sec_per_card()

        start = self.skip_step if self.is_first_period else 0
        return (
            np.sum(self.tokens[start : self.local_step])
            / np.sum(self.durations[start : self.local_step])
            / self.world_size
        )
    
    def get_avg_efficient_tokens_per_sec_per_card(self):
        """get_avg_tokens_per_sec_per_card"""
        if self.step <= self.skip_step:
            return self.get_efficient_tokens_per_sec_per_card()

        start = self.skip_step if self.is_first_period else 0
        return (
            np.sum(self.efficient_tokens[start : self.local_step])
            / np.sum(self.durations[start : self.local_step])
            / self.world_size
        )

    def get_samples_per_sec_per_card(self):
        """get_samples_per_sec_per_card"""
        return (
            self.samples[self.local_step - 1]
            / self.durations[self.local_step - 1]
            / self.world_size
        )

    def get_avg_samples_per_sec_per_card(self):
        """get_avg_samples_per_sec_per_card"""
        if self.step <= self.skip_step:
            return self.get_samples_per_sec_per_card()

        start = self.skip_step if self.is_first_period else 0
        return (
            np.sum(self.samples[start : self.local_step])
            / np.sum(self.durations[start : self.local_step])
            / self.world_size
        )

    def get_tokens(self):
        """get_tokens"""
        return self.tokens[: self.local_step]

    def get_efficient_tokens(self):
        """get_efficient_tokens"""
        return self.efficient_tokens[: self.local_step]

    def get_samples(self):
        """get_samples"""
        return self.samples[: self.local_step]

    def get_durations(self):
        """get_durations"""
        return self.durations[: self.local_step]

    def get_total_tokens_per_card(self):
        """get_total_tokens_per_card"""
        return np.sum(self.tokens[: self.local_step]) / self.world_size
    
    def get_total_efficient_tokens_per_card(self):
        """get_total_efficient_tokens_per_card"""
        return np.sum(self.efficient_tokens[: self.local_step]) / self.world_size

    def get_total_samples_per_card(self):
        """get_total_samples_per_card"""
        return np.sum(self.samples[: self.local_step]) / self.world_size

    def get_skip_duration(self):
        """get_skip_duration"""
        if self.is_first_period:
            return np.sum(self.durations[: min(self.local_step, self.skip_step)])
        else:
            return 0.0


    def get_result(self):
        """get_result"""
        runtime = np.sum(self.durations[: self.local_step])
        local_samples = np.sum(self.local_samples[: self.local_step])
        local_tokens = np.sum(self.local_tokens[: self.local_step])
        global_samples = np.sum(self.samples[: self.local_step])
        global_tokens = np.sum(self.tokens[: self.local_step])
        global_efficient_tokens = np.sum(self.efficient_tokens[: self.local_step])

        alloc, max_alloc, reserved, max_reserved = get_memory_info()
        result = {
            f'runtime': runtime,
            f"local_samples": local_samples,
            f"global_samples": global_samples,
            f"local_tokens": local_tokens,
            f"global_tokens": global_tokens,
            f"global_efficient_tokens": global_efficient_tokens,
            f"samples_per_sec_per_card": self.get_samples_per_sec_per_card(),
            f"avg_samples_per_sec_per_card": self.get_avg_samples_per_sec_per_card(),
            f"efficient_tokens_per_sec_per_card": self.get_efficient_tokens_per_sec_per_card(),
            f"tokens_per_sec_per_card": self.get_tokens_per_sec_per_card(),
            f"avg_tokens_per_sec_per_card": self.get_avg_tokens_per_sec_per_card(),
            f"avg_efficient_tokens_per_sec_per_card": self.get_avg_efficient_tokens_per_sec_per_card(),
            "memory_allocated_gb": alloc,
            "max_memory_allocated_gb": max_alloc,
            "memory_reserved_gb": reserved,
            "max_memory_reserved_gb": max_reserved,
        }
        
        return result


class BenchmarkCallback(TrainerCallback):
    """
    used to benchmark the training process.
    """

    ACC_SAMPLES = "acc_global_samples"
    ACC_TOKENS = "acc_global_tokens"

    def __init__(self, trainer, save_steps, skip_step):
        super().__init__()
        self.trainer = trainer
        self.state = Statistical(save_steps, skip_step)
        self.efficient_token_count = 0
        self.cur_tokens = 0
        self.cur_samples = 0

    def set_save_time(self, save_time):
        """
        set the time to save model.
        """
        self.save_time = save_time

    def on_train_begin(self, args, state, control, **kwargs):
        """
        record the start time of training.
        """
        if state.trial_params is None:
            state.trial_params = {}

        if self.ACC_SAMPLES not in state.trial_params:
            state.trial_params[self.ACC_SAMPLES] = 0
        if self.ACC_TOKENS not in state.trial_params:
            state.trial_params[self.ACC_TOKENS] = 0

        if paddle.distributed.is_initialized():
            if hasattr(fleet.fleet, "_hcg"):
                dp_degree = fleet.get_hybrid_communicate_group().get_data_parallel_world_size()
            else:
                dp_degree = paddle.distributed.get_world_size()
            assert dp_degree <= 1, f"data_parallel_degree should be 1 but got {dp_degree}"
            paddle.distributed.barrier()
        self.end_save_time = time.time()

    def on_epoch_begin(self, args, state, control, **kwargs):
        """
        record the start time of epoch.
        """
        self.epoch_start = time.time()
        self.batch_start = time.time()

    def on_substep_end(self, args, state, control, **kwargs):
        """
        record the start time of batch.
        """
        model = kwargs["model"]
        batch_size, seq_length, _ = model.input_shape
        self.efficient_token_count = model.efficient_token_count
        self.cur_tokens += batch_size * seq_length
        self.cur_samples += batch_size

    def on_step_end(self, args, state, control, **kwargs):
        """
        record the start time of sub-batch.
        """
        self.on_substep_end(args, state, control, **kwargs)

        tokens, batches = self.state.add(
            start_time=self.batch_start, efficient_tokens_num=self.efficient_token_count, tokens_num=self.cur_tokens, batch_size=self.cur_samples
        )
        state.trial_params[self.ACC_SAMPLES] += batches
        state.trial_params[self.ACC_TOKENS] += tokens

        self.efficient_token_count = 0
        self.cur_tokens = 0
        self.cur_samples = 0

        self.batch_start = time.time()
        if control.should_log:
            self.maybe_log_save_evaluate_start = time.time()

    def on_save(self, args, state, control, **kwargs):
        """
        record the infomation of saving model.
        """
        pass
        # end_save_time = time.time()

        # train_time_with_save = end_save_time - self.end_save_time
        # train_time_without_save = train_time_with_save - self.save_time

        # total_tokens = self.state.get_total_tokens_per_card()
        # total_efficient_tokens = self.state.get_total_efficient_tokens_per_card()
        # total_samples = self.state.get_total_samples_per_card()
        # world_size = paddle.distributed.get_world_size()

        # skip_time = self.state.get_skip_duration()

        # token_speed_without_save = total_tokens / (train_time_without_save - skip_time)
        # token_speed_with_save = total_tokens / (train_time_with_save - skip_time)

        # one_day_billion_tokens_without_save = token_speed_without_save * world_size * 8.64e-5
        # one_day_billion_tokens_with_save = token_speed_with_save * world_size * 8.64e-5

        # sample_speed_without_save = total_samples / (train_time_without_save - skip_time)
        # sample_speed_with_save = total_samples / (train_time_with_save - skip_time)
        # one_day_billion_samples_without_save = sample_speed_without_save * world_size * 8.64e-5
        # one_day_billion_samples_with_save = sample_speed_with_save * world_size * 8.64e-5

        # logs = {
        #     "global_save_step": state.global_step,
        #     "train_time_sec_without_save": train_time_without_save,
        #     "train_time_with_save": train_time_with_save,
        #     "save_ckpt_time_sec": self.save_time,
        #     "average_tokens_per_sec_per_card_without_save": token_speed_without_save,
        #     "average_tokens_per_sec_per_card_with_save": token_speed_with_save,
        #     "one_day_billion_tokens_without_save": one_day_billion_tokens_without_save,
        #     "one_day_billion_tokens_with_save": one_day_billion_tokens_with_save,
        #     "average_samples_per_sec_per_card_without_save": sample_speed_without_save,
        #     "average_samples_per_sec_per_card_with_save": sample_speed_with_save,
        #     "one_day_billion_samples_without_save": one_day_billion_samples_without_save,
        #     "one_day_billion_samples_with_save": one_day_billion_samples_with_save,
        # }

        # self._log(logs)
        # self.state.reset()
        # self.end_save_time = time.time()
        # self.trainer._globalstep_last_start_time = self.end_save_time
        # self.batch_start = self.end_save_time

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        record the infomation of a logging step.
        """
        logs.update(self.state.get_result())
        logs[self.ACC_SAMPLES] = state.trial_params[self.ACC_SAMPLES]
        logs[self.ACC_TOKENS] = state.trial_params[self.ACC_TOKENS]
        self._log(logs)

    def _log(self, logs):
        """
        record the information accurately and neatly.
        """
        logs_str = []
        logs_str = []
        for k, v in logs.items():
            if isinstance(v, float):
                if abs(v) < 1e-3:
                    v = f"{v:e}"
                elif abs(v) > 100:
                    v = f"{v:.04f}"
                else:
                    v = f"{v:.06f}"
            logs_str.append(f"{k}: {v}")
        logger.info(", ".join(logs_str))
def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]
    num_indices_per_chunk = len(indices) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [(0) for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")
    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])
    mm_shuffle = [
        mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)
    ]
    lang_shuffle = [
        lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)
    ]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]
    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = paddle.randperm(n=len(megabatches))
    megabatches = [megabatches[i] for i in megabatch_indices]
    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))
    return [[i] for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    indices = paddle.randperm(n=len(lengths))
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]
    return [[i] for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(paddle.io.Sampler):
    """
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


class LLaVATrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.benchmark_callback = BenchmarkCallback(self, self.args.save_steps, skip_step=self.args.benchmark_skip_steps)
        self.benchmark_callback = BenchmarkCallback(self, 12, 1)
        
        self.add_callback(self.benchmark_callback)
    def _get_train_sampler(self) -> Optional[paddle.io.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self, lr_scheduler=None):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """

        opt_model = self.model

        for p in self.model.llama.mm_projector.parameters():
            p.stop_gradient = not True

        if self.optimizer is None:
            decay_parameters = [
                p.name for n, p in opt_model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])
            ]

            if self.args.mm_projector_lr is not None:

                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (p.name in decay_parameters and n not in projector_parameters and not p.stop_gradient)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                p.name not in decay_parameters
                                and n not in projector_parameters
                                and not p.stop_gradient
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if p.name in decay_parameters and n in projector_parameters and not p.stop_gradient
                        ],
                        "weight_decay": self.args.weight_decay,
                        "learning_rate": self.args.mm_projector_lr / self.args.learning_rate,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if p.name not in decay_parameters and n in projector_parameters and not p.stop_gradient
                        ],
                        "weight_decay": 0.0,
                        "learning_rate": self.args.mm_projector_lr / self.args.learning_rate,
                    },
                ]

            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if p.name in decay_parameters and not p.stop_gradient
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if p.name not in decay_parameters and not p.stop_gradient
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if hasattr(optimizer_cls, "_create_master_weight") and self.args.fp16_opt_level == "O2":
                optimizer_kwargs["multi_precision"] = True

            def is_new_version_sharding_stage1_optimizer():
                signature_keys = set(inspect.signature(DygraphShardingOptimizer).parameters.keys())
                return "inner_optimizer_class" not in signature_keys

            if ShardingOption.SHARD_OP in self.args.sharding and not is_new_version_sharding_stage1_optimizer():
                # for backward compatibility.
                # this call will raise, if sharding stage1 is supported in HybridParallelOptimizer,
                # in which case, the logic follows will handle it
                self.optimizer = DygraphShardingOptimizer(
                    hcg=fleet.get_hybrid_communicate_group(),
                    user_defined_strategy=None,
                    params=optimizer_grouped_parameters,
                    inner_optimizer_class=optimizer_cls,
                    learning_rate=self.lr_scheduler if lr_scheduler is None else lr_scheduler,
                    apply_decay_param_fun=None,
                    weight_decay=self.args.weight_decay,
                    grad_clip=nn.ClipGradByGlobalNorm(self.args.max_grad_norm)
                    if self.args.max_grad_norm > 0
                    else None,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(
                    learning_rate=self.lr_scheduler if lr_scheduler is None else lr_scheduler,
                    apply_decay_param_fun=None,
                    parameters=optimizer_grouped_parameters,
                    weight_decay=self.args.weight_decay,
                    grad_clip=nn.ClipGradByGlobalNorm(self.args.max_grad_norm)
                    if self.args.max_grad_norm > 0
                    else None,
                    **optimizer_kwargs,
                )
        return self.optimizer

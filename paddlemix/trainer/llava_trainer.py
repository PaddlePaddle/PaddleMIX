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
from paddlenlp.trainer.trainer import Trainer, has_length
from paddlenlp.trainer.trainer_utils import ShardingOption
from paddlenlp.trainer.integrations import TrainerCallback
from paddlenlp.trainer import PrinterCallback, ProgressCallback
from paddlenlp.utils.log import logger

class BenchmarkCallback(TrainerCallback):
    def __init__(self, benchmark=False):
        self.benchmark = benchmark

    def on_train_begin(self, args, state, control, **kwargs):
        # assert args.gradient_accumulation_steps == 1 and not args.do_eval and not args.do_predict
        if self.benchmark:
            pass

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.benchmark:
            pass

    def on_step_begin(self, args, state, control, **kwargs):
        if self.benchmark:
            pass

    def on_step_end(self, args, state, control, **kwargs):
        if self.benchmark:
            pass
    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.benchmark:
            if logs is not None and "interval_samples_per_second" in logs:
                max_mem_reserved_msg = ""
                max_mem_allocated_msg = ""

                if paddle.device.is_compiled_with_cuda():
                    max_mem_reserved_msg = (
                        f"max_mem_reserved: {paddle.device.cuda.max_memory_reserved() // (1024 ** 2)} MB,"
                    )
                    max_mem_allocated_msg = (
                        f"max_mem_allocated: {paddle.device.cuda.max_memory_allocated() // (1024 ** 2)} MB"
                    )

                logger.info(
                    "global step %d, loss: %.5f, ips: %.5f, %s %s"
                    % (
                        state.global_step,
                        logs["loss"],
                        logs["interval_samples_per_second"],
                        max_mem_reserved_msg,
                        max_mem_allocated_msg,
                    )
                )
            else:
                logger.info(logs)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.benchmark:
            pass


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
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        if self.args.benchmark:
            self.add_callback(
                BenchmarkCallback(
                    benchmark=self.args.benchmark
                )
            )
            if self.args.benchmark:
                if self.args.disable_tqdm:
                    self.pop_callback(PrinterCallback)
                else:
                    self.pop_callback(ProgressCallback)
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

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# code is heavily based on https://github.com/tianweiy/DMD2

import contextlib
import json
import os
import random
import shutil
import sys
import time
import warnings
from collections import OrderedDict
from typing import Optional

import numpy as np
import paddle
import paddle.distributed.fleet as fleet
import paddle.nn as nn
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_stage2 import (
    GroupShardedStage2,
)
from paddlenlp.trainer import Trainer as NLPTrainer
from paddlenlp.trainer import get_last_checkpoint
from paddlenlp.trainer.trainer import (
    COMPUTE_GENERATOR_GRADIENT,
    LORA_WEIGHTS_NAME,
    LOSS_INF_ERROR,
    LOSS_NAN_ERROR,
    OPTIMIZER_NAME,
    PADDLE_WEIGHTS_INDEX_NAME,
    PADDLE_WEIGHTS_NAME,
    PREFIX_CHECKPOINT_DIR,
    PREFIX_WEIGHTS_NAME,
    SCALER_NAME,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
    Dict,
    DistributedBatchSampler,
    GroupShardedOptimizerStage2,
    HybridParallelOptimizer,
    IntervalStrategy,
    LoKrModel,
    LoRAModel,
    NlpDistributedBatchSampler,
    OptimizerNames,
    PrefixModelForCausalLM,
    PretrainedModel,
    QuantizationLinear,
    ReFTModel,
    ShardingOption,
    TrainerState,
    TrainOutput,
    VeRAModel,
    _add_variant,
    _obtain_optimizer_parameters_list,
    autocast,
    broadcast_dataset_rank0_model,
    core,
    dist,
    distributed_file,
    distributed_isfile,
    fused_allreduce_gradients,
    get_env_device,
    get_scheduler,
    has_length,
    in_auto_parallel_align_mode,
    is_paddle_cuda_available,
    load_sharded_checkpoint,
    logger,
    mix_precision_utils,
    reshard_util,
    should_skip_data,
    speed_metrics,
    split_inputs_sequence_dim,
    split_inputs_sequence_dim_load_balance,
    split_parallel_config,
    strtobool,
    unwrap_model,
    visual,
)
from tqdm import tqdm

from .wandb_callback import WandbCallback


class DMD2Trainer(NLPTrainer):
    """DMD2 Trainer"""

    def __init__(
        self,
        guidance_model=None,
        optimizers_guidance=(None, None),
        dmd2_args=None,
        unified_model=None,
        accelerator=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dmd2_args = dmd2_args
        # self.is_in_train = False
        # # self.do_grad_scaling = args.fp16

        # # memory metrics - must set up as early as possible
        # self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        # self._memory_tracker.start()

        # Seed must be set before instantiating the model when using model

        # self.model_wrapped = model
        # self.model = model
        self.guidance_model_wrapped = guidance_model
        self.guidance_model = guidance_model
        self.unified_model = unified_model
        # self.criterion = criterion

        # self.compute_metrics = compute_metrics
        # self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        # self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_guidance, self.lr_scheduler_guidance = optimizers_guidance

        if self.args.fp16 or self.args.bf16:
            # set do_grad_scaling, enable_autocast_context_manager
            # self._wrap_amp_model(args, model)
            self._wrap_amp_model(self.args, guidance_model)

        if self.args.recompute:

            def fn(layer):
                if hasattr(layer, "enable_recompute") and (
                    layer.enable_recompute is False or layer.enable_recompute == 0
                ):
                    layer.enable_recompute = True

            # model.apply(fn)
            guidance_model.apply(fn)

        self.wandb_callback = WandbCallback(dmd2_args, accelerator)
        self.add_callback(self.wandb_callback)

    def _load_from_checkpoint(self, resume_from_checkpoint=None):
        """load state_dict from_checkpoint, Only load model state dict.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. Only load model state dict.
        """
        self.runtime_timer.start("checkpoint loading time")
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            uc_async_save = self.args.unified_checkpoint and "async_save" in self.args.unified_checkpoint_config
            resume_from_checkpoint = get_last_checkpoint(
                self.args.output_dir, signal_folder=self.args.output_signal_dir, uc_async_save=uc_async_save
            )
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({self.args.output_dir})")

        if self.args.unified_checkpoint:
            if resume_from_checkpoint is not None:
                use_unified_checkpoint = False
                if self.is_unified_checkpoint(resume_from_checkpoint):
                    use_unified_checkpoint = True
                else:
                    logger.info("Loading origin checkpoint, the next checkpoint will be saved as unified checkpoint")

                if use_unified_checkpoint:
                    self.unified_checkpoint_handler.load_unified_checkpoint(
                        self.model,
                        resume_from_checkpoint,
                    )
                    logger.info(f"Loading model from {resume_from_checkpoint} using unified checkpoint.")
                    self.runtime_timer.stop()
                    return

        if (
            isinstance(self.model, LoRAModel)
            or isinstance(self.model, PrefixModelForCausalLM)
            or isinstance(self.model, VeRAModel)
            or isinstance(self.model, LoKrModel)
            or isinstance(self.model, ReFTModel)
        ):
            self._load_from_peft_checkpoint(resume_from_checkpoint)
            self.runtime_timer.stop()
            return

        weight_name = PADDLE_WEIGHTS_NAME
        weight_index_name = PADDLE_WEIGHTS_INDEX_NAME  # currently set paddle as default, do not support safetensors.

        if self.args.should_load_sharding_stage1_model:
            state_dict = self.sharding_io.load_state_dict_from_checkpoint_with_reshard(
                resume_from_checkpoint,
                base_weight_name=weight_name,
                model_wrapped=self.model_wrapped,
            )
            old_state_dict = self.model.state_dict()
            new_state_dict = {}
            for k, v in state_dict.items():
                if k not in old_state_dict or id(v) != id(old_state_dict[k]):
                    new_state_dict[k] = v
            self.model.set_state_dict(new_state_dict)
        else:
            if resume_from_checkpoint is not None and (self.args.dataset_rank == 0 or self.args.use_expert_parallel):

                weights_file = os.path.join(
                    resume_from_checkpoint, _add_variant(weight_name, self.args.weight_name_suffix)
                )
                weights_index_file = os.path.join(
                    resume_from_checkpoint, _add_variant(weight_index_name, self.args.weight_name_suffix)
                )

                if not any(
                    os.path.isfile(f)
                    for f in [
                        weights_file,
                        weights_index_file,
                    ]
                ):
                    raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint} -- {weights_file}")

                logger.info(f"Loading model from {resume_from_checkpoint} .")

                if os.path.isfile(weights_file):
                    # We load the model state dict on the CPU to avoid an OOM error.
                    state_dict = paddle.load(weights_file, return_numpy=True)
                    # If the model is on the GPU, it still works!
                    self._set_state_dict_in_model(state_dict)
                    # release memory
                    del state_dict
                else:
                    # We load the sharded checkpoint.
                    missing_keys, unexpected_keys = load_sharded_checkpoint(
                        self.model, resume_from_checkpoint, self.args.weight_name_suffix, prefer_safe=False
                    )
                    logger.info(f"set state_dict: {missing_keys, unexpected_keys}")

            elif resume_from_checkpoint is not None:
                logger.info(f"not loading ckpt :{self.args.dataset_rank}")
        self.runtime_timer.stop()

    def _wrap_model_and_load_sharded_checkpoint(self, resume_from_checkpoint):
        # In the sharded mode, should invoke _load_from_checkpoint after _wrap_model.
        # In this mode, each sharding rank load sharded params, do not need to implement the broadcast logic.
        model, guidance_model = self._wrap_model(self.model_wrapped, self.guidance_model_wrapped)
        # guidance_model = self._wrap_model(self.guidance_model_wrapped)
        if self.sharding_io is not None:
            # the self.optimizer should be wrapped and it is done in _wrap_model
            self.sharding_io.set_optimizer(self.optimizer)
            self.sharding_io.set_optimizer(self.optimizer_guidance)
        if model is not self.model:
            self.model_wrapped = model
            self.guidance_model = guidance_model
        # Should invoke _load_from_checpoint after _load_optimizer_and_scheduler
        # because the _load_from_checkpoint method rely on the optimizer in the shareded mode.
        if resume_from_checkpoint:
            assert False
            # self._load_optimizer_and_scheduler(resume_from_checkpoint)
            # self._load_from_checkpoint(resume_from_checkpoint)
        return model, guidance_model

    def train(
        self,
        resume_from_checkpoint=None,
        ignore_keys_for_eval=None,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
        """
        args = self.args
        self.is_in_train = True

        logger.info(f"Starting training from resume_from_checkpoint : {resume_from_checkpoint}")

        # The resume_from_checkpoint could be None in some machine node.
        # Here we reset None to temp directory.
        if args.world_size > 1:
            is_resume_from_checkpoint = paddle.to_tensor([resume_from_checkpoint is not None], dtype="int32")
            paddle.distributed.all_reduce(is_resume_from_checkpoint)
            is_resume_from_checkpoint = is_resume_from_checkpoint.item()
            if is_resume_from_checkpoint > 0 and is_resume_from_checkpoint < paddle.distributed.get_world_size():
                if resume_from_checkpoint is None:
                    resume_from_checkpoint = os.path.join(self.args.output_dir, "local_tempdir")
                    if os.path.exists(resume_from_checkpoint) and self.args.local_rank == 0:
                        shutil.rmtree(resume_from_checkpoint)
                    os.makedirs(resume_from_checkpoint, exist_ok=True)
                    logger.info(f"Reset resume_from_checkpoint to temp directory : {resume_from_checkpoint}")

        train_dataloader = self.get_train_dataloader()

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.dataset_world_size
        len_dataloader = None

        if args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                f"args.max_steps must be set to a positive value if dataloader does not have a length, was {args.max_steps}"
            )

        delay_optimizer_creation = False

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if not self.args.enable_auto_parallel:
            if not self.args.should_load_sharding_stage1_model:
                self._load_from_checkpoint(resume_from_checkpoint)

            if self.args.should_load_sharding_stage1_model:
                model, guidance_model = self._wrap_model_and_load_sharded_checkpoint(resume_from_checkpoint)
            elif self.args.should_save_sharding_stage1_model:
                # In the non-sharded mode, should invoke _load_from_checkpoint before _wrap_model.
                # In this mode, the rank0 load all params and the _wrap_model implicitly broadcast params from rank0 to the other ranks.
                model, guidance_model = self._wrap_model(self.model_wrapped, self.guidance_model_wrapped)
                if self.sharding_io is not None:
                    assert delay_optimizer_creation is False, "delay_optimizer_creation should be False"
                    # the self.optimizer should be wrapped and it is done in _wrap_model
                    self.sharding_io.set_optimizer(self.optimizer)
                    self.sharding_io.set_optimizer(self.optimizer_guidance)
                # for the rest of this function `model` is the outside model, whether it was wrapped or not
                if model is not self.model:
                    self.model_wrapped = model
                    self.guidance_model_wrapped = guidance_model
                if delay_optimizer_creation:
                    self.create_optimizer_and_scheduler(num_training_steps=max_steps)
                self._load_optimizer_and_scheduler(resume_from_checkpoint)
            else:
                model, guidance_model = self._wrap_model(self.model_wrapped, self.guidance_model_wrapped)
                # for the rest of this function `model` is the outside model, whether it was wrapped or not
                if model is not self.model:
                    self.model_wrapped = model
                    self.guidance_model_wrapped = guidance_model
                if delay_optimizer_creation:
                    self.create_optimizer_and_scheduler(num_training_steps=max_steps)
                self._load_optimizer_and_scheduler(resume_from_checkpoint)
        else:
            model = self.model_wrapped
            guidance_model = self.guidance_model_wrapped
            if delay_optimizer_creation:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        logger.info(f"{self.runtime_timer.log()}")
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Total num train samples = {num_train_samples:,}")
        # per_device_trainable_numel = sum(p.numel().item() for p in model.parameters() if not p.stop_gradient)
        # TODO: Temporary fix since Tensor.numel() not supported in distributed mode
        if self.args.enable_auto_parallel:
            per_device_trainable_numel = 0
            for p in model.parameters():
                if not p.stop_gradient:
                    per_device_trainable_numel += np.prod(p._local_shape) if p.is_dist() else np.prod(p.shape)
        else:
            per_device_trainable_numel = sum(np.prod(p.shape) for p in model.parameters() if not p.stop_gradient)
        logger.debug(f"  Number of trainable parameters = {per_device_trainable_numel:,} (per device)")
        if self.args.use_hybrid_parallel:
            # todo fix for pipeline_parallel_degree
            parts_num = max(self.args.tensor_parallel_degree, 1) * max(self.args.pipeline_parallel_degree, 1)
            if parts_num > 1:
                all_reduce_dtype = "int64"
                if paddle.get_device().split(":")[0] in ["npu", "xpu"]:
                    # TODO(duanyanhui): fix when NPU all_reduce supports int64
                    all_reduce_dtype = "float32"
                trainable_numel_tensor = paddle.to_tensor(per_device_trainable_numel, dtype=all_reduce_dtype)
                paddle.distributed.all_reduce(trainable_numel_tensor)
                trainable_numel = int(trainable_numel_tensor.item()) // self.args.dataset_world_size
                if self.args.sep_parallel_degree > 0:
                    trainable_numel = trainable_numel // self.args.sep_parallel_degree
                if self.args.context_parallel_degree > 0:
                    trainable_numel = trainable_numel // self.args.context_parallel_degree
                # the numel is roughly, because the tensor parallel still hold own bias or layer_norm weight without splited
                # so, the trainable numel is a little bigger than real.
                logger.debug(f"  Number of trainable parameters = {trainable_numel:,} (all devices, roughly)")

        if isinstance(model, GroupShardedStage2):
            model._auto_refresh_trainable = False
        if isinstance(guidance_model, GroupShardedStage2):
            guidance_model._auto_refresh_trainable = False

        self.unified_model.feedforward_model = model
        self.unified_model.guidance_model = guidance_model
        return self._inner_training_loop(
            args,
            model,
            guidance_model,
            train_dataloader,
            len_dataloader,
            max_steps,
            num_train_epochs,
            num_update_steps_per_epoch,
            num_train_samples,
            resume_from_checkpoint,
            ignore_keys_for_eval,
        )

    def _inner_training_loop(
        self,
        args,
        model,
        guidance_model,
        train_dataloader,
        len_dataloader,
        max_steps,
        num_train_epochs,
        num_update_steps_per_epoch,
        num_train_samples,
        resume_from_checkpoint,
        ignore_keys_for_eval,
    ):
        start_time = time.time()
        self._globalstep_last_start_time = time.time()
        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if (
            resume_from_checkpoint is not None
            and distributed_isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            and not self.args.ignore_load_lr_and_optim
        ):
            self.state = TrainerState.load_from_json(
                distributed_file(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            )
            if self.args.world_size > 1:
                global_step_list = []
                paddle.distributed.all_gather(
                    global_step_list, paddle.to_tensor([self.state.global_step], dtype="int64")
                )
                assert (
                    paddle.sum(paddle.stack(global_step_list) - global_step_list[0]) == 0
                ), f"Error, get different globel step, please check! step list: {[x.item() for x in global_step_list]}"

            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")
            if not args.ignore_data_skip:
                if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                    train_dataloader.batch_sampler, NlpDistributedBatchSampler
                ):
                    consumed_samples = (
                        self.state.global_step
                        * args.train_batch_size
                        * args.gradient_accumulation_steps
                        * args.dataset_world_size
                    )
                    train_dataloader.batch_sampler.set_epoch(consumed_samples=consumed_samples)
                    logger.info(f"Set DistributedBatchSampler consumed_samples to {consumed_samples}")

        epoch_iterator = train_dataloader
        # steps_in_epoch = len(epoch_iterator)
        steps_in_epoch = (
            len(epoch_iterator) if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
        )
        if len_dataloader is not None:
            if self.args.gradient_accumulation_steps > len(epoch_iterator):
                logger.warning(
                    f"changing accumulation step from `{self.args.gradient_accumulation_steps}` to `{len(epoch_iterator)}` to avoid, cross epoch accumulate"
                )
                self.args.gradient_accumulation_steps = len(epoch_iterator)

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader

        self.state.max_steps = int(max_steps)
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        tr_loss = paddle.to_tensor(0.0)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        if self.args.device == "npu" and self.args.flatten_param_grads:
            from .plugins.npu_plugin import npu_accelerate_plugin

            npu_accelerate_plugin(self.optimizer)

        if self.args.ignore_data_skip:
            self.timers and self.timers("read-data").start()

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                train_dataloader.batch_sampler, DistributedBatchSampler
            ):
                train_dataloader.batch_sampler.set_epoch(epoch)

            step_control = 0  # used in loop control, reset to 0 after every step
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            step = -1
            for step in range(self.state.max_steps):
                inputs = {}
                if self.args.use_hybrid_parallel and self.args.sep_parallel_degree > 1:
                    inputs = split_inputs_sequence_dim(inputs)
                if self.args.use_hybrid_parallel and self.args.context_parallel_degree > 1:
                    inputs = split_inputs_sequence_dim_load_balance(inputs)

                # 4 channel for SD-VAE, please adapt for other autoencoders
                noise = paddle.randn(
                    [
                        self.dmd2_args.batch_size,
                        self.dmd2_args.latent_channel,
                        self.dmd2_args.latent_resolution,
                        self.dmd2_args.latent_resolution,
                    ],
                )  # device=accelerator.device)
                visual = self.state.global_step % self.dmd2_args.wandb_iters == 0
                # visual = False
                COMPUTE_GENERATOR_GRADIENT = self.state.global_step % self.dmd2_args.dfake_gen_update_ratio == 0
                self._visual = visual
                self._COMPUTE_GENERATOR_GRADIENT = COMPUTE_GENERATOR_GRADIENT

                if COMPUTE_GENERATOR_GRADIENT:
                    text_embedding = next(self.train_dataloader)
                else:
                    text_embedding = next(self.guidance_dataloader)

                if self.dmd2_args.sdxl:
                    # SDXL uses zero as the uncond_embedding
                    uncond_embedding = None
                else:
                    text_embedding = text_embedding["text_input_ids_one"].squeeze(
                        1
                    )  # actually it is tokenized text prompts
                    uncond_embedding = self.uncond_embedding.tile([len(text_embedding), 1, 1])

                if self.dmd2_args.denoising:
                    denoising_dict = next(self.denoising_dataloader)
                else:
                    denoising_dict = None

                if self.dmd2_args.cls_on_clean_image:
                    real_train_dict = next(self.real_dataloader)
                else:
                    real_train_dict = None

                inputs["noise"] = noise
                inputs["text_embedding"] = text_embedding
                inputs["uncond_embedding"] = uncond_embedding
                inputs["denoising_dict"] = denoising_dict
                inputs["real_train_dict"] = real_train_dict
                # inputs['text_embedding'] = text_embedding

                if self.args.ignore_data_skip:
                    self.timers and self.timers("read-data").stop()

                os.environ["TRAINER_GLOBAL_STEP"] = str(self.state.global_step)
                self.callback_handler.on_load_data_end(args, self.state, self.control, inputs=inputs)

                # Skip past any already trained steps if resuming training
                # for paddlenlp.utils.batch_sampler.DistributedBatchSampler
                # We use consumed_samples to reset the status
                if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                    train_dataloader.batch_sampler, NlpDistributedBatchSampler
                ):
                    if step == 0:
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(steps_trained_in_current_epoch)
                            steps_trained_progress_bar.close()
                            steps_trained_progress_bar = None
                        self._load_rng_state(resume_from_checkpoint)
                    step += steps_trained_in_current_epoch
                elif steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    self.timers and self.timers("read-data").start()
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if should_skip_data(self.state.global_step, self.args.skip_data_intervals):
                    # skip this step

                    if (step_control + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                    ):
                        # update current global step and skip step
                        self.state.global_step += 1
                        self._skip_global_steps += 1
                        self._skip_steps_since_last_logged += 1

                        self.state.epoch = epoch + (step + 1) / steps_in_epoch

                        if self.state.global_step == 1 and self.args.logging_first_step:
                            self.control.should_log = True
                        if (
                            self.args.logging_strategy == IntervalStrategy.STEPS
                            and self.state.global_step % self.args.logging_steps == 0
                        ):
                            self.control.should_log = True

                        self.control.should_evaluate = False
                        self.control.should_save = False

                        # log loss and memeory usage
                        self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)
                        self._print_timer()
                        step_control = 0
                    else:
                        step_control += 1
                    if self.state.global_step >= self.state.max_steps:
                        break

                    self.timers and self.timers("read-data").start()
                    continue

                if step_control % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                    self.timers and self.timers("generator-forward-backward").start()

                # stage2 and stage3 should not no_sync, because the is no DDP wrapper and no_sync API
                # hybrid_parallel (tp or pp or sharding stage 1) should not no_sync
                availiable_no_sync = hasattr(model, "no_sync")
                is_no_sync = (
                    (
                        ((step_control + 1) % args.gradient_accumulation_steps != 0)
                        and args._no_sync_in_gradient_accumulation
                    )
                    or args.recompute
                    or args.use_expert_parallel
                ) and availiable_no_sync
                # sharding
                # stage1. the same as ddp
                # stage2. manualy collect gradient on dp group

                dp_master_grad = (
                    self.args.world_size > 1 and self.args.amp_master_grad and not self.args.use_hybrid_parallel
                )
                if dp_master_grad:
                    is_no_sync = True

                if is_no_sync:
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step, generator_log_dict = self.training_step(self.unified_model, inputs)
                else:
                    tr_loss_step, generator_log_dict = self.training_step(self.unified_model, inputs)

                tr_loss += tr_loss_step

                def fused_allreduce_gradients_no_sync(paramlist, hcg):
                    paramlist = list(paramlist)
                    nonmoe_list = [p for p in paramlist if not getattr(p, "no_sync", False)]
                    moelist = [p for p in paramlist if getattr(p, "no_sync", False)]
                    if moelist and not self.args.use_expert_parallel:
                        logger.warning("found `no sync` param when `use_expert_parallel=False`")
                    fused_allreduce_gradients(nonmoe_list, hcg)

                if COMPUTE_GENERATOR_GRADIENT:
                    if (step_control + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                    ):
                        if self.args.pipeline_parallel_degree <= 1 and self._enable_delay_scale_loss():
                            tr_loss /= self.args.gradient_accumulation_steps

                        self.timers and self.timers("generator-forward-backward").stop()
                        # Maunally collect gradients
                        # Case 1: Use recompute and dp
                        # Case 2: Hack dp with master_grad
                        # Case 3: Pipeline or sharding overlap
                        # local_rank != -1 don't means dp in networks.
                        self.timers and self.timers("all-reduce").start()

                        # Case 1: Use recompute and dp / sharding stage1,
                        # manualy collect gradient for dp.
                        if (args.recompute or args.use_expert_parallel) and availiable_no_sync:
                            fused_allreduce_gradients_no_sync(list(model.parameters()), None)

                        # Case 2: hack dp with master_grad
                        elif dp_master_grad:
                            fused_allreduce_gradients_no_sync(list(model.parameters()), None)

                        # Pipeline parallel mode,  handle gradient reduce here to overlap
                        enable_dp_comm_overlap = "enable_dp_comm_overlap" in args.pipeline_parallel_config

                        enable_release_grads = False
                        if args.sharding_parallel_degree > 1:
                            enable_release_grads = "enable_release_grads" in args.sharding_parallel_config
                        if not enable_release_grads and args.pipeline_parallel_degree > 1:
                            enable_release_grads = "enable_release_grads" in args.pipeline_parallel_config

                        # Case 3: Pipeline parallel mode, overlap with dp
                        if isinstance(self.optimizer, HybridParallelOptimizer) and not self.do_grad_scaling:
                            parameters_list = _obtain_optimizer_parameters_list(self.optimizer._inner_opt)

                            if not enable_dp_comm_overlap:
                                if self.optimizer._sharding_enable:
                                    assert reshard_util.is_sharding_opt(self.optimizer)
                                    self.optimizer._inner_opt.reduce_gradients(
                                        list(parameters_list), self.optimizer._hcg
                                    )

                                if self.optimizer._dp_enable or getattr(self.optimizer, "_sep_enable", False):
                                    fused_allreduce_gradients_no_sync(list(parameters_list), self.optimizer._hcg)
                        self.timers and self.timers("generator-all-reduce").stop()
                        self.timers and self.timers("generator-optimizer-step").start()

                        if self.args.gradient_accumulation_steps > 1 and self._enable_delay_scale_loss():
                            paddle.device.synchronize()
                            for p in model._layers.parameters():
                                with paddle.no_grad():
                                    if hasattr(p, "main_grad") and p.main_grad is not None:
                                        assert p.grad is None
                                        p.main_grad.scale_(1.0 / self.args.gradient_accumulation_steps)
                                    elif p.grad is not None:
                                        p.grad.scale_(1.0 / self.args.gradient_accumulation_steps)

                        # Optimizer step
                        self.callback_handler.on_optimizer_begin(
                            args, self.state, self.control, scaler=self.scaler if self.do_grad_scaling else None
                        )
                        optimizer_was_run = True

                        if self.args.offload_optim:
                            self._reload_optimizer()

                        if self.do_grad_scaling:
                            if args.pipeline_parallel_degree > 1:
                                assert not self.args.use_expert_parallel, "pipeline moe not work under fp16"
                            scale_before = paddle.assign(self.scaler._scale)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale_after = self.scaler._scale
                            # Compatible with paddlepaddle 2.6.0 using typo word.
                            if hasattr(self.scaler, "_cache_founf_inf"):
                                optimizer_was_run = not self.scaler._cache_founf_inf
                            else:
                                optimizer_was_run = not self.scaler._cache_found_inf
                            if not optimizer_was_run:
                                scale_before_value = scale_before.cpu().numpy()
                                scale_after_value = scale_after.cpu().numpy()
                                logger.warning(
                                    f"optimizer not run, scale_before: {scale_before_value[0]}, scale_after: {scale_after_value[0]}"
                                )
                        elif isinstance(self.optimizer, HybridParallelOptimizer):
                            print("HybridParallelOptimizer_step")
                            self.optimizer._step(parameters_list)
                        else:
                            self.optimizer.step()

                        if self.args.offload_optim:
                            self._offload_optimizer()

                        self.timers and self.timers("generator-optimizer-step").stop()

                        if optimizer_was_run:
                            self.lr_scheduler.step()

                        if args.release_grads or enable_release_grads:
                            self.optimizer.clear_grad(set_to_zero=False)
                            self.optimizer_guidance.clear_grad(set_to_zero=False)
                            if args.pipeline_parallel_degree > 1:
                                for _, buffers in model._chunk_2_comm_buffers.items():
                                    for buffer in buffers:
                                        buffer._clear_grad_storage()
                        else:
                            self.optimizer.clear_grad()
                            self.optimizer_guidance.clear_grad()

                        self.callback_handler.on_optimizer_end(
                            args, self.state, self.control, scaler=self.scaler if self.do_grad_scaling else None
                        )

                        # self.state.global_step += 1
                        # self.state.epoch = epoch + (step + 1) / steps_in_epoch
                        # self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)
                        # self._print_timer()
                        # step_control = 0
                    else:
                        pass
                    # self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                    # step_control += 1
                tr_loss = 0.0
                inputs["guidance_data_dict"] = generator_log_dict["guidance_data_dict"]
                if is_no_sync:
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with guidance_model.no_sync():
                        tr_guidance_loss_step, guidance_log_dict = self.training_step(
                            self.unified_model, inputs, generator_turn=False
                        )
                else:
                    tr_guidance_loss_step, guidance_log_dict = self.training_step(
                        self.unified_model, inputs, generator_turn=False
                    )

                tr_loss += tr_guidance_loss_step

                if (step_control + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    if self.args.pipeline_parallel_degree <= 1 and self._enable_delay_scale_loss():
                        tr_loss /= self.args.gradient_accumulation_steps

                    self.timers and self.timers("forward-backward").stop()
                    # Maunally collect gradients
                    # Case 1: Use recompute and dp
                    # Case 2: Hack dp with master_grad
                    # Case 3: Pipeline or sharding overlap
                    # local_rank != -1 don't means dp in networks.
                    self.timers and self.timers("all-reduce").start()

                    # Case 1: Use recompute and dp / sharding stage1,
                    # manualy collect gradient for dp.
                    if (args.recompute or args.use_expert_parallel) and availiable_no_sync:
                        fused_allreduce_gradients_no_sync(list(model.parameters()), None)

                    # Case 2: hack dp with master_grad
                    elif dp_master_grad:
                        fused_allreduce_gradients_no_sync(list(model.parameters()), None)

                    # Pipeline parallel mode,  handle gradient reduce here to overlap
                    enable_dp_comm_overlap = "enable_dp_comm_overlap" in args.pipeline_parallel_config

                    enable_release_grads = False
                    if args.sharding_parallel_degree > 1:
                        enable_release_grads = "enable_release_grads" in args.sharding_parallel_config
                    if not enable_release_grads and args.pipeline_parallel_degree > 1:
                        enable_release_grads = "enable_release_grads" in args.pipeline_parallel_config

                    # Case 3: Pipeline parallel mode, overlap with dp
                    if isinstance(self.optimizer_guidance, HybridParallelOptimizer) and not self.do_grad_scaling:
                        parameters_list = _obtain_optimizer_parameters_list(self.optimizer_guidance._inner_opt)

                        if not enable_dp_comm_overlap:
                            if self.optimizer_guidance._sharding_enable:
                                assert reshard_util.is_sharding_opt(self.optimizer_guidance)
                                self.optimizer_guidance._inner_opt.reduce_gradients(
                                    list(parameters_list), self.optimizer_guidance._hcg
                                )

                            if self.optimizer._dp_enable or getattr(self.optimizer_guidance, "_sep_enable", False):
                                fused_allreduce_gradients_no_sync(list(parameters_list), self.optimizer_guidance._hcg)
                    self.timers and self.timers("guidance-all-reduce").stop()
                    self.timers and self.timers("guidance-optimizer-step").start()

                    if self.args.gradient_accumulation_steps > 1 and self._enable_delay_scale_loss():
                        paddle.device.synchronize()
                        for p in model._layers.parameters():
                            with paddle.no_grad():
                                if hasattr(p, "main_grad") and p.main_grad is not None:
                                    assert p.grad is None
                                    p.main_grad.scale_(1.0 / self.args.gradient_accumulation_steps)
                                elif p.grad is not None:
                                    p.grad.scale_(1.0 / self.args.gradient_accumulation_steps)

                    # Optimizer step
                    self.callback_handler.on_optimizer_begin(
                        args, self.state, self.control, scaler=self.scaler if self.do_grad_scaling else None
                    )
                    optimizer_was_run = True

                    if self.args.offload_optim:
                        self._reload_optimizer()

                    if self.do_grad_scaling:
                        if args.pipeline_parallel_degree > 1:
                            assert not self.args.use_expert_parallel, "pipeline moe not work under fp16"
                        scale_before = paddle.assign(self.scaler._scale)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler._scale
                        # Compatible with paddlepaddle 2.6.0 using typo word.
                        if hasattr(self.scaler, "_cache_founf_inf"):
                            optimizer_was_run = not self.scaler._cache_founf_inf
                        else:
                            optimizer_was_run = not self.scaler._cache_found_inf
                        if not optimizer_was_run:
                            scale_before_value = scale_before.cpu().numpy()
                            scale_after_value = scale_after.cpu().numpy()
                            logger.warning(
                                f"optimizer not run, scale_before: {scale_before_value[0]}, scale_after: {scale_after_value[0]}"
                            )
                    elif isinstance(self.optimizer_guidance, HybridParallelOptimizer):
                        self.optimizer_guidance._step(parameters_list)
                    else:
                        self.optimizer_guidance.step()

                    if self.args.offload_optim:
                        self._offload_optimizer()

                    self.timers and self.timers("guidance-optimizer-step").stop()

                    if optimizer_was_run:
                        self.lr_scheduler_guidance.step()

                    if args.release_grads or enable_release_grads:
                        self.optimizer_guidance.clear(set_to_zero=False)
                        self.optimizer.clear_grad(set_to_zero=False)
                        if args.pipeline_parallel_degree > 1:
                            for _, buffers in model._chunk_2_comm_buffers.items():
                                for buffer in buffers:
                                    buffer._clear_grad_storage()
                    else:
                        self.optimizer_guidance.clear_grad()
                        # self.optimizer.clear_grad()

                    self.callback_handler.on_optimizer_end(
                        args, self.state, self.control, scaler=self.scaler if self.do_grad_scaling else None
                    )

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    log_dict = {**generator_log_dict, **guidance_log_dict}
                    self.state._log_dict = log_dict
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)
                    self._print_timer()
                    step_control = 0
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                    step_control += 1

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

                if self.args.ignore_data_skip:
                    self.timers and self.timers("read-data").start()

                paddle.device.cuda.empty_cache()
                import gc

                gc.collect()

            if step < 0:
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({self.state.max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\nTraining completed. \n")

        # unlink shared_memory if used.
        if self.args.unified_checkpoint:
            self.unified_checkpoint_handler.unlink_shared_memory()

        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            if args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, LoRAModel) or isinstance(self.model, PrefixModelForCausalLM):
                self._load_best_model_from_peft_checkpoint()
            else:
                if self.args.unified_checkpoint:
                    self.unified_checkpoint_handler.load_unified_checkpoint(
                        self.model,
                        self.state.best_model_checkpoint,
                    )
                    if self.args.sharding_parallel_degree > 1 or self.args.data_parallel_degree > 1:
                        broadcast_dataset_rank0_model(self.model)
                else:
                    weight_name = PADDLE_WEIGHTS_NAME
                    best_model_path = os.path.join(
                        self.state.best_model_checkpoint, _add_variant(weight_name, self.args.weight_name_suffix)
                    )
                    if os.path.exists(best_model_path):
                        # We load the model state dict on the CPU to avoid an OOM error.
                        state_dict = paddle.load(best_model_path, return_numpy=True)
                        # If the model is on the GPU, it still works!
                        self._set_state_dict_in_model(state_dict)
                    else:
                        logger.warning(
                            f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                            "on multiple nodes, you should activate `--save_on_each_node`."
                        )

        self._total_loss_scalar += tr_loss.item()

        # In case all steps were skipped, the total loss is set to 0.
        if self.state.global_step == self._skip_global_steps:
            logger.info("All steps were skipped, the total loss is set to 0.")
            train_loss = 0.0
        else:
            train_loss = self._total_loss_scalar / (self.state.global_step - self._skip_global_steps)

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)

        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _load_best_model_from_peft_checkpoint(self):
        if self.args.unified_checkpoint:
            self.unified_checkpoint_handler.load_unified_checkpoint(
                self.model,
                self.state.best_model_checkpoint,
            )
            if self.args.sharding_parallel_degree > 1 or self.args.data_parallel_degree > 1:
                broadcast_dataset_rank0_model(self.model)
            return

        convert_tp = False
        if isinstance(self.model, LoRAModel):
            if self.model.quantized or self.args.pipeline_parallel_degree > 1:
                best_model_path = os.path.join(
                    self.state.best_model_checkpoint, _add_variant(LORA_WEIGHTS_NAME, self.args.weight_name_suffix)
                )
            else:
                best_model_path = os.path.join(self.state.best_model_checkpoint, LORA_WEIGHTS_NAME)
                if self.model.lora_config.tensor_parallel_degree > 1:
                    convert_tp = True

        elif isinstance(self.model, PrefixModelForCausalLM):
            best_model_path = os.path.join(self.state.best_model_checkpoint, PREFIX_WEIGHTS_NAME)
            if self.model.prefix_config.tensor_parallel_degree > 1:
                convert_tp = True

        if os.path.exists(best_model_path):
            # We load the model state dict on the CPU to avoid an OOM error.
            state_dict = paddle.load(best_model_path, return_numpy=True)
            if convert_tp:
                state_dict = self.model._convert_tensor_parallel(state_dict)
            # If the model is on the GPU, it still works!
            self._set_state_dict_in_model(state_dict)
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )

    def _get_train_sampler(self):
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.world_size <= 1:
            return paddle.io.BatchSampler(
                dataset=self.train_dataset,
                shuffle=True,
                batch_size=self.args.per_device_train_batch_size,
                drop_last=self.args.dataloader_drop_last,
            )

        return DistributedBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_replicas=self.args.dataset_world_size,
            rank=self.args.dataset_rank,
            drop_last=self.args.dataloader_drop_last,
        )

    def _set_state_dict_in_model(self, state_dict):
        # TODO  @ZHUI paddle need return the results of set_state_dict.
        logger.info(f"set state-dict :{self.model.set_state_dict(state_dict)}")

    def _print_timer(self):
        """print timer and clear states"""
        paddle_timer_info = ""
        try:
            from paddle.distributed.fleet.utils.timer_helper import (
                get_timers as paddle_get_timers,
            )

            paddle_pipeline_timers = paddle_get_timers()
            for name, timer in paddle_pipeline_timers.timers.items():
                elapsed_time = timer.elapsed(reset=False) * 1000.0
                paddle_timer_info += f" | {name}: {elapsed_time:.2f}"
            paddle_pipeline_timers.log(paddle_pipeline_timers.timers.keys(), reset=True)
        except ImportError:  # paddle version too old, timer not support
            warnings.warn(f"paddle version:{paddle.__git_commit__} does not support pipeline timer")
        except AssertionError:  # paddle timer not enabled
            pass

        if self.timers is not None:
            timer_info = self.timers.log(self.timers.timers.keys(), reset=True)
        else:
            timer_info = ""

        if timer_info or paddle_timer_info:
            logger.info(f"[Profile global_step: {self.state.global_step}] {timer_info} {paddle_timer_info}")

    def _get_item_from_loss(self, loss):
        assert isinstance(loss, paddle.Tensor) and loss._is_initialized()
        loss_value = loss.item()
        if not self.args.fp16:
            if not np.isfinite(loss_value).all():
                err_msg = LOSS_NAN_ERROR if np.isnan(loss_value).any() else LOSS_INF_ERROR
                raise ValueError(f"{err_msg}. Loss contains inf or nan values, its value is {loss_value}")
        return loss_value

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch, ignore_keys_for_eval, **kwargs):
        if self.control.should_log:

            logs: Dict[str, float] = {}
            num_steps = self.state.global_step - self._globalstep_last_logged - self._skip_steps_since_last_logged
            self._skip_steps_since_last_logged = 0
            # all_gather + mean() to get average loss over all processes
            avg_loss = self._nested_gather(tr_loss).mean()
            tr_loss_scalar = self._get_item_from_loss(avg_loss)

            # reset tr_loss to zero
            tr_loss.subtract_(tr_loss)
            # set loss to zero if all steps are skipped since last log
            if num_steps == 0:
                logs["loss"] = 0.0
            else:
                logs["loss"] = round(tr_loss_scalar / num_steps, 8)

            logs["learning_rate"] = float("{0:.3e}".format(self._get_learning_rate()))
            logs["global_step"] = int(self.state.global_step)
            if in_auto_parallel_align_mode():
                logs["loss_md5"] = avg_loss._md5sum()

            divisor = 2**30
            # TODO(@gexiao): replace these codes with unified APIs in Paddle
            current_device = paddle.framework._current_expected_place_()
            if str(current_device) != "Place(cpu)":
                device_id = current_device.get_device_id()
                current_memory_allocated = core.device_memory_stat_current_value("Allocated", device_id)
                current_memory_reserved = core.device_memory_stat_current_value("Reserved", device_id)
                max_memory_allocated = core.device_memory_stat_peak_value("Allocated", device_id)
                max_memory_reserved = core.device_memory_stat_peak_value("Reserved", device_id)
                logs["current_memory_allocated"] = current_memory_allocated / divisor
                logs["current_memory_reserved"] = current_memory_reserved / divisor
                logs["max_memory_allocated"] = max_memory_allocated / divisor
                logs["max_memory_reserved"] = max_memory_reserved / divisor

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self._globalstep_last_start_time = time.time()

            # Add additional memory in log.
            if not self.args.skip_memory_metrics:
                logs.update(
                    {
                        "cpu_mem_used": self._memory_tracker.cpu_mem_used() >> 20,
                        "cpu_mem_used_peak": self._memory_tracker.cpu_mem_used_peak >> 20,
                    }
                )
                if is_paddle_cuda_available():
                    logs.update(
                        {
                            "gpu_max_memory_allocated": paddle.device.cuda.max_memory_allocated() >> 20,
                            "gpu_max_memory_reserved": paddle.device.cuda.max_memory_reserved() >> 20,
                        }
                    )

            self.log(logs, **kwargs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.optimizer, GroupShardedOptimizerStage2) and self.optimizer._broadcast_overlap:
                paddle.device.synchronize()

            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)

        if self.control.should_save:
            if isinstance(self.optimizer, GroupShardedOptimizerStage2) and self.optimizer._broadcast_overlap:
                paddle.device.synchronize()

            self._save_checkpoint(model, metrics=metrics)
            logger.info(f"{self.runtime_timer.log()}")
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _get_learning_rate(self):
        return self.optimizer.get_lr()

    def get_train_dataloader(self):
        """
        Returns the training [`~paddle.io.DataLoader`].

        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        return self.train_dataloader

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()

    def create_optimizer(self, lr_scheduler=None):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            if self.optimizer_grouped_parameters is not None:
                params = self.optimizer_grouped_parameters
                params_guidance = self.optimizer_grouped_parameters
                apply_decay_param_fun = None
            else:
                params = self.model.parameters()
                params_guidance = self.guidance_model.parameters()
                decay_parameters = [
                    p.name for n, p in self.model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])
                ]

                def apply_decay_param_fun(x):
                    return x in decay_parameters

                decay_parameters_guidance = [
                    p.name
                    for n, p in self.guidance_model.named_parameters()
                    if not any(nd in n for nd in ["bias", "norm"])
                ]

                def apply_decay_param_guidance_fun(x):
                    return x in decay_parameters_guidance

            optimizer_cls, optimizer_kwargs = NLPTrainer.get_optimizer_cls_and_kwargs(self.args)
            if hasattr(optimizer_cls, "_create_master_weight") and self.args.fp16_opt_level == "O2":
                optimizer_kwargs["multi_precision"] = True

            self.optimizer = optimizer_cls(
                learning_rate=self.lr_scheduler if lr_scheduler is None else lr_scheduler,
                apply_decay_param_fun=apply_decay_param_fun,
                parameters=params,
                weight_decay=self.args.weight_decay,
                grad_clip=nn.ClipGradByGlobalNorm(self.args.max_grad_norm) if self.args.max_grad_norm > 0 else None,
                **optimizer_kwargs,
            )
            self.optimizer_guidance = optimizer_cls(
                learning_rate=self.lr_scheduler_guidance if lr_scheduler is None else lr_scheduler,
                apply_decay_param_fun=apply_decay_param_guidance_fun,
                parameters=params_guidance,
                weight_decay=self.args.weight_decay,
                grad_clip=nn.ClipGradByGlobalNorm(self.args.max_grad_norm) if self.args.max_grad_norm > 0 else None,
                **optimizer_kwargs,
            )

        return self.optimizer

    def _apply_to_optimizer(self, action):
        attributes = [
            ("_accumulators", "_moment1_acc_str"),
            ("_accumulators", "_moment2_acc_str"),
            ("_master_weights",),
            ("_accumulators_holder",),
        ]

        for attr in attributes:
            if all(hasattr(self.optimizer, a) for a in attr):
                target_attr = getattr(self.optimizer, attr[0])
                if len(attr) == 2:
                    target_attr = target_attr[getattr(self.optimizer, attr[1])]

                for key, value in target_attr.items():
                    if get_env_device() == "gpu":
                        target_attr[key] = getattr(value, action)()
                    else:
                        target_attr[key] = getattr(value, "to")(action)

    def _offload_optimizer(self):
        if get_env_device() == "gpu":
            self._apply_to_optimizer("pin_memory")
        else:
            self._apply_to_optimizer("cpu")

    def _reload_optimizer(self):
        if get_env_device() == "gpu":
            self._apply_to_optimizer("cuda")
        else:
            self._apply_to_optimizer(get_env_device())

    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        # if use distributed training
        if self.args.world_size > 1:
            process_index = self.args.process_index
            rng_file_list = [None for x in range(self.args.world_size)]
            if self.args.should_save:
                rng_file = os.path.join(checkpoint, f"rng_state_{self.args.world_size}.pth")
                if os.path.isfile(rng_file):
                    rng_file_list = paddle.load(rng_file, return_numpy=True)
            paddle.distributed.broadcast_object_list(rng_file_list, src=0)
            # if rng_file_list still empty, not log rng state.
            if rng_file_list[0] is None:
                logger.info(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
            else:
                checkpoint_rng_state = rng_file_list[process_index]
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

            checkpoint_rng_state = paddle.load(rng_file, return_numpy=True)

        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])

        core.default_cpu_generator().set_state(checkpoint_rng_state["cpu"])
        if core.is_compiled_with_cuda():
            if not len(checkpoint_rng_state["cuda"]) == core.get_cuda_device_count():
                raise ValueError("Length of gpu state list shoule be equal to the gpu device count")
            for i in range(core.get_cuda_device_count()):
                core.default_cuda_generator(i).set_state(checkpoint_rng_state["cuda"][i])

        if core.is_compiled_with_xpu():
            if not len(checkpoint_rng_state["cuda"]) == core.get_xpu_device_count():
                raise ValueError("Length of xpu state list shoule be equal to the xpu device count")
            for i in range(core.get_xpu_device_count()):
                core.default_xpu_generator(i).set_state(checkpoint_rng_state["cuda"][i])

        if paddle.device.get_all_custom_device_type() is not None:
            custom_device_type = paddle.device.get_all_custom_device_type()
            for device in custom_device_type:
                if not len(checkpoint_rng_state["cuda"]) == core.get_custom_device_count(device):
                    raise ValueError("Length of custom device state list shoule be equal to the custom device count")
                for i in range(core.get_custom_device_count(device)):
                    core.default_custom_device_generator(paddle.CustomPlace(device, i)).set_state(
                        checkpoint_rng_state["cuda"][i]
                    )

        if self.args.use_hybrid_parallel:
            if "hybrid_parallel_rng_state_tracker" in checkpoint_rng_state:
                if self.args.tensor_parallel_degree <= 1:
                    checkpoint_rng_state["hybrid_parallel_rng_state_tracker"].pop("model_parallel_rng", None)
                fleet.meta_parallel.get_rng_state_tracker().set_states_tracker(
                    checkpoint_rng_state["hybrid_parallel_rng_state_tracker"]
                )
            else:
                logger.warning("Not found hybrid parallel RNG state.")

    @staticmethod
    def get_optimizer_cls_and_kwargs(args):
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`paddlenlp.training_args.TrainingArguments`):
                The training arguments for the training session.

        """
        # optimizer_kwargs = {"lr": args.learning_rate}
        optimizer_kwargs = {}
        adam_kwargs = {
            "beta1": args.adam_beta1,
            "beta2": args.adam_beta2,
            "epsilon": args.adam_epsilon,
        }
        if args.optim == OptimizerNames.ADAMW:
            from paddle.optimizer import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
        return optimizer_cls, optimizer_kwargs

    def create_scheduler(self, num_training_steps: int):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        warmup = (
            self.args.warmup_steps if self.args.warmup_steps > 0 else int(self.args.warmup_ratio * num_training_steps)
        )
        decay_steps = num_training_steps
        if getattr(self.args, "decay_steps", None) and self.args.decay_steps > 0:
            decay_steps = self.args.decay_steps

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                learning_rate=self.args.learning_rate,
                num_warmup_steps=warmup,
                num_training_steps=decay_steps,
                num_cycles=self.args.num_cycles,
                lr_end=self.args.lr_end,
                power=self.args.power,
            )

        return self.lr_scheduler

    def num_examples(self, dataloader) -> int:
        """
        Helper to get number of samples in a [`~paddle.io.DataLoader`] by accessing its dataset. When
        dataloader.dataset does not exist or has no length, estimates as best it can
        """
        try:
            dataset = dataloader.dataset
            # Special case for IterableDatasetShard, we need to dig deeper
            if isinstance(dataset, IterableDatasetShard):
                return len(dataloader.dataset.dataset)
            return len(dataloader.dataset)
        except (NameError, AttributeError, TypeError):  # no dataset or length, estimate by length of dataloader
            return len(dataloader) * self.args.per_device_train_batch_size

    def _wrap_model(self, model, guidance_model=None, training=True):

        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model

        # Note: in paddle.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        # Mixed precision training
        if training and self.do_grad_scaling:  # self.args.fp16_opt_level=="O2":
            # model, self.optimizer
            decorated = paddle.amp.decorate(
                models=model,
                optimizers=self.optimizer,
                level=self.args.fp16_opt_level,
                dtype=self.amp_dtype,
                excluded_layers=[QuantizationLinear] + self._decorate_exclude_layers(model),
            )
            decorated_guidance = paddle.amp.decorate(
                models=guidance_model,
                optimizers=self.optimizer_guidance,
                level=self.args.fp16_opt_level,
                dtype=self.amp_dtype,
                excluded_layers=[QuantizationLinear] + self._decorate_exclude_layers(model),
            )

            if self.optimizer is None:
                model = decorated
                guidance_model = decorated
            else:
                model, self.optimizer = decorated
                guidance_model, self.optimizer_guidance = decorated_guidance

        if self.args.world_size == 1:
            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)
                mix_precision_utils.MixPrecisionLayer(guidance_model, dtype=self.amp_dtype)
                assert self.optimizer is not None, "optimizer is empty!"
                self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)
                self.optimizer_guidance = mix_precision_utils.MixPrecisionOptimizer(self.optimizer_guidance)

        in_pipeline_parallel_mode = self.args.pipeline_parallel_degree > 1
        in_sharding_parallel_mode = self.sharding is not None
        in_tensor_parallel_mode = self.args.tensor_parallel_degree > 1
        in_sep_parallel_mode = self.args.sep_parallel_degree > 1
        in_cp_parallel_mode = self.args.context_parallel_degree > 1

        # Multi-gpu training
        if self.args.world_size > 1 and (not self.args.use_hybrid_parallel):
            # MOE use DDP to broadcaset parameters.
            ddp_kwargs = {}
            if self.args.ddp_find_unused_parameters is not None:
                ddp_kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
            elif isinstance(model, PretrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                ddp_kwargs["find_unused_parameters"] = not any(
                    hasattr(m, "enable_recompute") and m.enable_recompute for m in model.sublayers(include_self=True)
                )
            else:
                ddp_kwargs["find_unused_parameters"] = True
            model = paddle.DataParallel(model, **ddp_kwargs)
            guidance_model = paddle.DataParallel(guidance_model, **ddp_kwargs)
            # Distributed training (should be after fp16 initialization)

            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)
                mix_precision_utils.MixPrecisionLayer(guidance_model, dtype=self.amp_dtype)
                assert self.optimizer is not None, "optimizer is empty!"
                self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)
                self.optimizer_guidance = mix_precision_utils.MixPrecisionOptimizer(self.optimizer_guidance)

        # Pipeline mode
        # todo
        if in_pipeline_parallel_mode:
            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)  # return value has no use
                mix_precision_utils.MixPrecisionLayer(guidance_model, dtype=self.amp_dtype)  # return value has no use
            # hack for pipeline model mini batch to batch
            # need batter solution @ZHUI
            # make batch_fn compatible for fleet.distributed_model decorate.
            prepare_pipeline_inputs_func = (
                model._prepare_pipeline_inputs_func if hasattr(model, "_prepare_pipeline_inputs_func") else None
            )
            prepare_pipeline_inputs_func_guidance = (
                guidance_model._prepare_pipeline_inputs_func
                if hasattr(guidance_model, "_prepare_pipeline_inputs_func")
                else None
            )
            if isinstance(model, LoRAModel):
                model = model.model
            if isinstance(guidance_model, LoRAModel):
                guidance_model = guidance_model.model

            model = fleet.distributed_model(model)
            guidance_model = fleet.distributed_model(guidance_model)
            if prepare_pipeline_inputs_func is not None:
                model._prepare_pipeline_inputs_func = prepare_pipeline_inputs_func
                guidance_model._prepare_pipeline_inputs_func = prepare_pipeline_inputs_func_guidance
            else:

                def _prepare_pipeline_inputs_func(inputs):
                    first_stage_keys = ["input_ids", "attention_mask", "position_ids"]
                    last_stage_keys = ["labels"]

                    def get_expected_keys(inputs, keys):
                        ret = tuple([inputs.pop(k) for k in keys if k in inputs])
                        if len(ret) == 1:
                            ret = ret[0]
                        return ret

                    if type(inputs) is dict or type(inputs) is OrderedDict:
                        return [
                            get_expected_keys(inputs, first_stage_keys),
                            get_expected_keys(inputs, last_stage_keys),
                        ]

                    keys = list(inputs[0].keys())
                    inputs_batch = {key: [data.pop(key) for data in inputs] for key in keys}
                    return [
                        get_expected_keys(inputs_batch, first_stage_keys),
                        get_expected_keys(inputs_batch, last_stage_keys),
                    ]

                logger.warning(
                    "Using default prepare pipeline inputs func, only support input_ids and labels as inputs."
                )
                model._prepare_pipeline_inputs_func = _prepare_pipeline_inputs_func
                guidance_model._prepare_pipeline_inputs_func = _prepare_pipeline_inputs_func

            assert self.optimizer is not None, "Pipeline mode need decorate optimizer, pelease init optimizer."
            if self.args.amp_master_grad:
                self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)
                self.optimizer_guidance = mix_precision_utils.MixPrecisionOptimizer(self.optimizer_guidance)
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
            self.optimizer_guidance = fleet.distributed_optimizer(self.optimizer_guidance)

            if (
                hasattr(self.args, "enable_sharding_comm_overlap")
                and self.args.enable_sharding_comm_overlap
                and self.args.unified_checkpoint
                and "split_param" in split_parallel_config(self.args.sharding_parallel_config)
            ):
                model.register_sharding_comm_overlap_hook(self.optimizer)
                guidance_model.register_sharding_comm_overlap_hook(self.optimizer_guidance)

        # No pipeline mode, sharding only
        if not in_pipeline_parallel_mode and in_sharding_parallel_mode:
            # Sharded DDP!
            if self.args.tensor_parallel_degree > 1:
                hcg = fleet.get_hybrid_communicate_group()
                assert (
                    ShardingOption.SHARD_GRAD_OP in self.args.sharding or ShardingOption.SHARD_OP in self.args.sharding
                ), "Only support tensor parallel + sharding stage1/stage2 hybrid parallel now."
                model = paddle.distributed.fleet.meta_parallel.TensorParallel(model, hcg, strategy=None)
                guidance_model = paddle.distributed.fleet.meta_parallel.TensorParallel(
                    guidance_model, hcg, strategy=None
                )

            if ShardingOption.SHARD_OP in self.args.sharding:
                if self.args.amp_master_grad:
                    mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)  # return value has no use
                    mix_precision_utils.MixPrecisionLayer(
                        guidance_model, dtype=self.amp_dtype
                    )  # return value has no use
                model = fleet.distributed_model(model)
                guidance_model = fleet.distributed_model(guidance_model)

                if self.args.amp_master_grad:
                    self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)
                    self.optimizer_guidance = mix_precision_utils.MixPrecisionOptimizer(self.optimizer_guidance)
                self.optimizer = fleet.distributed_optimizer(self.optimizer)
                self.optimizer_guidance = fleet.distributed_optimizer(self.optimizer_guidance)
            else:
                cpu_offload = ShardingOption.OFFLOAD in self.args.sharding
                assert self.optimizer is not None, "optimizer is empty!"
                level = None
                if ShardingOption.SHARD_GRAD_OP in self.args.sharding:
                    level = "os_g"
                if ShardingOption.FULL_SHARD in self.args.sharding:
                    level = "p_g_os"

                from paddle.distributed.sharding import group_sharded_parallel

                # add dp_group and exclude_layer params
                # https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/sharding/group_sharded_parallel_cn.html#group-sharded-parallel
                extra_kwargs = {}
                extra_kwargs["dp_group"] = self.dp_group
                extra_kwargs["exclude_layer"] = ["GroupNorm"]

                if self.args.amp_master_grad:
                    assert (
                        self.args.data_parallel_degree == 1
                    ), "Sharding stage 2 / Sharding stage 3 main grad is not compatible with dp for now."
                    mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)  # return value has no use
                    mix_precision_utils.MixPrecisionLayer(
                        guidance_model, dtype=self.amp_dtype
                    )  # return value has no use
                    self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)
                    self.optimizer_guidance = mix_precision_utils.MixPrecisionOptimizer(self.optimizer_guidance)

                model, optimizer, _ = group_sharded_parallel(
                    model,
                    self.optimizer,
                    level=level,
                    scaler=None,
                    group=self.sharding_group,
                    offload=cpu_offload,
                    **extra_kwargs,
                )
                guidance_model, optimizer_guidance, _ = group_sharded_parallel(
                    guidance_model,
                    self.optimizer_guidance,
                    level=level,
                    scaler=None,
                    group=self.sharding_group,
                    offload=cpu_offload,
                    **extra_kwargs,
                )
                if ShardingOption.SHARD_GRAD_OP in self.args.sharding and self.args.amp_master_grad:
                    assert hasattr(optimizer, "use_main_grad"), (
                        "Current installed paddle doesn't support sharding stage 2 with main grad, "
                        "please upgrade your paddle (using nightly version)."
                    )

                if level == "os_g" and "enable_stage2_overlap" in self.args.sharding_parallel_config:
                    model._set_reduce_overlap(True)
                    optimizer._set_broadcast_overlap(True, model)
                    guidance_model._set_reduce_overlap(True)
                    optimizer_guidance._set_broadcast_overlap(True, guidance_model)

                self.optimizer = optimizer
                self.optimizer_guidance = optimizer_guidance

        # pure tesnor parallel mode, no pipeline_parallel, no sharding.
        if (
            not in_pipeline_parallel_mode
            and not in_sharding_parallel_mode
            and (in_tensor_parallel_mode or in_sep_parallel_mode or in_cp_parallel_mode)
        ):
            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)  # return value has no use
                mix_precision_utils.MixPrecisionLayer(guidance_model, dtype=self.amp_dtype)  # return value has no use

            model = fleet.distributed_model(model)
            guidance_model = fleet.distributed_model(guidance_model)
            assert self.optimizer is not None, "Tensor parallel mode need decorate optimizer, pelease init optimizer."
            if self.args.amp_master_grad:
                self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)
                self.optimizer_guidance = mix_precision_utils.MixPrecisionOptimizer(self.optimizer_guidance)
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
            self.optimizer_guidance = fleet.distributed_optimizer(self.optimizer_guidance)

        # stage1 has v1 and v2 version
        if in_sharding_parallel_mode and ShardingOption.SHARD_OP in self.args.sharding:
            print("sharding_parallel_config:", self.args.sharding_parallel_config)
            if "split_param" in self.args.sharding_parallel_config:
                if (
                    hasattr(self.optimizer, "_set_all_gather_overlap_forward")
                    and "enable_stage1_allgather_overlap" in self.args.sharding_parallel_config
                ):
                    self.optimizer._set_all_gather_overlap_forward(True, model)
                    self.optimizer_guidance._set_all_gather_overlap_forward(True, guidance_model)
            else:
                if (
                    hasattr(self.optimizer, "_set_broadcast_overlap")
                    and "enable_stage1_broadcast_overlap" in self.args.sharding_parallel_config
                ):
                    self.optimizer._set_broadcast_overlap(True, model)
                    self.optimizer_guidance._set_broadcast_overlap(True, guidance_model)
        # exit()
        return model, guidance_model

    def autocast_smart_context_manager(self):
        """
        A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation.
        """
        if self.enable_autocast_context_manager:
            custom_black_list = ["reduce_sum", "c_softmax_with_cross_entropy"]
            custom_white_list = []
            if self.args.fp16_opt_level == "O2":
                # https://github.com/PaddlePaddle/Paddle/blob/eb97f4f0adca40b16a309b927e480178beb8ae96/python/paddle/amp/amp_lists.py#L85-L86
                # the lookup_table is in black_list, but in O2, we need it return fp16
                custom_white_list.extend(["lookup_table", "lookup_table_v2"])

            if self.args.amp_custom_white_list is not None:
                custom_white_list.extend(self.args.amp_custom_white_list)
            if self.args.amp_custom_black_list is not None:
                custom_black_list.extend(self.args.amp_custom_black_list)

            ctx_manager = autocast(
                True,
                custom_black_list=set(custom_black_list),
                custom_white_list=set(custom_white_list),
                level=self.args.fp16_opt_level,
                dtype=self.amp_dtype,
            )
        else:
            ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()

        return ctx_manager

    def compute_loss(self, model, inputs, return_outputs=False, generator_turn=True):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        noise = inputs["noise"]
        text_embedding = inputs["text_embedding"]
        uncond_embedding = inputs["uncond_embedding"]
        denoising_dict = inputs["denoising_dict"]
        real_train_dict = inputs["real_train_dict"]
        if generator_turn:
            outputs = model(
                noise,
                text_embedding,
                uncond_embedding,
                visual=self._visual,
                denoising_dict=denoising_dict,
                compute_generator_gradient=self._COMPUTE_GENERATOR_GRADIENT,
                real_train_dict=real_train_dict,
                generator_turn=True,
                guidance_turn=False,
            )
        else:
            guidance_data_dict = inputs["guidance_data_dict"]

            guidance_loss_dict, guidance_log_dict = self.model(
                noise,
                text_embedding,
                uncond_embedding,
                visual=visual,
                denoising_dict=denoising_dict,
                real_train_dict=real_train_dict,
                compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
                generator_turn=False,
                guidance_turn=True,
                guidance_data_dict=guidance_data_dict,
            )

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if isinstance(outputs, dict):
            loss = outputs["loss"]
        elif isinstance(outputs, tuple):
            loss = outputs[0]
        else:
            loss = outputs

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Layer, inputs, generator_turn=True) -> paddle.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Layer`):
                The model to train.
            inputs (`Dict[str, Union[paddle.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `paddle.Tensor`: The tensor with training loss on this batch.
        """
        if self.args.pipeline_parallel_degree > 1:
            return self.training_pipeline_step(model, inputs)

        model.train()

        noise = inputs["noise"]
        text_embedding = inputs["text_embedding"]
        uncond_embedding = inputs["uncond_embedding"]
        denoising_dict = inputs["denoising_dict"]
        real_train_dict = inputs["real_train_dict"]

        if real_train_dict["images"].shape[-1] != self.dmd2_args.latent_resolution:
            real_train_dict["images"] = paddle.nn.functional.interpolate(
                real_train_dict["images"], size=(self.dmd2_args.latent_resolution, self.dmd2_args.latent_resolution)
            )
        if generator_turn:
            generator_loss_dict, loss_log = model(
                noise,
                text_embedding,
                uncond_embedding,
                visual=self._visual,
                denoising_dict=denoising_dict,
                compute_generator_gradient=self._COMPUTE_GENERATOR_GRADIENT,
                real_train_dict=real_train_dict,
                generator_turn=True,
                guidance_turn=False,
            )

            generator_loss = paddle.zeros([1])
            if self._COMPUTE_GENERATOR_GRADIENT:
                if not self.dmd2_args.gan_alone:
                    generator_loss += generator_loss_dict["loss_dm"] * self.dmd2_args.dm_loss_weight

                if self.dmd2_args.cls_on_clean_image and self.dmd2_args.gen_cls_loss:
                    generator_loss += generator_loss_dict["gen_cls_loss"] * self.dmd2_args.gen_cls_loss_weight

                if self.args.gradient_accumulation_steps > 1 and not self._enable_delay_scale_loss():
                    generator_loss = generator_loss / self.args.gradient_accumulation_steps

                if self.do_grad_scaling:
                    self.scaler.scale(generator_loss).backward()
                else:
                    generator_loss.backward()
            loss = generator_loss
            loss_log["generator_loss_dict"] = generator_loss_dict
        else:
            guidance_data_dict = inputs["guidance_data_dict"]

            guidance_loss_dict, loss_log = model(
                noise,
                text_embedding,
                uncond_embedding,
                visual=self._visual,
                denoising_dict=denoising_dict,
                real_train_dict=real_train_dict,
                compute_generator_gradient=self._COMPUTE_GENERATOR_GRADIENT,
                generator_turn=False,
                guidance_turn=True,
                guidance_data_dict=guidance_data_dict,
            )

            guidance_loss = 0.0

            guidance_loss += guidance_loss_dict["loss_fake_mean"]

            if self.dmd2_args.cls_on_clean_image:
                guidance_loss += guidance_loss_dict["guidance_cls_loss"] * self.dmd2_args.guidance_cls_loss_weight

            if self.args.gradient_accumulation_steps > 1 and not self._enable_delay_scale_loss():
                guidance_loss = guidance_loss / self.args.gradient_accumulation_steps

            if self.do_grad_scaling:
                self.scaler.scale(guidance_loss).backward()
            else:
                guidance_loss.backward()

            loss = guidance_loss
            loss_log["guidance_loss_dict"] = guidance_loss_dict

        return loss.detach(), loss_log

    def save_model(
        self,
        output_dir: Optional[str] = None,
        merge_tensor_parallel: Optional[bool] = False,
    ):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if PREFIX_CHECKPOINT_DIR in os.path.split(output_dir)[-1]:
            signal_dir = os.path.join(self.args.output_signal_dir, os.path.split(output_dir)[-1])
        else:
            signal_dir = self.args.output_signal_dir

        if ShardingOption.FULL_SHARD in self.args.sharding:
            self.model_wrapped.get_all_parameters(convert2cpu=True)

        if self.args.should_save_model_state:
            self._save(output_dir=output_dir, merge_tensor_parallel=merge_tensor_parallel)
        else:
            if self.args.unified_checkpoint and "async_save" in self.args.unified_checkpoint_config:
                os.makedirs(signal_dir, exist_ok=True)
                if self.is_in_train:
                    global_rank = paddle.distributed.get_rank() if paddle.distributed.get_world_size() > 1 else -1
                    paddle.save(global_rank, os.path.join(signal_dir, f".model_weight.done.{global_rank}"))

        if strtobool(os.getenv("FLAG_LLM_PDC", "False")):
            # save model_done file to ensure model is complete
            if (
                self.args.should_save_model_state
                and self.args.should_save
                and not ("async_save" in self.args.unified_checkpoint_config)
            ):
                # For ckpt integrity
                paddle.save(self.state.global_step, os.path.join(output_dir, ".model_done"))
        if (
            self.args.unified_checkpoint
            and "async_save" in self.args.unified_checkpoint_config
            and not self.is_in_train
        ):
            print("unified_checkpoint", signal_dir)
            os.makedirs(signal_dir, exist_ok=True)
            global_rank = paddle.distributed.get_rank() if paddle.distributed.get_world_size() > 1 else -1
            paddle.save(self.state.global_step, os.path.join(signal_dir, f".model_weight.done.{global_rank}"))

    def _filter_moe_no_sync_optimizer_params(self):
        """
        filter optimizer params which should not sync
        """
        state_dict = self.model.state_dict()
        optimzier_state_dict = self.optimizer.state_dict()
        filter_optimzier_state_dict = OrderedDict()
        param_names_in_master_weights = list(optimzier_state_dict["master_weights"].keys()) if self.args.bf16 else []
        filter_optimzier_state_dict["master_weights"] = OrderedDict()
        for _, v in state_dict.items():
            if getattr(v, "no_sync", False):
                if v.name in param_names_in_master_weights:
                    filter_optimzier_state_dict["master_weights"][v.name] = optimzier_state_dict["master_weights"][
                        v.name
                    ]
                for op_k, op_v in optimzier_state_dict.items():
                    if op_k.startswith(v.name):
                        filter_optimzier_state_dict[op_k] = op_v
        return filter_optimzier_state_dict

    def _ordered_save(self, state_dict, save_path):
        group_size = self.args.ordered_save_group_size
        hcg = fleet.get_hybrid_communicate_group()
        if hcg.get_sharding_parallel_world_size() > 1 or hcg.get_model_parallel_world_size() <= 1:
            return paddle.save(state_dict, save_path)

        mp_group = hcg.get_model_parallel_group()
        ranks = list(mp_group.ranks)
        n = len(ranks)

        group_num = (n + group_size - 1) // group_size
        groups = []
        for i in range(group_num):
            groups.append([ranks[j] for j in range(i, n, group_num)])

        for group in groups:
            if dist.get_rank() in group:
                paddle.save(state_dict, save_path)
            dist.barrier(mp_group)

    def _save_checkpoint(self, model, metrics=None):
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"
        self.runtime_timer.start("checkpoint saving time")

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self.args.output_dir
        run_signal_dir = self.args.output_signal_dir

        output_dir = os.path.join(run_dir, checkpoint_folder)
        signal_dir = os.path.join(run_signal_dir, checkpoint_folder)

        if isinstance(self.model, LoRAModel) and (self.model.quantized or self.args.pipeline_parallel_degree > 1):
            self.save_model(output_dir)
        elif isinstance(self.model, LoRAModel) or isinstance(self.model, PrefixModelForCausalLM):
            print("Is_LoRAModel", output_dir)
            self.save_model(output_dir, True)
        else:
            print("Not_LoRAModel", output_dir)
            self.save_model(output_dir)

        # only save model state dict, ignore optimizer and scheduler
        if not self.args.ignore_save_lr_and_optim:
            optimizer_name = _add_variant(OPTIMIZER_NAME, self.args.optimizer_name_suffix)
            saved_signal_path = os.path.join(output_dir, f"saved_signal_{dist.get_rank()}")

            if self.args.use_hybrid_parallel:
                if self.dp_group.rank <= 0 or self.args.use_expert_parallel:
                    os.makedirs(output_dir, exist_ok=True)
                    logger.info("Saving optimizer files.")
                    if self.args.unified_checkpoint:
                        self.unified_checkpoint_handler.save_unified_optimizer(
                            self.model,
                            self.optimizer,
                            output_dir,
                            signal_dir,
                        )
                    else:
                        if self.dp_group.rank > 0:  # this should only work for MoE saving
                            self._save_ckpt_func(
                                self._filter_moe_no_sync_optimizer_params(),
                                os.path.join(output_dir, optimizer_name),
                                saved_signal_path,
                            )

                        else:
                            state_dict = self.optimizer.state_dict()
                            save_path = os.path.join(output_dir, optimizer_name)
                            if self.args.use_async_save:
                                assert not strtobool(os.getenv("FLAG_LLM_PDC", "False")), "Dont support FLAG_LLM_PDC"
                                self._async_optimizer_saver.run(
                                    state_dict, save_path, saved_signal_path=saved_signal_path
                                )
                            else:
                                self._save_ckpt_func(state_dict, save_path, saved_signal_path)
                else:
                    if self.args.unified_checkpoint and "async_save" in self.args.unified_checkpoint_config:
                        global_rank = paddle.distributed.get_rank() if paddle.distributed.get_world_size() > 1 else -1
                        os.makedirs(signal_dir, exist_ok=True)
                        paddle.save(global_rank, os.path.join(signal_dir, f".optimizer_weight.done.{global_rank}"))
                        if "skip_save_model_weight" not in self.args.unified_checkpoint_config:
                            paddle.save(global_rank, os.path.join(signal_dir, f".master_weight.done.{global_rank}"))
            if self.args.should_save or self.args.use_expert_parallel:
                if not self.args.use_hybrid_parallel:
                    logger.info("Saving optimizer files.")
                    if self.args.unified_checkpoint:
                        self.unified_checkpoint_handler.save_unified_optimizer(
                            self.model,
                            self.optimizer,
                            output_dir,
                            signal_dir,
                        )
                    else:
                        if self.args.data_parallel_rank > 0 and self.args.use_expert_parallel:
                            self._save_ckpt_func(
                                self._filter_moe_no_sync_optimizer_params(),
                                os.path.join(output_dir, optimizer_name),
                                saved_signal_path,
                            )
                        else:
                            self._save_ckpt_func(
                                self.optimizer.state_dict(),
                                os.path.join(output_dir, optimizer_name),
                                saved_signal_path,
                            )

                # FIXME: maybe only save one copy
                paddle.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

                if self.do_grad_scaling:
                    paddle.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
            else:
                if self.args.unified_checkpoint and not self.args.use_hybrid_parallel:
                    if "async_save" in self.args.unified_checkpoint_config:
                        global_rank = paddle.distributed.get_rank() if paddle.distributed.get_world_size() > 1 else -1
                        os.makedirs(signal_dir, exist_ok=True)
                        paddle.save(global_rank, os.path.join(signal_dir, f".optimizer_weight.done.{global_rank}"))
                        if "skip_save_model_weight" not in self.args.unified_checkpoint_config:
                            paddle.save(global_rank, os.path.join(signal_dir, f".master_weight.done.{global_rank}"))

        self.runtime_timer.stop()
        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cuda": paddle.get_rng_state(),
            "cpu": paddle.framework.core.default_cpu_generator().get_state(),
        }
        if self.args.use_hybrid_parallel:
            rng_states[
                "hybrid_parallel_rng_state_tracker"
            ] = fleet.meta_parallel.get_rng_state_tracker().get_states_tracker()

        if self.args.world_size > 1:
            rng_states_list = []
            paddle.distributed.all_gather_object(rng_states_list, rng_states)
            if self.args.should_save:
                os.makedirs(output_dir, exist_ok=True)
                paddle.save(rng_states_list, os.path.join(output_dir, f"rng_state_{self.args.world_size}.pth"))
        else:
            os.makedirs(output_dir, exist_ok=True)
            paddle.save(rng_states, os.path.join(output_dir, "rng_state.pth"))

        # Maybe delete some older checkpoints.
        # For hybrid parallel training, the checkpoint files maybe on different node.
        need_to_rotate_checkpoints = False
        if self.args.use_hybrid_parallel:
            if self.dp_group.rank <= 0 or self.args.use_expert_parallel:
                need_to_rotate_checkpoints = True
        else:
            need_to_rotate_checkpoints = self.args.should_save_model_state

        # Delete only by one process
        need_to_rotate_checkpoints = need_to_rotate_checkpoints and self.args.local_rank == 0
        if need_to_rotate_checkpoints:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
            self._rotate_checkpoints(use_mtime=True, output_dir=run_signal_dir)

        if strtobool(os.getenv("FLAG_LLM_PDC", "False")) and not ("async_save" in self.args.unified_checkpoint_config):
            # save checkpoint_done file to ensure checkpoint is complete
            if self.args.should_save_model_state and self.args.should_save:
                # For ckpt integrity
                paddle.save(self.state.global_step, os.path.join(output_dir, ".checkpoint_done"))

    def set_optimizer_grouped_parameters(self, optimizer_grouped_parameters=None):
        """
        set optimizer grouped parameters:

        you can set optimizer_grouped_parameters with whatever argments on whatever parameters to train.
        """
        self.optimizer_grouped_parameters = optimizer_grouped_parameters

    def disable_autocast_context_manager(self):
        """
        For pure fp16 or pure bf16 training, the paddle.amp.autocast is annoy for always cast fp32 to fp16.
        if you networks cast fp16 to fp32 manually to get higher precision, autocast make it not work, since it cast fp32 to fp16 back.

        """
        assert self.args.fp16_opt_level == "O2", "disable_autocast_context_manager should only work for pure fp16/bf16"
        self.enable_autocast_context_manager = False

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
            and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            # ignore_errors for shared disks between train nodes.
            shutil.rmtree(checkpoint, ignore_errors=True)

    def _save(
        self,
        output_dir: Optional[str] = None,
        state_dict=None,
        merge_tensor_parallel=False,
    ):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # signal_dir is used for asynchronous saving situations.
        signal_dir = self.args.output_signal_dir
        if self.args.unified_checkpoint and "async_save" in self.args.unified_checkpoint_config:
            if PREFIX_CHECKPOINT_DIR in os.path.split(output_dir)[-1]:
                signal_dir = os.path.join(signal_dir, os.path.split(output_dir)[-1])
            os.makedirs(signal_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint finish signal to {signal_dir}")

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        if (
            strtobool(os.getenv("FLAG_LLM_PDC", "False"))
            and paddle.distributed.get_rank() == 0
            and self.args.unified_checkpoint
            and "async_save" in self.args.unified_checkpoint_config
        ):
            world_size = paddle.distributed.get_world_size()
            save_info = {
                "world_size": world_size,
                "ignore_save_lr_and_optim": self.args.ignore_save_lr_and_optim,
                "skip_save_model_weight": "skip_save_model_weight" in self.args.unified_checkpoint_config,
                "remove_master_weight": "remove_master_weight" in self.args.unified_checkpoint_config,
            }
            if os.path.exists(
                os.path.join(self.args.output_signal_dir, "async_save_info.json")
            ):  # afs cannot overwrite
                os.remove(os.path.join(self.args.output_signal_dir, "async_save_info.json"))
            with open(os.path.join(self.args.output_signal_dir, "async_save_info.json"), "w") as f:
                json.dump(save_info, f)

        if self.args.should_save:
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            # Good practice: save your training arguments together with the trained model
            paddle.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        if self.args.unified_checkpoint:
            unified_checkpoint_config_backup = self.args.unified_checkpoint_config
            # backup and remove unified_checkpoint_config for not trine stage
            if not self.is_in_train:
                self.args.unified_checkpoint_config = []

            self.unified_checkpoint_handler.save_unified_checkpoint(self.model, self.optimizer, output_dir, signal_dir)

            # recover unified_checkpoint_config for not trine stage
            if not self.is_in_train:
                self.args.unified_checkpoint_config = unified_checkpoint_config_backup

            return

        merge_tensor_parallel = merge_tensor_parallel and self.args.use_hybrid_parallel
        # peft model
        if (
            isinstance(self.model, LoRAModel)
            or isinstance(self.model, PrefixModelForCausalLM)
            or isinstance(self.model, VeRAModel)
            or isinstance(self.model, LoKrModel)
            or isinstance(self.model, ReFTModel)
        ):
            self.model.save_pretrained(
                output_dir,
                variant=self.args.weight_name_suffix,
                save_function=self._save_ckpt_func,
                merge_tensor_parallel=merge_tensor_parallel,
                is_main_process=self.args.should_save,
                max_shard_size="1024GB",
            )
        # TODO: @ZHUI unifiy unwrap_model(self.model) and self.model
        elif not isinstance(self.model, PretrainedModel):
            if isinstance(unwrap_model(self.model), PretrainedModel):
                if self.args.should_save_sharding_stage1_model:
                    config_to_save = None
                    state_dict, config_to_save, weight_name_suffix = self.sharding_io.manipulate_state_dict_and_config(
                        unwrap_model(self.model), merge_tensor_parallel=merge_tensor_parallel
                    )
                    unwrap_model(self.model).save_pretrained(
                        output_dir,
                        state_dict=state_dict,
                        config_to_save=config_to_save,
                        merge_tensor_parallel=merge_tensor_parallel,
                        variant=weight_name_suffix,
                        save_function=self._save_ckpt_func,
                        is_main_process=self.args.should_save,
                        max_shard_size="1024GB",
                    )
                else:
                    unwrap_model(self.model).save_pretrained(
                        output_dir,
                        merge_tensor_parallel=merge_tensor_parallel,
                        variant=self.args.weight_name_suffix,
                        save_function=self._save_ckpt_func,
                        is_main_process=self.args.should_save,
                        max_shard_size="1024GB",
                    )
            else:
                logger.info("Trainer.model is not a `PretrainedModel`, only saving its state dict.")
                if merge_tensor_parallel:
                    logger.warning("Trainer.model is not a `PretrainedModel`, not suppor for merge_tensor_parallel.")
                if state_dict is None:
                    state_dict = self.model.state_dict()

                if self.args.should_save_sharding_stage1_model:
                    state_dict, _, _ = self.sharding_io.manipulate_state_dict_and_config(
                        unwrap_model(self.model), merge_tensor_parallel=False, state_dict=state_dict
                    )
                    variant = _add_variant(PADDLE_WEIGHTS_NAME, self.args.sharded_name_suffix())
                else:
                    variant = _add_variant(PADDLE_WEIGHTS_NAME, self.args.weight_name_suffix)

                self._save_ckpt_func(state_dict, os.path.join(output_dir, variant))
        else:
            if isinstance(self.model, PretrainedModel) and self.args.should_save_sharding_stage1_model:
                config_to_save = None
                state_dict, config_to_save, weight_name_suffix = self.sharding_io.manipulate_state_dict_and_config(
                    self.model, merge_tensor_parallel=merge_tensor_parallel
                )
                self.model.save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    config_to_save=config_to_save,
                    merge_tensor_parallel=merge_tensor_parallel,
                    variant=weight_name_suffix,
                    save_function=self._save_ckpt_func,
                    is_main_process=self.args.should_save,
                    max_shard_size="1024GB",
                )
            else:
                self.model.save_pretrained(
                    output_dir,
                    merge_tensor_parallel=merge_tensor_parallel,
                    variant=self.args.weight_name_suffix,
                    save_function=self._save_ckpt_func,
                    is_main_process=self.args.should_save,
                    max_shard_size="1024GB",
                )
        if self.args.should_save_sharding_stage1_model:
            self.sharding_io.save_distributed_model_meta(output_dir)

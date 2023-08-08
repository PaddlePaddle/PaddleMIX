# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import time

import torch
from torch.utils.data import DataLoader
from transformers.integrations import INTEGRATION_TO_CALLBACK, rewrite_logs
from transformers.modeling_utils import unwrap_model
from transformers.trainer import *
from transformers.trainer_callback import CallbackHandler
from transformers.utils.logging import get_logger

from .text_image_pair_dataset import TextImagePair, worker_init_fn

logger = get_logger("transformers")


def on_log(self,
           args: TrainingArguments,
           state: TrainerState,
           control: TrainerControl,
           logs,
           **kwargs):
    control.should_log = False
    return self.call_event("on_log", args, state, control, logs=logs, **kwargs)


CallbackHandler.on_log = on_log

SAFE_WEIGHTS_NAME = "model.safetensors"

WEIGHTS_NAME = "pytorch_model.bin"

TRAINING_ARGS_NAME = "training_args.bin"


class VisualDLWithImageCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [VisualDL](https://www.paddlepaddle.org.cn/paddle/visualdl).
    Args:
        vdl_writer (`LogWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, vdl_writer=None):
        has_visualdl = True
        try:
            import visualdl
        except ImportError:
            visualdl
            has_visualdl = False
        if not has_visualdl:
            raise RuntimeError(
                "VisualDLWithImageCallback requires visualdl to be installed. Please install visualdl."
            )
        if has_visualdl:
            try:
                from visualdl import LogWriter

                self._LogWriter = LogWriter
            except ImportError:
                self._LogWriter = None
        else:
            self._LogWriter = None
        self.vdl_writer = vdl_writer

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if hasattr(model, "on_train_batch_end"):
            model.on_train_batch_end()
        if args.image_logging_steps > 0 and state.global_step % args.image_logging_steps == 0:
            control.should_log = True

    def _init_summary_writer(self, args, log_dir=None):
        log_dir = log_dir or args.logging_dir
        if self._LogWriter is not None:
            self.vdl_writer = self._LogWriter(logdir=log_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = None

        if self.vdl_writer is None:
            self._init_summary_writer(args, log_dir)

        if self.vdl_writer is not None:
            self.vdl_writer.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    self.vdl_writer.add_text("model_config", model_config_json)

            if hasattr(self.vdl_writer, "add_hparams"):
                self.vdl_writer.add_hparams(
                    args.to_sanitized_dict(), metrics_list=[])

    def on_log(self, args, state, control, logs=None, **kwargs):
        # log image on each node
        inputs = kwargs.get("inputs", None)
        model = kwargs.get("model", None)
        image_logs = {}
        if (inputs is not None and model is not None and
                args.image_logging_steps > 0 and
                state.global_step % args.image_logging_steps == 0):
            max_batch = 4 if args.resolution > 256 else 8

            image_logs["reconstruction"] = model.decode_image(
                pixel_values=inputs["pixel_values"].to(args.device),
                max_batch=max_batch)
            image_logs["ddim-samples-1.0"] = model.log_image(
                input_ids=inputs["input_ids"].to(args.device),
                guidance_scale=1.0,
                height=args.resolution,
                width=args.resolution,
                max_batch=max_batch, )
            image_logs["ddim-samples-7.5"] = model.log_image(
                input_ids=inputs["input_ids"].to(args.device),
                guidance_scale=7.5,
                height=args.resolution,
                width=args.resolution,
                max_batch=max_batch, )

        if not state.is_world_process_zero:
            return

        if self.vdl_writer is None:
            self._init_summary_writer(args)

        if self.vdl_writer is not None:
            logs = rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.vdl_writer.add_scalar(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of VisualDL's writer.add_scalar() "
                        "is incorrect so we dropped this attribute.")
            # log images
            for k, v in image_logs.items():
                self.vdl_writer.add_image(
                    k, v, state.global_step, dataformats="NHWC")
            self.vdl_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.vdl_writer:
            self.vdl_writer.close()
            self.vdl_writer = None


class AverageStatistical(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_cnt = 0
        self.time = 0

    def record(self, val, cnt=1):
        self.time += val
        self.total_cnt += cnt

    def get_average(self):
        if self.total_cnt == 0:
            return 0

        return self.time / self.total_cnt

    def get_average_per_sec(self):
        if self.time == 0.0:
            return 0.0

        return float(self.total_cnt) / self.time

    def get_total_cnt(self):
        return self.total_cnt

    def get_total_time(self):
        return self.time


class BenchmarkCallback(TrainerCallback):
    def __init__(self, benchmark=True, **kwargs):
        self.benchmark = benchmark

    def on_train_begin(self, args, state, control, **kwargs):
        assert args.gradient_accumulation_steps == 1 and not args.do_eval and not args.do_predict
        if self.benchmark:
            self.reader_cost_avg = AverageStatistical()

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.benchmark:
            self.epoch_start = time.time()
            self.batch_start = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        if self.benchmark:
            self.reader_cost_avg.record(time.time() - self.batch_start)

    def on_step_end(self, args, state, control, **kwargs):
        if self.benchmark:
            self.batch_start = time.time()
            if control.should_log:
                self.maybe_log_save_evaluate_start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.benchmark:
            if logs is not None and "interval_steps_per_second" in logs:
                self.batch_start = self.batch_start + (
                    time.time() - self.maybe_log_save_evaluate_start)
                ips = logs["interval_steps_per_second"] * args.train_batch_size
                avg_batch_cost = 1 / logs["interval_steps_per_second"]
                logger.info(
                    "global step %d / %d, loss: %f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, avg_samples: %.5f, ips: %.5f sample/sec"
                    % (
                        state.global_step,
                        state.max_steps,
                        logs["loss"],
                        self.reader_cost_avg.get_average(),
                        avg_batch_cost,
                        args.train_batch_size,
                        ips, ))
                self.reader_cost_avg.reset()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.benchmark:
            train_epoch_cost = time.time() - self.epoch_start
            logger.info("train epoch: %d, epoch_cost: %.5f s" %
                        (state.epoch, train_epoch_cost))


# register visualdl_with_image
INTEGRATION_TO_CALLBACK.update({"custom_visualdl": VisualDLWithImageCallback})


class StableDiffusionTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.args.benchmark is not None:
            self.add_callback(BenchmarkCallback(benchmark=self.args.benchmark))
            if self.args.benchmark:
                if self.args.disable_tqdm:
                    self.pop_callback(PrinterCallback)
                else:
                    self.pop_callback(ProgressCallback)

    def compute_loss(self, model, inputs, return_outputs=False):
        # move eval 
        model.vae.eval()
        if not model.model_args.train_text_encoder:
            model.text_encoder.eval()
        loss = model(**inputs)
        return loss

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if isinstance(self.train_dataset, TextImagePair):
            return DataLoader(
                self.train_dataset,
                batch_size=self._train_batch_size,
                num_workers=self.args.dataloader_num_workers,
                worker_init_fn=None
                if self.args.world_size <= 1 else worker_init_fn, )
        else:
            return super().get_train_dataloader()

    def _inner_training_loop(self,
                             batch_size=None,
                             args=None,
                             resume_from_checkpoint=None,
                             trial=None,
                             ignore_keys_for_eval=None):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        logger.debug(
            f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0)
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs *
                                      num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(
                    train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}")

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps and args.logging_steps < 1:
            args.logging_steps = math.ceil(max_steps * args.logging_steps)
        if args.eval_steps and args.eval_steps < 1:
            args.eval_steps = math.ceil(max_steps * args.eval_steps)
        if args.save_steps and args.save_steps < 1:
            args.save_steps = math.ceil(max_steps * args.save_steps)

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch).")
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None and
            self.sharded_ddp != ShardedDDPOption.SIMPLE or
            is_sagemaker_mp_enabled() or self.fsdp is not None)

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # Fairscale Sharded DDP, FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(
                        self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler)

        if self.is_fsdp_enabled:
            self.model = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        self._globalstep_last_start_time = time.time()
        # deepspeed ckpt loading
        if resume_from_checkpoint is not None and self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(self.model_wrapped,
                                      resume_from_checkpoint)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(
            f"  Instantaneous batch size per device = {self._train_batch_size:,}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(
            f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                if skip_first_batches is None:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch. If this takes a lot of time,"
                        " you can install the latest version of Accelerate with `pip install -U accelerate`.You can"
                        " also add the `--ignore_data_skip` flag to your launch command, but you will resume the"
                        " training on data already seen by your model.")
                else:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )
                if self.is_local_process_zero(
                ) and not args.disable_tqdm and skip_first_batches is None:
                    steps_trained_progress_bar = tqdm(
                        total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description(
                        "Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state,
                                                            self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(
                    train_dataloader, "sampler") and isinstance(
                        train_dataloader.sampler, RandomSampler)
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(
                    train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(
                    train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(
                    train_dataloader,
                    [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (len(epoch_iterator)
                              if len_dataloader is not None else
                              args.max_steps * args.gradient_accumulation_steps)
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if skip_first_batches is not None and steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(
                    epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (args.logging_nan_inf_filter and
                        not is_torch_tpu_available() and
                    (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step -
                                          self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # should this be under the accumulate context manager?
                # the `or` condition of `steps_in_epoch <= args.gradient_accumulation_steps` is not covered
                # in accelerate
                if total_batched_samples % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps and
                    (step + 1) == steps_in_epoch):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce(
                                    "sum",
                                    gradients,
                                    scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm, )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm, )

                    # Optimizer step
                    optimizer_was_run = True
                    if is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()
                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped

                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(
                                self.lr_scheduler,
                                torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped
                                                ) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control)

                    self._maybe_log_save_evaluate(
                        tr_loss,
                        model,
                        trial,
                        epoch,
                        ignore_keys_for_eval,
                        inputs=inputs)
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state,
                                                              self.control)
            self._maybe_log_save_evaluate(
                tr_loss,
                model,
                trial,
                epoch,
                ignore_keys_for_eval,
                inputs=inputs)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
                    )
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state,
                                                          self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch,
                                 ignore_keys_for_eval, **kwargs):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (
                self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            logs["global_step"] = int(self.state.global_step)

            total_train_batch_size = (self.args.train_batch_size *
                                      self.args.gradient_accumulation_steps *
                                      self.args.world_size)
            num_steps = self.state.global_step - self._globalstep_last_logged
            self.store_flos()
            logs.update(
                speed_metrics(
                    "interval",
                    self._globalstep_last_start_time,
                    num_samples=total_train_batch_size * num_steps,
                    num_steps=num_steps, ))

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self._globalstep_last_start_time = time.time()

            self.log(logs, **kwargs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items(
                ):
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}", )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state,
                                                         self.control)

    def log(self, logs: Dict[str, float], **kwargs) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = { ** logs, ** {"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if self.args.only_save_updated_model:
            unwraped_model = unwrap_model(self.model)
            logger.info(f"Saving unet checkpoint to {output_dir}/unet")
            unwraped_model.unet.save_pretrained(
                os.path.join(output_dir, "unet"))

            if unwraped_model.use_ema:
                logger.info(f"Saving ema unet checkpoint to {output_dir}/unet")
                with unwraped_model.ema_scope():
                    unwraped_model.unet.save_pretrained(
                        os.path.join(output_dir, "unet"), variant="ema")

            if unwraped_model.train_text_encoder:
                logger.info(
                    f"Saving text encoder checkpoint to {output_dir}/text_encoder"
                )
                unwraped_model.text_encoder.save_pretrained(
                    os.path.join(output_dir, "text_encoder"))
        else:
            logger.info(f"Saving model checkpoint to {output_dir}")
            if state_dict is None:
                state_dict = self.model.state_dict()

            logger.info(
                "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
            )
            if self.args.save_safetensors:
                import safetensors

                safetensors.torch.save_file(
                    state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME))
            else:
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

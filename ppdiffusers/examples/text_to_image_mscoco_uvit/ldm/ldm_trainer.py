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

import contextlib
import os
import sys
import time

import paddle
import paddle.amp.auto_cast as autocast
from paddlenlp.trainer import PrinterCallback, ProgressCallback, Trainer
from paddlenlp.trainer.integrations import (
    INTEGRATION_TO_CALLBACK,
    TrainerCallback,
    VisualDLCallback,
    rewrite_logs,
)
from paddlenlp.transformers.model_utils import _add_variant
from paddlenlp.utils import profiler
from paddlenlp.utils.log import logger

from ppdiffusers.optimization import get_scheduler
from ppdiffusers.training_utils import unwrap_model

from .text_to_image_dataset import worker_init_fn

PADDLE_WEIGHTS_NAME = "model_state.pdparams"
TRAINING_ARGS_NAME = "training_args.bin"

use_tensorboard = False
if use_tensorboard:
    from tensorboardX import SummaryWriter


class VisualDLWithImageCallback(VisualDLCallback):
    def autocast_smart_context_manager(self, args):
        if args.fp16 or args.bf16:
            amp_dtype = "float16" if args.fp16 else "bfloat16"
            custom_black_list = ["reduce_sum", "c_softmax_with_cross_entropy"]
            custom_white_list = []
            if args.fp16_opt_level == "O2":
                # https://github.com/PaddlePaddle/Paddle/blob/eb97f4f0adca40b16a309b927e480178beb8ae96/python/paddle/amp/amp_lists.py#L85-L86
                # the lookup_table is in black_list, but in O2, we need it return fp16
                custom_white_list.extend(["lookup_table", "lookup_table_v2"])

            if hasattr(args, "amp_custom_white_list"):
                if args.amp_custom_white_list is not None:
                    custom_white_list.extend(args.amp_custom_white_list)
            if hasattr(args, "amp_custom_black_list"):
                if args.amp_custom_black_list is not None:
                    custom_black_list.extend(args.amp_custom_black_list)

            ctx_manager = autocast(
                True,
                custom_black_list=set(custom_black_list),
                custom_white_list=set(custom_white_list),
                level=args.fp16_opt_level,
                dtype=amp_dtype,
            )
        else:
            ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()

        return ctx_manager

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if hasattr(model, "on_train_batch_end"):
            model.on_train_batch_end()
        if args.image_logging_steps > 0 and state.global_step % args.image_logging_steps == 0:
            control.should_log = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        # log image on each node
        inputs = kwargs.get("inputs", None)
        model = kwargs.get("model", None)
        image_logs = {}
        if (
            inputs is not None
            and model is not None
            and args.image_logging_steps > 0
            and state.global_step % args.image_logging_steps == 0
        ):
            with self.autocast_smart_context_manager(args):
                max_batch = 4 if args.resolution > 256 else 8
                image_logs["reconstruction"] = model.decode_image(
                    pixel_values=inputs["pixel_values"], max_batch=max_batch
                )
                image_logs["ddim-samples-1.0"] = model.log_image(
                    input_ids=inputs["input_ids"],
                    guidance_scale=1.0,
                    height=args.resolution,
                    width=args.resolution,
                    max_batch=max_batch,
                )
                image_logs["ddim-samples-7.5"] = model.log_image(
                    input_ids=inputs["input_ids"],
                    guidance_scale=7.5,
                    height=args.resolution,
                    width=args.resolution,
                    max_batch=max_batch,
                )

        if not state.is_world_process_zero:
            return

        if self.vdl_writer is None:
            self._init_summary_writer(args)

        base_learning_rate = logs.get("learning_rate", None)
        if base_learning_rate is not None:
            logs["unet_lr"] = base_learning_rate
            if model.train_text_encoder:
                if args.text_encoder_learning_rate != args.unet_learning_rate:
                    logs["unet_lr"] = base_learning_rate * args.unet_learning_rate
                    logs["text_encoder_lr"] = base_learning_rate * args.text_encoder_learning_rate
                else:
                    logs["text_encoder_lr"] = base_learning_rate

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
                        "is incorrect so we dropped this attribute."
                    )
            # log images
            for k, v in image_logs.items():
                self.vdl_writer.add_image(k, v, state.global_step, dataformats="NHWC")
            self.vdl_writer.flush()


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
    def __init__(self, benchmark=True, profiler_options=None):
        self.benchmark = benchmark
        self.profiler_options = profiler_options

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
        if self.profiler_options is not None:
            profiler.add_profiler_step(self.profiler_options)

        if self.benchmark:
            self.batch_start = time.time()
            if control.should_log:
                self.maybe_log_save_evaluate_start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.benchmark:
            if logs is not None and "interval_steps_per_second" in logs:
                self.batch_start = self.batch_start + (time.time() - self.maybe_log_save_evaluate_start)
                ips = logs["interval_steps_per_second"] * args.train_batch_size
                avg_batch_cost = 1 / logs["interval_steps_per_second"]
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
                    "global step %d / %d, loss: %f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, avg_samples: %.5f, ips: %.5f sample/sec, %s %s"
                    % (
                        state.global_step,
                        state.max_steps,
                        logs["loss"],
                        self.reader_cost_avg.get_average(),
                        avg_batch_cost,
                        args.train_batch_size,
                        ips,
                        max_mem_reserved_msg,
                        max_mem_allocated_msg,
                    )
                )
                self.reader_cost_avg.reset()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.benchmark:
            train_epoch_cost = time.time() - self.epoch_start
            logger.info("train epoch: %d, epoch_cost: %.5f s" % (state.epoch, train_epoch_cost))


# register visualdl_with_image
if not use_tensorboard:
    INTEGRATION_TO_CALLBACK.update({"custom_visualdl": VisualDLWithImageCallback})


class LatentDiffusionTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if use_tensorboard:
            self.rank = paddle.distributed.get_rank()
            if self.rank == 0:
                self.writer = SummaryWriter("output/tensorboard")
                self.logstep = 0

        if self.args.benchmark or self.args.profiler_options is not None:
            self.add_callback(
                BenchmarkCallback(
                    benchmark=self.args.benchmark,
                    profiler_options=self.args.profiler_options,
                )
            )
            if self.args.benchmark:
                if self.args.disable_tqdm:
                    self.pop_callback(PrinterCallback)
                else:
                    self.pop_callback(ProgressCallback)

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(**inputs)
        return loss

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.lr_scheduler = get_scheduler(
            name="constant_with_warmup",
            learning_rate=self.args.learning_rate,
            num_warmup_steps=self.args.warmup_steps * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_steps * self.args.gradient_accumulation_steps,
        )

        grad_clip = None
        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            grad_clip = paddle.nn.ClipGradByGlobalNorm(self.args.max_grad_norm)

        self.optimizer = paddle.optimizer.AdamW(
            learning_rate=self.lr_scheduler,
            parameters=self.model.unet.parameters(),  # only unet/uvit
            beta1=self.args.adam_beta1,
            beta2=self.args.adam_beta2,
            weight_decay=self.args.weight_decay,
            epsilon=self.args.adam_epsilon,
            grad_clip=grad_clip,
            multi_precision=self.args.fp16_opt_level == "O2",
        )

    def training_step(self, model, inputs) -> paddle.Tensor:
        model = unwrap_model(model)
        model.train()

        if self.do_grad_scaling:
            if self.args.fp16:
                model = model.float16()
                inputs = {"pixel_values": inputs[0].cast("float16"), "input_ids": inputs[1].cast("float16")}
            elif self.args.bf16:
                # bug now
                model = model.bfloat16()
                inputs = {"pixel_values": inputs[0].cast("bfloat16"), "input_ids": inputs[1].cast("bfloat16")}
        else:
            inputs = {"pixel_values": inputs[0], "input_ids": inputs[1]}

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.do_grad_scaling:
            self.scaler.unscale_(self.optimizer)

        if use_tensorboard:
            loss_value = loss.item()
            grad_norms = get_grad_norm_and_clip(model, self.args.max_grad_norm)
            if self.rank == 0:
                self.writer.add_scalar("train/loss", loss_value, self.logstep)
                self.writer.add_scalar("train/grad_norm", grad_norms.item(), self.logstep)
                self.writer.add_scalar("train/lr_abs", self.lr_scheduler.get_lr(), self.logstep)
                self.logstep += 1
        return loss.detach()

    def _get_learning_rate(self):
        return self.lr_scheduler.get_lr()

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = paddle.io.DistributedBatchSampler(
            self.train_dataset,
            self.args.per_device_train_batch_size,
            num_replicas=None,
            rank=None,
            shuffle=False,
            drop_last=True,
        )
        train_dataloader = paddle.io.DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            num_workers=self.args.dataloader_num_workers,
            use_shared_memory=True,
            worker_init_fn=worker_init_fn,
        )
        return train_dataloader

    def _save_todo(self, output_dir=None, state_dict=None, merge_tensor_parallel=False):
        # TODO: merge_tensor_parallel
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if self.args.only_save_updated_model:
            unwrapped_model = unwrap_model(self.model)
            logger.info(f"Saving unet checkpoint to {output_dir}/unet")
            unwrapped_model.unet.save_pretrained(
                os.path.join(output_dir, "unet"),
                # merge_tensor_parallel=merge_tensor_parallel,
            )

            if unwrapped_model.use_ema:
                logger.info(f"Saving ema unet checkpoint to {output_dir}/unet")
                with unwrapped_model.ema_scope():
                    unwrapped_model.unet.save_pretrained(
                        os.path.join(output_dir, "unet"),
                        # merge_tensor_parallel=merge_tensor_parallel,
                        variant="ema",
                    )

            if unwrapped_model.train_text_encoder:
                logger.info(f"Saving text encoder checkpoint to {output_dir}/text_encoder")
                unwrapped_model.text_encoder.save_pretrained(
                    os.path.join(output_dir, "text_encoder"),
                    # merge_tensor_parallel=merge_tensor_parallel,
                )
        else:
            logger.info(f"Saving model checkpoint to {output_dir}")
            if state_dict is None:
                state_dict = self.model.state_dict()
            paddle.save(
                state_dict,
                os.path.join(
                    output_dir,
                    _add_variant(PADDLE_WEIGHTS_NAME, self.args.weight_name_suffix),
                ),
            )
            if self.args.should_save:
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(output_dir, merge_tensor_parallel=merge_tensor_parallel)
                paddle.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def get_grad_norm_and_clip(model, max_norm, norm_type=2.0, error_if_nonfinite=False):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    parameters = model.parameters()
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm) if max_norm else 0.0
    norm_type = float(norm_type)
    if len(grads) == 0:
        return paddle.to_tensor([0.0])
    if norm_type == float("inf"):
        norms = [g.detach().abs().max() for g in grads]
        total_norm = norms[0] if len(norms) == 1 else paddle.max(paddle.stack(norms))
    else:
        total_norm = paddle.norm(paddle.stack([paddle.norm(g.detach(), norm_type) for g in grads]), norm_type)

    if max_norm is None or max_norm <= 0.0:
        return total_norm

    if error_if_nonfinite and paddle.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = paddle.clip(clip_coef, max=1.0)
    for g in grads:
        clipg = paddle.multiply(g, clip_coef_clamped)
        g.set_value(clipg)

    return total_norm

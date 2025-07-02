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
import random
import sys
import time

import numpy as np
import paddle
import paddle.amp.auto_cast as autocast
from paddle.distributed import fleet
from paddle.io import get_worker_info
from paddlenlp.trainer.argparser import strtobool
from paddlenlp.trainer.auto_trainer import (
    DIST_CKPT_PATH,
    FREE_SVAE_LOAD_KEY_PATTERNS,
    MODEL_NAME,
    OPTIMIZER_NAME,
    AutoTrainer,
)
from paddlenlp.trainer.integrations import (
    INTEGRATION_TO_CALLBACK,
    TrainerCallback,
    VisualDLCallback,
    rewrite_logs,
)
from paddlenlp.trainer.trainer import (
    SCALER_NAME,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
)
from paddlenlp.trainer.trainer_utils import PREFIX_CHECKPOINT_DIR
from paddlenlp.utils import profiler
from paddlenlp.utils.log import logger

use_tensorboard = False
if use_tensorboard:
    pass


def worker_init_fn(_):
    """
    初始化函数，用于每个工作者的初始化。

    该函数会获取当前工作者的信息（包括数据集、本地排名、全局排名和工作者ID），并根据这些信息将数据集中的文件ID分配给不同的工作者进行处理。

    返回值是一个随机种子，用于设置每个工作者的随机数生成器状态。

    Args:
        _ (None): 无参数，仅为了与DataLoader的worker_init_fn函数匹配。

    Returns:
        int: 一个随机种子，用于设置每个工作者的随机数生成器状态。
    """
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id

    hcg = getattr(fleet.fleet, "_hcg", None)
    if hcg is not None:
        hcg = fleet.get_hybrid_communicate_group()

    # 初始化默认值
    # world_size = 1
    local_rank = 0

    # 检查是否处于分布式环境中，并且hcg不为None
    if paddle.distributed.get_world_size() > 1 and hcg:
        # dp_size = hcg.get_data_parallel_world_size()
        dp_rank = hcg.get_data_parallel_rank()

        sd_size = hcg.get_sharding_parallel_world_size()
        sd_rank = hcg.get_sharding_parallel_rank()

        # world_size = sd_size * dp_size
        local_rank = dp_rank * sd_size + sd_rank

    num_workers = worker_info.num_workers
    worker_id = worker_info.id
    worker_global_id = local_rank * num_workers + worker_id

    dataset.rng = np.random.RandomState(worker_global_id)
    return np.random.seed(np.random.get_state()[1][0] + worker_id)


class VisualDLWithImageCallback(VisualDLCallback):
    def autocast_smart_context_manager(self, args):
        if args.fp16 or args.bf16:
            amp_dtype = "float16" if args.fp16 else "bfloat16"
            ctx_manager = autocast(
                True,
                custom_black_list=[
                    "reduce_sum",
                    "c_softmax_with_cross_entropy",
                ],
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
                image_logs["reconstruction"] = inputs["latents"].transpose([0, 2, 3, 1]).numpy().round()
                image_logs["ddim-samples-1.0"] = model.log_image(
                    input_ids=None,
                    guidance_scale=1.0,
                    class_labels=inputs["label_id"],
                    height=args.resolution,
                    width=args.resolution,
                    max_batch=8,
                )
                image_logs["ddim-samples-4.0"] = model.log_image(
                    input_ids=None,
                    guidance_scale=4.0,
                    class_labels=inputs["label_id"],
                    height=args.resolution,
                    width=args.resolution,
                    max_batch=8,
                )

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
                logger.info(
                    "global step %d / %d, loss: %f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, avg_samples: %.5f, ips: %.5f sample/sec"
                    % (
                        state.global_step,
                        state.max_steps,
                        logs["loss"],
                        self.reader_cost_avg.get_average(),
                        avg_batch_cost,
                        args.train_batch_size,
                        ips,
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


def unwrap_model(model):
    """
    解包模型，返回最底层的模型。
    如果模型是被多个层包装的，则递归地进行解包。

    Args:
        model (Union[tf.keras.Model, tf.keras.layers.Layer]): 需要解包的模型或层。

    Returns:
        Union[tf.keras.Model, tf.keras.layers.Layer]: 最底层的模型或层。
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "_layers"):
        return unwrap_model(model._layers)
    else:
        return model


def create_qk_layernorm_hook(param, accumulation_steps):
    """create_qk_layernorm_hook"""
    hcg = fleet.get_hybrid_communicate_group()
    pg = hcg.get_model_parallel_group().process_group
    step = [0]

    @paddle.autograd.no_grad()
    def __impl__():
        step[0] += 1
        if (step[0] % accumulation_steps) == 0:
            if hasattr(param, "main_grad"):
                pg.allreduce(param.main_grad).wait()
            else:
                pg.allreduce(param.grad).wait()

    return __impl__


class LatentDiffusionAutoTrainer(AutoTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_meshes_for_loader(self):
        def _get_mesh(pp_idx=0):
            return fleet.auto.get_mesh().get_mesh_with_dim("pp")[pp_idx]

        return _get_mesh(0)  # label_id is not label

    def _wrap_for_dist_loader(self, train_dataloader):
        dist_loader = super()._wrap_for_dist_loader(train_dataloader)
        dist_loader._input_keys = ["latents", "label_id"]
        return dist_loader

    def _save_checkpoint(self, model, metrics=None):

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self.args.output_dir
        output_dir = f"{run_dir}/{checkpoint_folder}"

        if self.args.should_save or self.args.should_save_model_state:
            os.makedirs(output_dir, exist_ok=True)

        if self.args.should_save:
            logger.info(f"Saving checkpoinit files into {output_dir}")

            if self.args.should_save_model_state:
                if self.args.to_static:
                    opt_state_dict = {
                        key: value
                        for key, value in model.state_dict("opt").items()
                        if not any(keyword in key for keyword in FREE_SVAE_LOAD_KEY_PATTERNS)
                    }
                    model_state_dict = model.state_dict("param")
                    if self.args.should_save_model_with_tensor_fusion:
                        model_state_dict = self._convert_state_dict_for_saving_tensor_fusion_ckpt(model_state_dict)
                        opt_state_dict = self._convert_state_dict_for_saving_tensor_fusion_ckpt(opt_state_dict)

                    state_dict = {
                        MODEL_NAME: model_state_dict,
                        OPTIMIZER_NAME: opt_state_dict,
                    }
                else:
                    optim_state_dict = self.optimizer.state_dict()
                    optim_state_dict.pop("LR_Scheduler", None)
                    opt_state_keys = ["_moment1_0", "_moment2_0", "_beta1_pow_acc_0", "_beta2_pow_acc_0"]
                    for p_name, p in model.state_dict().items():
                        # liaojincheng: Fix process_mesh missing error for none-auto sub model likes model_ema
                        if (
                            hasattr(p.process_mesh, "process_ids")
                            and paddle.distributed.get_rank() not in p.process_mesh.process_ids
                        ):
                            var_name = p.name
                            for key in opt_state_keys:
                                if (
                                    var_name + key in optim_state_dict
                                    and not optim_state_dict[var_name + key].is_dist()
                                ):
                                    optim_state_dict.pop(var_name + key)

                    state_dict = {
                        MODEL_NAME: model.state_dict(),
                        OPTIMIZER_NAME: optim_state_dict,
                    }

                self._save(output_dir=os.path.join(output_dir, DIST_CKPT_PATH), state_dict=state_dict)
                # FIXME: maybe only save one copy
                paddle.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

                if self.do_grad_scaling:
                    paddle.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

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

        if self.args.world_size > 1:
            rng_states_list = []
            paddle.distributed.all_gather_object(rng_states_list, rng_states)
            if self.args.should_save:
                os.makedirs(output_dir, exist_ok=True)
                paddle.save(rng_states_list, os.path.join(output_dir, f"rng_state_{self.args.world_size}.pth"))
        else:
            os.makedirs(output_dir, exist_ok=True)
            paddle.save(rng_states, os.path.join(output_dir, "rng_state.pth"))

        if strtobool(os.getenv("FLAG_LLM_PDC", "False")):
            # save checkpoint_done file to ensure checkpoint is complete
            if self.args.should_save_model_state and self.args.should_save:
                # For ckpt integrity
                paddle.save(self.state.global_step, os.path.join(output_dir, ".checkpoint_done"))

    def _save(
        self,
        output_dir=None,
        state_dict=None,
        merge_tensor_parallel=False,
    ):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if self.args.should_save:
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            # Good practice: save your training arguments together with the trained model
            paddle.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        if self.args.should_save_model_state:
            self._save_ckpt_func(self.model.state_dict(), output_dir)
            logger.info(f"Model weights and optimizer states saved in {output_dir}")


def clip_grad_norm_(
    parameters, max_norm, norm_type=2.0, error_if_nonfinite: bool = False, return_cliped_norm: bool = False
):
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
        return_cliped_norm (bool): if True, total norm clipped will be return and it is
            only used for tensorboard. Default: False.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    paddle_dtype = paddle.get_default_dtype()
    if len(grads) == 0:
        return paddle.to_tensor([0.0])
    if norm_type == float("inf"):
        norms = [g.detach().abs().max() for g in grads]
        total_norm = norms[0] if len(norms) == 1 else paddle.max(paddle.stack(norms))
    else:
        total_norm = paddle.norm(
            paddle.stack(
                [
                    paddle.norm(g.detach(), norm_type)
                    if g.dtype == paddle_dtype
                    else paddle.norm(g.detach().cast(paddle_dtype), norm_type)
                    for g in grads
                ]
            ),
            norm_type,
        )
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
    clip_coef_clamped_low_precison = None
    for g in grads:
        if g.dtype == paddle.float32:
            g.detach().multiply_(clip_coef_clamped)
        else:
            clip_coef_clamped_low_precison = (
                clip_coef_clamped.cast(g.dtype)
                if clip_coef_clamped_low_precison is None
                else clip_coef_clamped_low_precison
            )
            g.detach().multiply_(clip_coef_clamped_low_precison)

    if return_cliped_norm:
        total_norm_clip = paddle.norm(
            paddle.stack(
                [
                    paddle.norm(g.detach(), norm_type)
                    if g.dtype == paddle_dtype
                    else paddle.norm(g.detach().cast(paddle_dtype), norm_type)
                    for g in grads
                ]
            ),
            norm_type,
        )
        return total_norm_clip
    return total_norm

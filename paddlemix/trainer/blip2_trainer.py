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
import contextlib
import inspect
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import paddle
import paddle.amp.auto_cast as autocast
import paddle.nn as nn
from paddle.distributed import fleet
from paddle.io import DataLoader, Dataset
from paddlenlp.trainer import PrinterCallback, ProgressCallback, Trainer
from paddlenlp.trainer.integrations import TrainerCallback
from paddlenlp.trainer.trainer_callback import DefaultFlowCallback
from paddlenlp.trainer.trainer_utils import (  # set_hybrid_parallel_seed,
    EvalLoopOutput,
    IterableDatasetShard,
    ShardingOption,
    has_length,
    speed_metrics,
)
from paddlenlp.transformers.model_utils import unwrap_model
from paddlenlp.utils import device_guard, profiler
from paddlenlp.utils.import_utils import is_datasets_available
from paddlenlp.utils.log import logger

import paddlemix
from paddlemix.models.blip2.utils import VQA, VQAEval, coco_caption_eval, save_result
from paddlemix.optimization import FilterParamsName

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"

OPTIMIZER_NAME = "optimizer.pdopt"
SCHEDULER_NAME = "scheduler.pdparams"
SCALER_NAME = "scaler.pdparams"

if is_datasets_available():
    import datasets

try:
    from paddle.distributed.fleet.utils import mix_precision_utils
except:
    mix_precision_utils = None


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


def paddlenlp_load(path, return_numpy=False):
    if return_numpy:
        with device_guard():
            return paddle.load(path)
    else:
        return paddle.load(path, return_numpy=return_numpy)


def is_dp_group_support_in_group_sharded_parallel():
    return "dp_group" in set(inspect.signature(paddle.distributed.sharding.group_sharded_parallel).parameters.keys())


__all__ = ["BLIP2Trainer"]


class BLIP2Trainer(Trainer):
    """
    BLIP2Trainer is a feature-complete training and eval loop for BLIP2.

    Args:
    processor: (`Blip2Processor`) low level data processors to convert input text to PaddleNLP Datasets.
    eval_processor: (`Blip2Processor`) Unlike processor, eval_processor is used for model evaluation.
    eval_collator: (`BlipCollator`) dynamically pad the inputs to the longest sequence in the batch.

    """

    def __init__(self, processor=None, eval_processor=None, eval_collator=None, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.eval_processor = eval_processor
        self.eval_collator = eval_collator
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

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.lr_scheduler = self.create_scheduler(num_training_steps // self.args.num_train_epochs)
        param_filter = FilterParamsName()
        p_wd, p_non_wd = param_filter(self.model)
        self.optimizer = paddle.optimizer.AdamW(
            parameters=p_wd + p_non_wd,
            learning_rate=self.lr_scheduler,
            weight_decay=float(self.args.weight_decay),
            beta1=self.args.adam_beta1,
            beta2=self.args.adam_beta2,
            apply_decay_param_fun=param_filter._apply_decay_param_fun,
        )

    def create_scheduler(self, num_training_steps):
        lr_sched_func = getattr(paddlemix.optimization, self.args.lr_scheduler_name)
        lr_sched = lr_sched_func(
            learning_rate=self.args.learning_rate,
            total_steps=self.args.num_train_epochs * num_training_steps,
            warmup_start_lr=self.args.warmup_start_lr,
            eta_min=self.args.eta_min,
            warmup=self.args.warmup_steps,
            step_each_epoch=num_training_steps,
        )
        return lr_sched

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~paddle.io.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`paddle.io.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not accepted by
                the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")

        if self._is_iterable_dataset(eval_dataset):
            if self.args.dataset_world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.dataset_world_size,
                    process_index=self.args.dataset_rank,
                )

            return DataLoader(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=self.eval_collator,
                num_workers=self.args.dataloader_num_workers,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=self.eval_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def _wrap_model(self, model, training=True):

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
            if hasattr(model, "language_model"):
                decorated = paddle.amp.decorate(
                    models=[model.visual_encoder, model.language_model], optimizers=self.optimizer, level="O2"
                )
                model.visual_encoder, model.language_model = decorated[0]
            else:
                decorated = paddle.amp.decorate(models=[model.visual_encoder], optimizers=self.optimizer, level="O2")
                model.visual_encoder = decorated[0][0]
            self.optimizer.set_state_dict(decorated[1].state_dict())

        # Multi-gpu training
        if self.args.world_size > 1 and not self.args.use_hybrid_parallel:
            model = paddle.DataParallel(model)
            assert self.args.tensor_parallel_degree < 2, "tensor_parallel_degree = {}, please init optimizer.".format(
                self.args.tensor_parallel_degree
            )

        in_pipeline_parallel_mode = self.args.pipeline_parallel_degree > 1
        in_sharding_parallel_mode = self.sharding is not None
        in_tensor_parallel_model = self.args.tensor_parallel_degree > 1
        if in_pipeline_parallel_mode:
            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)  # return value has no use
            # hack for pipeline model mini batch to batch
            # need batter solution @ZHUI
            # make batch_fn compatible for fleet.distributed_model decorate.
            prepare_pipeline_inputs_func = (
                model._prepare_pipeline_inputs_func if hasattr(model, "_prepare_pipeline_inputs_func") else None
            )
            model = fleet.distributed_model(model)
            if prepare_pipeline_inputs_func is not None:
                model._prepare_pipeline_inputs_func = prepare_pipeline_inputs_func
            else:

                def _prepare_pipeline_inputs_func(inputs):
                    first_stage_keys = ["input_ids", "attention_mask", "position_ids"]
                    last_stage_keys = ["labels"]

                    def get_expected_keys(inputs, keys):
                        ret = tuple([inputs.pop(k) for k in keys if k in inputs])
                        if len(ret) == 1:
                            ret = ret[0]
                        return ret

                    if type(inputs) is dict:
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

            assert self.optimizer is not None, "Pipeline mode need decorate optimizer, please init optimizer."
            if self.args.amp_master_grad:
                self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)
            self.optimizer = fleet.distributed_optimizer(self.optimizer)

        # No pipeline mode, sharding only
        if not in_pipeline_parallel_mode and in_sharding_parallel_mode:
            # Sharded DDP!
            if self.args.tensor_parallel_degree > 1:
                hcg = fleet.get_hybrid_communicate_group()
                assert (
                    ShardingOption.SHARD_GRAD_OP in self.args.sharding or ShardingOption.SHARD_OP in self.args.sharding
                ), "Only support tensor parallel + sharding stage1/stage2 hybrid parallel now."
                model = paddle.distributed.fleet.meta_parallel.TensorParallel(model, hcg, strategy=None)

            if ShardingOption.SHARD_OP in self.args.sharding:
                model = fleet.distributed_model(model)
                self.optimizer = fleet.distributed_optimizer(self.optimizer)
            else:
                # sync params (broadcast) buffers in dp group

                if not is_dp_group_support_in_group_sharded_parallel() and self.args.data_parallel_degree > 1:
                    try:
                        from paddle.fluid.dygraph.parallel import sync_params_buffers
                    except ImportError:
                        # fix for new api in paddlepaddle v2.5
                        from paddle.distributed.parallel import sync_params_buffers

                    hcg = fleet.get_hybrid_communicate_group()
                    dp_group = hcg.get_data_parallel_group()
                    sync_params_buffers(model, comm_group=dp_group, src_rank=dp_group.ranks[0])

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
                if is_dp_group_support_in_group_sharded_parallel():
                    extra_kwargs["dp_group"] = self.dp_group
                    extra_kwargs["exclude_layer"] = ["GroupNorm"]

                model, optimizer, _ = group_sharded_parallel(
                    model,
                    self.optimizer,
                    level=level,
                    scaler=None,
                    group=self.sharding_group,
                    offload=cpu_offload,
                    **extra_kwargs,
                )
                self.optimizer = optimizer

        # pure tensor parallel mode, no pipeline_parallel, no sharding.
        if not in_pipeline_parallel_mode and not in_sharding_parallel_mode and in_tensor_parallel_model:
            if self.args.amp_master_grad:
                mix_precision_utils.MixPrecisionLayer(model, dtype=self.amp_dtype)  # return value has no use

            model = fleet.distributed_model(model)
            assert self.optimizer is not None, "Tensor parallel mode need decorate optimizer, please init optimizer."
            if self.args.amp_master_grad:
                self.optimizer = mix_precision_utils.MixPrecisionOptimizer(self.optimizer)
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
        return model

    def autocast_smart_context_manager(self):
        """
        A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation.
        """
        if self.enable_autocast_context_manager:
            ctx_manager = autocast(
                True,
            )
        else:
            ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()

        return ctx_manager

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        task_name="coco_caption",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        self.task_name = task_name
        if isinstance(eval_dataset, dict):
            eval_dataset = eval_dataset["test"]
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # total_batch_size = self.args.eval_batch_size * self.args.dataset_world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_eval_iters: Optional[int] = -1,
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self.model

        if isinstance(dataloader, paddle.io.DataLoader):
            batch_size = dataloader.batch_sampler.batch_size
        elif isinstance(dataloader, paddle.fluid.dataloader.dataloader_iter._DataLoaderIterBase):
            # support for inner dataloader
            batch_size = dataloader._batch_sampler.batch_size
            # alias for inner dataloader
            dataloader.dataset = dataloader._dataset
        else:
            raise ValueError("Only support for paddle.io.DataLoader")

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            if max_eval_iters > 0:
                logger.info(f"  Total prediction steps = {max_eval_iters}")
            else:
                logger.info(f"  Total prediction steps = {len(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
            if max_eval_iters > 0:
                logger.info(f"  Total prediction steps = {max_eval_iters}")

        logger.info(f"  Pre device batch size = {batch_size}")
        logger.info(f"  Total Batch size = {batch_size * self.args.dataset_world_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader

        if args.past_index >= 0:
            self._past = None
        results = []
        for step, inputs in enumerate(dataloader):
            # Prediction step
            eval_output = self.prediction_step(model, inputs)
            results.extend(eval_output)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
            if max_eval_iters > 0 and step >= max_eval_iters - 1:
                break
        if results is not None:
            metrics = self.after_evaluation(val_result=results)
        else:
            metrics = None

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=None)

    def prediction_step(
        self,
        model: nn.Layer,
        inputs: Dict[str, Union[paddle.Tensor, Any]],
    ) -> Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Layer`):
                The model to evaluate.
            inputs (`Dict[str, Union[paddle.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Return:
            Tuple[Optional[paddle.Tensor], Optional[paddle.Tensor], Optional[paddle.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        inputs = self._prepare_inputs(inputs)
        results = []
        if "caption" in self.task_name:
            with paddle.no_grad():
                # with paddle.amp.auto_cast(level='O2'):
                model_inputs = self.eval_processor(
                    text=[""] * inputs["pixel_values"].shape[0],
                    return_tensors="pd",
                    return_attention_mask=True,
                    mode="test",
                )
                model_inputs.update(inputs)
                generated_ids, scores = model.generate(**model_inputs)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                generated_text = [text.strip() for text in generated_text]
                for caption, img_id in zip(generated_text, inputs["image_id"]):
                    results.append({"caption": caption, "image_id": int(img_id)})
        elif "vqa" in self.task_name:
            with paddle.no_grad():
                # with paddle.amp.auto_cast(level='O2'):
                model_inputs = inputs
                generated_ids, scores = model.predict_answers(**model_inputs)
                answers = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                answers = [text.strip() for text in answers]
                question_id = inputs["question_id"]
                for answer, ques_id in zip(answers, question_id):
                    ques_id = int(ques_id)
                    results.append({"question_id": ques_id, "answer": answer})
        else:
            raise NotImplementedError
        return results

    def after_evaluation(self, val_result):
        if "caption" in self.task_name:
            eval_result_file = save_result(
                result=val_result,
                result_dir=self.args.output_dir + self.task_name + "/result",
                filename="{}_epoch{}".format("eval", "eval"),
                remove_duplicate="image_id",
                world_size=self.args.world_size,
            )

            metrics = self._report_metrics_caption(eval_result_file=eval_result_file)
        elif "vqa" in self.task_name:
            eval_result_file = save_result(
                val_result,
                result_dir=self.args.output_dir + self.task_name + "/result",
                filename="{}_epoch{}".format("eval", "eval"),
                remove_duplicate="question_id",
            )

            metrics = self._report_metrics_vqa(eval_result_file=eval_result_file)
        else:
            raise NotImplementedError
        return metrics

    def _report_metrics_caption(self, eval_result_file, split_name="test"):

        # TODO better way to define this
        coco_gt_root = os.path.join("/root/.paddlemix/datasets/", "")
        coco_val = coco_caption_eval(coco_gt_root, eval_result_file, split_name)

        agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        with open(os.path.join(self.args.output_dir, "evaluate.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res

    def _report_metrics_vqa(self, eval_result_file):

        metrics = {}
        self.anno_files = "/root/.paddlemix/datasets/coco/annotations/v2_mscoco_val2014_annotations.json"
        self.ques_files = "/root/.paddlemix/datasets/coco/annotations/v2_OpenEnded_mscoco_val2014_questions.json"

        vqa = VQA(self.anno_files, self.ques_files)
        vqa_result = vqa.loadRes(resFile=eval_result_file, quesFile=self.ques_files)
        vqa_scorer = VQAEval(vqa, vqa_result, n=2)
        logger.info("Start VQA evaluation.")
        vqa_scorer.evaluate()

        # print accuracies
        overall_acc = vqa_scorer.accuracy["overall"]
        metrics["agg_metrics"] = overall_acc

        logger.info("Overall Accuracy is: %.02f\n" % overall_acc)
        logger.info("Per Answer Type Accuracy is the following:")

        for ans_type in vqa_scorer.accuracy["perAnswerType"]:
            logger.info("%s : %.02f" % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type]))
            metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

        with open(os.path.join(self.args.output_dir, "evaluate.txt"), "a") as f:
            f.write(json.dumps(metrics) + "\n")

        return metrics

    def _save(self, output_dir: Optional[str] = None, state_dict=None, merge_tensor_parallel=False):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        merge_tensor_parallel = merge_tensor_parallel and self.args.use_hybrid_parallel
        if self.eval_processor is not None:
            self.eval_processor.image_processor.save_pretrained(os.path.join(output_dir, "processor", "eval"))
            self.eval_processor.text_processor.save_pretrained(os.path.join(output_dir, "processor", "eval"))
        self.processor.image_processor.save_pretrained(os.path.join(output_dir, "processor", "train"))
        self.processor.text_processor.save_pretrained(os.path.join(output_dir, "processor", "train"))
        self.model.save_pretrained(
            output_dir,
            merge_tensor_parallel=merge_tensor_parallel,
            variant=self.args.weight_name_suffix,
            is_main_process=self.args.should_save,
            processor=self.processor,
            eval_processor=self.eval_processor,
        )

        if self.args.should_save:
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            paddle.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

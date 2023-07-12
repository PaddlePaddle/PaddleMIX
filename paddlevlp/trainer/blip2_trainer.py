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

from paddlenlp.trainer.trainer import Trainer
from paddlevlp.examples.blip2.utils import coco_caption_eval
# Copyright 2020-present the HuggingFace Inc. team.
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

# This file is modified from
#  https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py

import contextlib
import inspect
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.amp.auto_cast as autocast
import paddle.nn as nn
from paddle.distributed import fleet
from paddle.io import DataLoader, Dataset, DistributedBatchSampler

from paddlenlp.transformers.model_utils import unwrap_model
from paddlenlp.utils import device_guard
from paddlenlp.utils.batch_sampler import DistributedBatchSampler as NlpDistributedBatchSampler
from paddlenlp.utils.import_utils import is_datasets_available
from paddlenlp.utils.log import logger
from paddlenlp.trainer.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from paddlenlp.trainer.trainer_utils import (  # set_hyrbid_parallel_seed,
    EvalLoopOutput,
    EvalPrediction,
    IterableDatasetShard,
    ShardingOption,
    find_batch_size,
    has_length,
    speed_metrics,
)
import json

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
    eval_processor: (`Blip2Processor`) Unlike rocessor, eval_processor is used for model evaluation.
    eval_collator: (`BlipCollator`) dynamically pad the inputs to the longest sequence in the batch.

    """

    from paddlenlp.trainer.trainer_utils import log_metrics, metrics_format, save_metrics, save_state


    def __init__(
        self,
        processor=None,
        eval_processor=None,
        eval_collator=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.processor=processor
        self.eval_processor=eval_processor
        self.eval_collator=eval_collator

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
            decorated = paddle.amp.decorate(
                models=[model.visual_encoder,model.language_model], optimizers=self.optimizer, level=self.args.fp16_opt_level
            )
            model.visual_encoder,model.language_model= decorated[0]
            self.optimizer = decorated[1]

        # Multi-gpu training
        if self.args.world_size > 1 and not self.args.use_hybrid_parallel:
            model = paddle.DataParallel(model)
            # Distributed training (should be after fp16 initialization)
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

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.dataset_world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
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
            metrics=self.after_evaluation(val_result=results)
        else:
            metrics=None


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
        with paddle.no_grad():
            with paddle.amp.auto_cast(level='O2'):
                prompt = self.args.prompt
                model_inputs = self.eval_processor(
                    images=[inputs['pixel_values'][i] for i in  range(inputs['pixel_values'].shape[0])],
                    text=[prompt]*inputs['pixel_values'].shape[0],
                    return_tensors="pd",
                    return_attention_mask=True,
                    mode="test",
                )
                generated_ids, scores = model.generate(**model_inputs)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                for caption, img_id in zip(generated_text, inputs['image_id']):
                    results.append({"caption": caption, "image_id": int(img_id)})
        return results



    def after_evaluation(self, val_result):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=self.args.output_dir+"/resulut",
            filename="{}_epoch{}".format('eval', 'eval'),
            remove_duplicate="image_id",
        )

        metrics = self._report_metrics(
            eval_result_file=eval_result_file
        )

        return metrics

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import logging
        rank_id_curr_node = int(os.environ.get("PADDLE_RANK_IN_NODE", 0))
        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, rank_id_curr_node)
        )
        if not os.path.exists(result_dir): os.mkdir(result_dir)
        json.dump(result, open(result_file, "w"))

        final_result_file = os.path.join(result_dir, "%s.json" % filename)
        if self.args.world_size > 1:
           paddle.distributed.barrier()
        if rank_id_curr_node==0:
            logging.warning("rank %d starts merging results." % rank_id_curr_node)
            result = []
            # for rank in range(get_world_size()):
            for rank in range(int(os.environ.get("PADDLE_TRAINERS_NUM", 1))):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)
        else:
            while not os.path.exists(final_result_file):
                time.sleep(0.5)
                logging.warning("rank %d waits rank0 to merge results." % rank_id_curr_node)

        # combine results from all processes
        return final_result_file

    def _report_metrics(self, eval_result_file, split_name="test"):

        # TODO better way to define this
        coco_gt_root = os.path.join('/export/home/.cache/lavis', "coco_gt")
        coco_val = coco_caption_eval(coco_gt_root, eval_result_file, split_name)

        agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        with open(os.path.join(self.args.output_dir, "evaluate.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res

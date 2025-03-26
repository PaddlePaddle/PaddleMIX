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

import paddle
from paddle.io import DataLoader
from paddlenlp.trainer.trainer import Trainer
from paddlenlp.transformers.model_utils import unwrap_model
from paddlenlp.utils.log import logger
from tensorboardX import SummaryWriter

from paddlemix.models.eva02.optim_factory import get_grad_norm_and_clip

PADDLE_WEIGHTS_NAME = "model_state.pdparams"
TRAINING_ARGS_NAME = "training_args.bin"


class EVA02PretrainTrainer(Trainer):
    def __init__(self, **kwargs):
        """
        Implementation of an `Trainer` suitable for EVA-02 Pretrain
        1、selfdefine optimizer for sharding which can't create by passing by args
        2、support for accum_freq

        Args:
            kwargs (dict): any arguments to pass to `Trainer`

        Returns:
            None
        """
        super().__init__(**kwargs)
        self.beit_like = True
        if self.args.accum_freq > 1:
            self.accum_samples = []
            self.accum_image = []
            self.accum_bool_masked_pos = []
            self.accu_step = 0

        self.iter = 0  # real iter

        self.rank = paddle.distributed.get_rank()
        if self.rank == 0 and self.args.tensorboard:
            self.writer = SummaryWriter(self.args.output_dir)
            self.logstep = 0

    def training_step(self, model, inputs) -> paddle.Tensor:
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
        model = unwrap_model(model)
        model.student.train()
        model.teacher.eval()

        it = self.iter // self.args.accum_freq
        if self.lr_schedule_values is not None or self.wd_schedule_values is not None:
            for i, param_group in enumerate(self.optimizer._param_groups):
                if self.lr_schedule_values is not None:
                    param_group["learning_rate"] = self.lr_schedule_values[it] * param_group["lr_scale"]
                    for param in param_group["params"]:
                        param.optimize_attr["learning_rate"] = self.lr_schedule_values[it] * param_group["lr_scale"]
                if self.wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = self.wd_schedule_values[it]

        if self.args.pipeline_parallel_degree > 1:
            return self.training_pipeline_step(model, inputs)
        elif self.args.accum_freq > 1:
            return self.training_step_accumfreq(model, inputs)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        loss_value = loss.item()

        if self.do_grad_scaling:
            self.scaler.unscale_(self.optimizer)

        grad_norms = get_grad_norm_and_clip(model, self.args.max_grad_norm)

        min_lr = 10.0
        max_lr = 0.0
        for group in self.optimizer._param_groups:
            min_lr = min(min_lr, group["learning_rate"])
            max_lr = max(max_lr, group["learning_rate"])
        self.curr_lr = max_lr

        if self.rank == 0 and self.args.tensorboard:
            self.writer.add_scalar("loss/loss", loss_value, self.logstep)
            self.writer.add_scalar("opt/grad_norm", grad_norms.item(), self.logstep)
            self.writer.add_scalar("opt/lr", max_lr, self.logstep)
            self.writer.add_scalar("opt/min_lr", min_lr, self.logstep)
            self.writer.add_scalar("opt/lr0", self.optimizer._param_groups[0]["learning_rate"], self.logstep)
            self.logstep += 1

        self.iter += 1
        return loss.detach()

    def _get_learning_rate(self):
        return self.curr_lr

    def training_step_accumfreq(self, model, inputs) -> paddle.Tensor:
        self.accum_samples.append(inputs["samples"])
        self.accum_image.append(inputs["image"])
        self.accum_bool_masked_pos.append(inputs["bool_masked_pos"])
        self.accu_step += 1

        # If (cnt + 1) % accum_freq is not zero, move on to the next batch.
        if (self.accu_step % self.args.accum_freq) > 0:
            # FIXME this makes data time logging unreliable when accumulating
            return paddle.full([1], 0, dtype="float32")

        # Now, ready to take gradients for the last accum_freq batches.
        # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
        # Call backwards each time, but only step optimizer at the end.
        self.optimizer.clear_grad()
        for j in range(self.args.accum_freq):
            with self.autocast_smart_context_manager():
                inputs_j = {
                    "samples": self.accum_samples[j],
                    "image": self.accum_image[j],
                    "bool_masked_pos": self.accum_bool_masked_pos[j],
                }
                loss = self.compute_loss(model, inputs_j)

            if self.do_grad_scaling:
                self.scaler.scale(loss / self.args.accum_freq).backward()
            else:
                (loss / self.args.accum_freq).backward()

        # clear for next accu batches
        self.accum_samples.clear()
        self.accum_image.clear()
        self.accum_bool_masked_pos.clear()
        self.accu_step = 0

        loss_value = loss.item()

        if self.do_grad_scaling:
            self.scaler.unscale_(self.optimizer)

        grad_norms = get_grad_norm_and_clip(model, self.args.max_grad_norm)

        min_lr = 10.0
        max_lr = 0.0
        for group in self.optimizer._param_groups:
            min_lr = min(min_lr, group["learning_rate"])
            max_lr = max(max_lr, group["learning_rate"])
        self.curr_lr = max_lr

        if self.rank == 0 and self.args.tensorboard:
            self.writer.add_scalar("loss/loss", loss_value, self.logstep)
            self.writer.add_scalar("opt/grad_norm", grad_norms.item(), self.logstep)
            self.writer.add_scalar("opt/lr", max_lr, self.logstep)
            self.writer.add_scalar("opt/min_lr", min_lr, self.logstep)
            self.writer.add_scalar("opt/lr0", self.optimizer._param_groups[0]["learning_rate"], self.logstep)
            self.logstep += 1
            # Note: logstep is not same as iter when accum_freq > 1

        self.iter += self.args.accum_freq
        return loss.detach()

    def get_train_dataloader(self):
        sampler_train = paddle.io.DistributedBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            num_replicas=self.args.data_world_size,
            rank=self.args.data_world_rank,
            shuffle=True,
            drop_last=True,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler_train,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.data_collator,
            use_shared_memory=True,
        )

    def _save(self, output_dir=None, state_dict=None, merge_tensor_parallel=False):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        merge_tensor_parallel = merge_tensor_parallel and self.args.use_hybrid_parallel

        if self.args.only_save_updated_model:  # default True
            unwrapped_model = unwrap_model(self.model)
            logger.info(f"Saving eva02_vit checkpoint to {output_dir}/eva02_vit")
            unwrapped_model.student.save_pretrained(
                os.path.join(output_dir, "eva02_vit"),
                merge_tensor_parallel=merge_tensor_parallel,
            )
        else:
            unwrapped_model = unwrap_model(self.model)
            logger.info(f"Saving evaclip + eva02_vit checkpoint to {output_dir}")
            unwrapped_model.save_pretrained(
                output_dir,
                merge_tensor_parallel=merge_tensor_parallel,
                variant=self.args.weight_name_suffix,
                is_main_process=self.args.should_save,
            )
            if self.args.should_save:
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(output_dir)
                paddle.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

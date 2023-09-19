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

import paddle
from paddle.io import DataLoader
from tensorboardX import SummaryWriter

from paddlemix.models.clip.utils import clip_grad_norm
from paddlenlp.trainer.trainer import Trainer


class CLIPTrainer(Trainer):
    def __init__(self, **kwargs):
        """
        Implementation of an `Trainer` suitable for EVA-CLIP
        1、selfdefine optimizer for sharding which can't create by passing by args

        Args:
            kwargs (dict): any arugments to pass to `Trainer`

        Returns:
            None
        """
        super().__init__(**kwargs)
        self.rank = paddle.distributed.get_rank()
        if self.rank == 0 and self.args.tensorboard:
            self.writer = SummaryWriter("output/tensorboard")
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

        if self.args.pipeline_parallel_degree > 1:
            return self.training_pipeline_step(model, inputs)

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=1)
        loss_itc, image_features, text_features, logit_scale = outputs

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.args.max_grad_norm > 0.0:
            grad_norms = clip_grad_norm(model, self.args.max_grad_norm)
        if self.rank == 0 and self.args.tensorboard:
            self.writer.add_scalar("train/loss", loss.item(), self.logstep)
            self.writer.add_scalar("train/lr", self.optimizer.get_lr(), self.logstep)
            self.writer.add_scalar("train/grad_norm", grad_norms.item(), self.logstep)
            self.writer.add_scalar("train/logit_scale", logit_scale.item(), self.logstep)
            self.logstep += 1

        return loss.detach()

    def get_train_dataloader(self):
        """
        Returns the training [`~paddle.io.DataLoader`].

        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            prefetch_factor=1,
            shuffle=False,
        )

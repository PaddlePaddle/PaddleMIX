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
from paddlenlp.trainer.trainer import Trainer
from tensorboardX import SummaryWriter

from paddlemix.datasets.collator import (
    CLIPCollator,
    EVA02Collator,
    InternLMXComposer2Collator,
    InternVL2Collator,
    LLaVACollator,
    MiniGPT4Collator,
    QwenVLCollator,
    VisualglmCollator,
)
from paddlemix.metrics.clip_zero_shot import ClipZeroShot
from paddlemix.models.blip2.utils import BlipCollator
from paddlemix.models.clip.utils import clip_grad_norm
from paddlemix.optimization import create_optimizer_simple
from paddlemix.trainer.blip2_trainer import BLIP2Trainer
from paddlemix.trainer.eva02_finetune_trainer import EVA02FinetuneTrainer
from paddlemix.trainer.llava_trainer import LLaVATrainer
from paddlemix.trainer.minigpt4_trainer import MiniGPT4Trainer


class CLIPTrainer(Trainer):
    def __init__(self, **kwargs):
        """
        Implementation of an `Trainer` suitable for EVA-CLIP
        1、selfdefine optimizer for sharding which can't create by passing by args

        Args:
            kwargs (dict): any arguments to pass to `Trainer`

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
            if self.args.tensor_fusion:
                parameters = self.optimizer.all_parameters
            else:
                parameters = model.parameters()
            grad_norms = clip_grad_norm(parameters, self.args.max_grad_norm, need_grad_norm=self.args.tensorboard)
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

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            self.args.learning_rate,
            num_training_steps - self.args.warmup_steps,
            last_epoch=self.args.last_epoch,
        )
        if self.args.warmup_steps > 0:
            self.lr_scheduler = paddle.optimizer.lr.LinearWarmup(
                self.lr_scheduler,
                self.args.warmup_steps,
                0,
                1.0,
                last_epoch=self.args.last_epoch,
            )
        self.optimizer = create_optimizer_simple(self.args, self.model, self.lr_scheduler)


def get_trainer(
    pretrained_model_name_or_path,
    model,
    args,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    train_processor=None,
    eval_processor=None,
    mixtokens=False,
):
    """
    Returns the trainer according to model.base_model_prefix
    Returns:
        Trainer: a trainer instance
    """
    pretrained_model_name_or_path = pretrained_model_name_or_path.lower().replace("-", "_").replace("vicuna", "llava")
    if "clip" in pretrained_model_name_or_path or "coca" in pretrained_model_name_or_path:
        zeroshot = ClipZeroShot(model, args)
        return CLIPTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=CLIPCollator(train_processor),
            compute_metrics=zeroshot.zero_shot_eval,
        )
    elif "blip2" in pretrained_model_name_or_path:
        blip_collator = BlipCollator(train_processor)
        blip_eval_collator = BlipCollator(eval_processor, mode="test")
        return BLIP2Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=blip_collator,
            eval_collator=blip_eval_collator,
            processor=train_processor,
            eval_processor=eval_processor,
            tokenizer=tokenizer,
        )
    elif "eva02" in pretrained_model_name_or_path:
        collator = EVA02Collator(train_processor, mode="train")
        return EVA02FinetuneTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            data_collator=collator,
        )
    elif "minigpt4" in pretrained_model_name_or_path:
        return MiniGPT4Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            data_collator=MiniGPT4Collator(train_processor),
            processor=train_processor,
            tokenizer=tokenizer,
        )
    elif "llava" in pretrained_model_name_or_path:
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        return LLaVATrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=LLaVACollator(train_processor, mode="train"),
        )
    else:
        if "qwen_vl" in pretrained_model_name_or_path:
            collator = QwenVLCollator(train_processor, mode="train")
        elif "visualglm" in pretrained_model_name_or_path:
            collator = VisualglmCollator(train_processor, mode="train")
        elif "internlm" in pretrained_model_name_or_path:
            collator = InternLMXComposer2Collator(train_processor, mode="train")
        elif "internvl" in pretrained_model_name_or_path:
            collator = InternVL2Collator(train_processor, mode="train")
        else:
            collator = None

        return Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=collator,
        )

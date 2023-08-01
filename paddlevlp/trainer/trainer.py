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
from paddlenlp.trainer.trainer import Trainer
from paddle.io import DataLoader
from paddlevlp.models.evaclip.eva_clip.utils import clip_grad_norm
from torch.utils.tensorboard import SummaryWriter

class CLIPTrainer(Trainer):
    def __init__(self, **kwargs):
        """
        Implementation of an `Trainer` suitable for EVA-CLIP
        1、selfdefine optimizer for sharding which can't create by passing by args
        2、support for accum_freq
        
        Args:
            kwargs (dict): any arugments to pass to `Trainer`
        
        Returns:
            None
        """
        super().__init__(**kwargs)
        if self.args.accum_freq > 1:
            self.accum_features = {}
            self.accum_images = []
            self.accum_texts = []
            self.step = 0
        
        self.rank = paddle.distributed.get_rank()
        if self.rank==0:
            self.writer = SummaryWriter("tensorboard_record")
            self.step = 0

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
        elif self.args.accum_freq > 1:
            return self.training_step_accumfreq(model, inputs)

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.args.max_grad_norm > 0.0:
            _ = clip_grad_norm(model, self.args.max_grad_norm)

        if self.rank == 0:
            self.step += 1
            self.writer.add_scalar("train/loss", loss.item(), self.step)

        return loss.detach()

    def training_step_accumfreq(self, model, inputs) -> paddle.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        with paddle.no_grad():
            preds = model(**inputs, skiploss=True)
            image_features, text_features, logit_scale = preds[:3]
        model_out = {
            'image_features': image_features,
            'text_features': text_features
        }
        for key, val in model_out.items():
            if key in self.accum_features:
                self.accum_features[key].append(val)
            else:
                self.accum_features[key] = [val]
        self.accum_images.append(inputs['image'])
        self.accum_texts.append(inputs['input_ids'])
        self.step += 1

        # If (cnt + 1) % accum_freq is not zero, move on to the next batch.
        if (self.step % self.args.accum_freq) > 0:
            # FIXME this makes data time logging unreliable when accumulating
            return paddle.full([1], 0, dtype="float32")

        if hasattr(model, '_layers'):
            modelloss = model._layers.loss
        else:
            modelloss = model.loss
        # Now, ready to take gradients for the last accum_freq batches.
        # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
        # Call backwards each time, but only step optimizer at the end.
        # optimizer.clear_grad()
        for j in range(self.args.accum_freq):
            preds = model(
                self.accum_images[j], self.accum_texts[j], skiploss=True)
            image_features, text_features, logit_scale = preds[:3]
            model_out = {
                'image_features': image_features,
                'text_features': text_features
            }
            inputs = {}
            for key, val in self.accum_features.items():
                accumulated = self.accum_features[key]
                inputs[key] = paddle.concat(
                    accumulated[:j] + [model_out[key]] + accumulated[j + 1:])
            loss, logits_per_image, logits_per_text, labels = modelloss(
                (inputs['image_features'], inputs['text_features'],
                 logit_scale))
            del inputs

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        if self.args.max_grad_norm > 0.0:
            _ = clip_grad_norm(model, self.args.max_grad_norm)

        self.accum_features.clear()
        self.accum_images.clear()
        self.accum_texts.clear()
        self.step = 0

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

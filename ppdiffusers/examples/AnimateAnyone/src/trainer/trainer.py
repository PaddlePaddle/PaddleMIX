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

import os
import os.path as osp
import shutil
from collections import OrderedDict
from typing import Any, Dict, Union

import paddle
from paddle.io import DataLoader
from paddlenlp.trainer import Trainer
from paddlenlp.trainer.integrations import (
    INTEGRATION_TO_CALLBACK,
    VisualDLCallback,
    rewrite_logs,
)
from paddlenlp.utils.log import logger
from src.trainer.dataset import HumanDanceDataset, HumanDanceVideoDataset


class VisualDLWithImageCallback(VisualDLCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if hasattr(model, "on_train_batch_end"):
            model.on_train_batch_end()
        control.should_log = True

    def on_log(self, args, state, control, logs=None, **kwargs):

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


# register visualdl
INTEGRATION_TO_CALLBACK.update({"custom_visualdl": VisualDLWithImageCallback})


class AnimateAnyoneTrainer_stage1(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(**inputs)
        return loss

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if isinstance(self.train_dataset, HumanDanceDataset):
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                num_workers=self.args.dataloader_num_workers,
                worker_init_fn=None,
                shuffle=True,
            )
        else:
            return super().get_train_dataloader()

    def training_step(self, model: paddle.nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
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
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1 and not self._enable_delay_scale_loss():
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self.model.reference_control_reader.clear()
        self.model.reference_control_writer.clear()

        if self.state.global_step % self.args.save_steps == 0:
            self.save_checkpoints(self.args.output_dir, self.state.global_step, total_limit=self.args.save_total_limit)

        return loss.detach()

    def save_checkpoints(self, save_dir, ckpt_num, total_limit=None):

        if total_limit is not None:
            checkpoints = os.listdir(save_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoints-")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))

            if len(checkpoints) >= total_limit:
                num_to_remove = len(checkpoints) - total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]
                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)

        save_dir = osp.join(save_dir, f"checkpoints-{ckpt_num}/")
        denoising_unet_save_path = osp.join(save_dir, "denoising_unet.pdparams")
        denoising_unet_state_dict = self.model.denoising_unet.state_dict()
        paddle.save(denoising_unet_state_dict, denoising_unet_save_path)

        reference_unet_save_path = osp.join(save_dir, "reference_unet.pdparams")
        reference_unet_state_dict = self.model.reference_unet.state_dict()
        paddle.save(reference_unet_state_dict, reference_unet_save_path)

        pose_guider_save_path = osp.join(save_dir, "pose_guider.pdparams")
        pose_guider_state_dict = self.model.pose_guider.state_dict()
        paddle.save(pose_guider_state_dict, pose_guider_save_path)


class AnimateAnyoneTrainer_stage2(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(**inputs)
        return loss

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if isinstance(self.train_dataset, HumanDanceVideoDataset):
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                num_workers=self.args.dataloader_num_workers,
                worker_init_fn=None,
                shuffle=True,
            )
        else:
            return super().get_train_dataloader()

    def training_step(self, model: paddle.nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
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
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1 and not self._enable_delay_scale_loss():
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self.model.reference_control_reader.clear()
        self.model.reference_control_writer.clear()

        if self.state.global_step % self.args.save_steps == 0:
            self.save_motion_module(
                model, self.args.output_dir, self.state.global_step, total_limit=self.args.save_total_limit
            )

        return loss.detach()

    def save_motion_module(self, model, save_dir, ckpt_num, total_limit=None):
        save_path = osp.join(save_dir, f"motion_module-{ckpt_num}.pdparams")

        if total_limit is not None:
            checkpoints = os.listdir(save_dir)
            checkpoints = [d for d in checkpoints if d.startswith("motion_module")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))

            if len(checkpoints) >= total_limit:
                num_to_remove = len(checkpoints) - total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]
                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                    os.remove(removing_checkpoint)

        mm_state_dict = OrderedDict()
        state_dict = model.state_dict()
        for key in state_dict:
            if "motion_module" in key:
                mm_state_dict[key] = state_dict[key]

        paddle.save(mm_state_dict, save_path)

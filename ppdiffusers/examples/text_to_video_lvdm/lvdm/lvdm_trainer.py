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
import sys

import paddle
import paddle.amp.auto_cast as autocast
from paddle.io import DataLoader
from paddlenlp.trainer import Trainer
from paddlenlp.trainer.integrations import (
    INTEGRATION_TO_CALLBACK,
    VisualDLCallback,
    rewrite_logs,
)
from paddlenlp.trainer.utils.helper import nested_detach
from paddlenlp.utils.log import logger

from .frame_dataset import VideoFrameDataset
from .webvid_dataset import WebVidDataset


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
        # only on card 0
        if not state.is_world_process_zero:
            return
        # log video frames on each node
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
                image_logs["reconstruction"] = model.log_reconstruct_frames(pixel_values=inputs["pixel_values"])
                if model.task_type == "text2video":
                    image_logs["ddim-samples"] = model.log_text2video_sample_frames(
                        input_ids=inputs["input_ids"],
                        height=256,
                        width=256,
                        eta=1.0,
                        guidance_scale=9,
                        num_frames=16,
                    )
                elif model.task_type == "short":
                    image_logs["ddim-samples"] = model.log_short_sample_frames(
                        height=256,
                        width=256,
                        eta=1.0,
                        num_frames=16,
                    )

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


# register visualdl_with_image
INTEGRATION_TO_CALLBACK.update({"custom_visualdl": VisualDLWithImageCallback})


class LatentVideoDiffusionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(**inputs)
        return loss

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if isinstance(self.train_dataset, VideoFrameDataset) or isinstance(self.train_dataset, WebVidDataset):
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                num_workers=self.args.dataloader_num_workers,
                shuffle=True,
                worker_init_fn=None,
                collate_fn=None,
            )
        else:
            return super().get_train_dataloader()

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys,
    ):
        if self.args.pipeline_parallel_degree > 1:
            # hack for pipeline mode
            inputs = self._prepare_inputs(inputs)
            return self.prediction_pipeline_step(model, inputs, prediction_loss_only, ignore_keys)

        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with paddle.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    loss = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

        return (loss, None, None)

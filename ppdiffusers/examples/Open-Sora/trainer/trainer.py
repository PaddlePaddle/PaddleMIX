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

from dataset.datasets import VariableVideoTextDataset
from dataset.sampler import VariableVideoBatchSampler
from paddle.io import DataLoader
from paddlenlp.trainer import Trainer
from paddlenlp.trainer.integrations import (
    INTEGRATION_TO_CALLBACK,
    VisualDLCallback,
    rewrite_logs,
)
from paddlenlp.utils.log import logger


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


class OpenSoraTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(**inputs)
        return loss

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if isinstance(self.train_dataset, VariableVideoTextDataset):

            bucket_config = {
                "144p": {1: (0.5, 48), 16: (1.0, 6), 32: (1.0, 3), 96: (1.0, 1)},
                "256": {1: (0.5, 24), 16: (0.5, 3), 48: (0.5, 1), 64: (0.0, None)},
                "240p": {16: (0.3, 2), 32: (0.3, 1), 64: (0.0, None)},
                "512": {1: (0.4, 12)},
                "1024": {1: (0.3, 3)},
            }

            batch_sampler = VariableVideoBatchSampler(
                self.train_dataset,
                bucket_config,
                shuffle=True,
                seed=self.args.seed,
                drop_last=True,
                verbose=True,
                num_bucket_build_workers=1,
            )

            data_loader = DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=1,
            )

            return data_loader
        else:
            return super().get_train_dataloader()

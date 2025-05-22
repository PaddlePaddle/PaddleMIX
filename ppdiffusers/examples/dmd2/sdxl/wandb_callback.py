# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import wandb
from paddle.distributed import fleet
from paddlenlp.trainer.integrations import TrainerCallback
from utils import (
    draw_probability_histogram,
    draw_valued_array,
    prepare_images_for_saving,
)


class WandbCallback(TrainerCallback):
    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args
        run = wandb.init(
            config=args,
            dir=args.log_path,
            **{"mode": "offline", "entity": args.wandb_entity, "project": args.wandb_project},
        )
        wandb.run.log_code(".")
        wandb.run.name = args.wandb_name
        print(f"run dir: {run.dir}")
        self.wandb_folder = run.dir
        self.accelerator = accelerator
        os.makedirs(self.wandb_folder, exist_ok=True)

        # Initialize communication for distributed training
        if paddle.distributed.get_world_size() > 1 and hasattr(fleet.fleet, "_hcg"):
            hcg = fleet.get_hybrid_communicate_group()
            self._sharding_world_size = max(1, hcg.get_sharding_parallel_world_size())
            self._sharding_rank = max(0, hcg.get_sharding_parallel_rank())
        else:
            self._sharding_world_size = 1
            self._sharding_rank = 0

    def on_step_end(self, args, state, control, **kwargs):
        """on_step_end"""
        cur_step = state.global_step - 1
        if cur_step % self.args.wandb_iters == 0 and self._sharding_rank == 0:
            log_dict = state._log_dict
            generator_loss_dict = log_dict.pop("generator_loss_dict", None)
            guidance_loss_dict = log_dict.pop("guidance_loss_dict", None)

            loss_dict = {**generator_loss_dict, **guidance_loss_dict}

            with paddle.no_grad():
                if not self.args.gan_alone:

                    data_dict = {
                        "loss_dm": loss_dict["loss_dm"].item(),
                    }
                else:
                    data_dict = {}

                generated_image = log_dict["generated_image"]
                generated_image_grid = prepare_images_for_saving(
                    generated_image, resolution=self.args.resolution, grid_size=self.args.grid_size
                )

                # generated_image_mean = generated_image.mean()
                # generated_image_std = generated_image.std()

                data_dict.update(
                    {
                        "generated_image": wandb.Image(generated_image_grid),
                        "loss_fake_mean": loss_dict["loss_fake_mean"].item(),
                    }
                )

                if self.args.denoising:
                    origianl_clean_image = log_dict["original_clean_image"]
                    origianl_clean_image_grid = prepare_images_for_saving(
                        origianl_clean_image, resolution=self.args.resolution, grid_size=self.args.grid_size
                    )

                    denoising_timestep = log_dict["denoising_timestep"]
                    denoising_timestep_grid = draw_valued_array(
                        denoising_timestep.cpu().numpy(), output_dir=self.wandb_folder, grid_size=self.args.grid_size
                    )

                    data_dict.update(
                        {
                            "original_clean_image": wandb.Image(origianl_clean_image_grid),
                            "denoising_timestep": wandb.Image(denoising_timestep_grid),
                        }
                    )

                if self.args.cls_on_clean_image:
                    data_dict["guidance_cls_loss"] = loss_dict["guidance_cls_loss"].item()

                    if self.args.gen_cls_loss:
                        data_dict["gen_cls_loss"] = loss_dict["gen_cls_loss"].item()

                    pred_realism_on_fake = log_dict["pred_realism_on_fake"]
                    pred_realism_on_real = log_dict["pred_realism_on_real"]
                    hist_pred_realism_on_fake = draw_probability_histogram(pred_realism_on_fake.cpu().numpy())
                    hist_pred_realism_on_real = draw_probability_histogram(pred_realism_on_real.cpu().numpy())

                    data_dict.update(
                        {
                            "hist_pred_realism_on_fake": wandb.Image(hist_pred_realism_on_fake),
                            "hist_pred_realism_on_real": wandb.Image(hist_pred_realism_on_real),
                        }
                    )

                wandb.log(data_dict, step=cur_step)

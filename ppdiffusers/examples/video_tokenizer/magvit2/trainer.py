# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from pathlib import Path

import paddle
from beartype import beartype
from beartype.typing import Literal, Optional, Type, Union
from data import (
    ImageDataset,
    VideoDataset,
    collate_tensors_and_strings,
    video_tensor_to_gif,
)
from einops import rearrange
from magvit2 import VideoTokenizer
from optimizer import get_optimizer

VideosOrImagesLiteral = Union[Literal["videos"], Literal["images"]]


def exists(v):
    return v is not None


def cycle(dl):
    while True:
        for data in dl:
            yield data


class VideoTokenizerTrainer:
    @beartype
    def __init__(
        self,
        model: VideoTokenizer,
        *,
        batch_size: int,
        num_train_steps: int,
        learning_rate: float = 1e-05,
        grad_accum_every: int = 1,
        apply_gradient_penalty_every: int = 4,
        max_grad_norm: Optional[float] = None,
        dataset: Optional[paddle.io.Dataset] = None,
        dataset_folder: Optional[str] = None,
        dataset_type: VideosOrImagesLiteral = "videos",
        checkpoints_folder="./checkpoints",
        results_folder="./results",
        random_split_seed=42,
        valid_frac=0.05,
        validate_every_step=100,
        checkpoint_every_step=100,
        num_frames=17,
        use_wandb_tracking=False,
        discr_start_after_step=0.0,
        warmup_steps=1000,
        scheduler: Optional[Type[paddle.optimizer.lr.LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        dataset_kwargs: dict = dict()
    ):
        self.use_wandb_tracking = use_wandb_tracking

        self.model = model

        dataset_kwargs.update(channels=model.channels)
        if not exists(dataset):
            if dataset_type == "videos":
                dataset_klass = VideoDataset
                dataset_kwargs = {**dataset_kwargs, "num_frames": num_frames}
            else:
                dataset_klass = ImageDataset
            assert exists(dataset_folder)
            dataset = dataset_klass(dataset_folder, image_size=model.image_size, **dataset_kwargs)

        assert 0 <= valid_frac < 1.0
        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(dataset))
            valid_size = len(dataset) - train_size
            dataset, valid_dataset = paddle.io.random_split(dataset=dataset, lengths=[train_size, valid_size])
            print(
                f"training with dataset of {len(dataset)} samples and validating with randomly splitted {len(valid_dataset)} samples"
            )
        else:
            valid_dataset = dataset
            print(f"training with shared training and valid dataset of {len(dataset)} samples")
        self.dataset = dataset
        self.dataloader = paddle.io.DataLoader(
            dataset, shuffle=True, drop_last=True, batch_size=batch_size, collate_fn=collate_tensors_and_strings
        )
        self.valid_dataset = valid_dataset
        self.valid_dataloader = paddle.io.DataLoader(
            valid_dataset, shuffle=True, drop_last=True, batch_size=batch_size, collate_fn=collate_tensors_and_strings
        )
        self.validate_every_step = validate_every_step
        self.checkpoint_every_step = checkpoint_every_step
        self.max_grad_norm = max_grad_norm

        optimizer_kwargs["max_grad_norm"] = max_grad_norm

        if exists(scheduler):
            self.scheduler = scheduler(learning_rate, **scheduler_kwargs)
            self.discr_scheduler = scheduler(learning_rate, **scheduler_kwargs)
        else:
            self.scheduler = paddle.optimizer.lr.LambdaDecay(
                learning_rate=learning_rate, lr_lambda=lambda x: min(1.0, (x + 1) / warmup_steps)
            )

            self.discr_scheduler = paddle.optimizer.lr.LambdaDecay(
                learning_rate=learning_rate, lr_lambda=lambda x: min(1.0, (x + 1) / warmup_steps)
            )

        self.optimizer = get_optimizer(
            model.parameters(), lr=learning_rate, lr_scheduler=self.scheduler, **optimizer_kwargs
        )

        self.discr_optimizer = get_optimizer(
            model.discr_parameters(), lr=learning_rate, lr_scheduler=self.discr_scheduler, **optimizer_kwargs
        )

        self.batch_size = batch_size
        self.num_train_steps = num_train_steps
        self.grad_accum_every = grad_accum_every

        self.apply_gradient_penalty_every = apply_gradient_penalty_every

        self.discr_start_after_step = discr_start_after_step
        self.has_multiscale_discrs = self.model.has_multiscale_discrs
        self.multiscale_discr_optimizers = []
        for ind, discr in enumerate(self.model.multiscale_discrs):
            multiscale_optimizer = get_optimizer(discr.parameters(), lr=learning_rate, **optimizer_kwargs)
            self.multiscale_discr_optimizers.append(multiscale_optimizer)

        checkpoints_folder = Path(checkpoints_folder)
        results_folder = Path(results_folder)
        checkpoints_folder.mkdir(parents=True, exist_ok=True)
        results_folder.mkdir(parents=True, exist_ok=True)
        assert checkpoints_folder.is_dir()
        assert results_folder.is_dir()
        self.checkpoints_folder = checkpoints_folder
        self.results_folder = results_folder
        self.step = 0

    @property
    def device(self):
        return self.model.place

    @property
    def ema_tokenizer(self):
        return self.ema_model.ema_model

    def tokenize(self, *args, **kwargs):
        return self.ema_tokenizer.tokenize(*args, **kwargs)

    def save(self, path, overwrite=True):
        path = Path(path)
        assert overwrite or not path.exists()
        state_dict = self.model.state_dict()
        paddle.save(state_dict, path=str(path))

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = paddle.load(path=str(path))
        self.model.set_state_dict(state_dict=pkg)

    def train_step(self, dl_iter):
        self.model.train()
        step = self.step
        train_adversarially = self.model.use_gan and step + 1 > self.discr_start_after_step
        adversarial_loss_weight = 0.0 if not train_adversarially else None
        multiscale_adversarial_loss_weight = 0.0 if not train_adversarially else None

        self.optimizer.clear_grad()

        for grad_accum_step in range(self.grad_accum_every):

            data, *_ = next(dl_iter)

            loss, loss_breakdown = self.model(
                data,
                return_loss=True,
                adversarial_loss_weight=adversarial_loss_weight,
                multiscale_adversarial_loss_weight=multiscale_adversarial_loss_weight,
            )
            loss = loss / self.grad_accum_every

            loss.backward()

        print(f"recon loss: {loss_breakdown.recon_loss.item():.3f}")

        self.optimizer.step()

        self.scheduler.step()

        if not train_adversarially:
            self.step += 1
            return

        self.discr_optimizer.clear_grad()
        if self.has_multiscale_discrs:
            for multiscale_discr_optimizer in self.multiscale_discr_optimizers:
                multiscale_discr_optimizer.clear_grad()

        apply_gradient_penalty = not step % self.apply_gradient_penalty_every
        apply_gradient_penalty = False

        for grad_accum_step in range(self.grad_accum_every):

            data, *_ = next(dl_iter)

            discr_loss, discr_loss_breakdown = self.model(
                data, return_discr_loss=True, apply_gradient_penalty=apply_gradient_penalty
            )

            discr_loss = discr_loss / self.grad_accum_every
            discr_loss.backward()
        print(f"discr loss: {discr_loss_breakdown.discr_loss.item():.3f}")

        self.discr_optimizer.step()

        self.discr_scheduler.step()

        if self.has_multiscale_discrs:
            for multiscale_discr_optimizer in self.multiscale_discr_optimizers:
                multiscale_discr_optimizer.step()
        self.step += 1

    @paddle.no_grad()
    def valid_step(self, dl_iter, save_recons=True, num_save_recons=1):

        recon_loss = 0.0
        valid_videos = []
        recon_videos = []
        for _ in range(self.grad_accum_every):
            (valid_video,) = next(dl_iter)

            loss, _ = self.model(valid_video, return_recon_loss_only=True)
            recon_loss, recon_video = self.model(valid_video, return_recon_loss_only=True)
            recon_loss += loss / self.grad_accum_every

            if valid_video.ndim == 4:
                valid_video = rearrange(valid_video, "b c h w -> b c 1 h w")
            valid_videos.append(valid_video)
            recon_videos.append(recon_video)

        print(f"validation recon loss {recon_loss.item():.3f}")
        if not save_recons:
            return

        valid_videos = paddle.concat(x=valid_videos)
        recon_videos = paddle.concat(x=recon_videos)
        recon_videos.clip_(min=0.0, max=1.0)
        valid_videos, recon_videos = map(lambda t: t[:num_save_recons], (valid_videos, recon_videos))
        real_and_recon = rearrange([valid_videos, recon_videos], "n b c f h w -> c f (b h) (n w)")
        validate_step = self.step // self.validate_every_step
        sample_path = str(self.results_folder / f"sampled.{validate_step}.gif")
        video_tensor_to_gif(real_and_recon, str(sample_path))

        print(f"sample saved to {str(sample_path)}")

    def train(self):
        step = self.step
        dl_iter = cycle(self.dataloader)
        valid_dl_iter = cycle(self.valid_dataloader)
        while step < self.num_train_steps:
            print(f"step {step}")

            self.train_step(dl_iter)

            if not step % self.validate_every_step:
                self.valid_step(valid_dl_iter)

            if not step % self.checkpoint_every_step:
                checkpoint_num = step // self.checkpoint_every_step
                checkpoint_path = self.checkpoints_folder / f"checkpoint.{checkpoint_num}.pdparams"
                self.save(str(checkpoint_path))

            step += 1

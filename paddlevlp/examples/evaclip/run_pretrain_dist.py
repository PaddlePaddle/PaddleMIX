# coding:utf-8
import sys
import os
import numpy as np
import time
import pprint
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
try:
    from paddle.fluid.dygraph.parallel import sync_params_buffers
except ImportError:
    from paddle.distributed.parallel import sync_params_buffers

from dataclasses import dataclass, field

from paddlevlp.models.evaclip.eva_clip.model import EVACLIP
from paddlevlp.models.evaclip.eva_clip.coca_model import CoCa
from paddlevlp.models.evaclip.training.optim import create_optimizer
from paddlevlp.models.evaclip.utils.checkpoint import save, load_model
from paddlevlp.optimization import CosineDecayWithWarmup
from paddlevlp.datasets import load_dataset
from paddlevlp.utils.env import set_hyrbid_parallel_seed

from paddlenlp.trainer import (PdArgumentParser, TrainingArguments,
                               get_last_checkpoint)
from paddlenlp.trainer import Trainer

import socket
import time
import random

from paddlevlp.processors.clip_processing import CLIPProcessor


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(
        default="coco_clip",
        metadata={
            "help": "The name of the task to use (via the datasets library)."
        }, )

    image_size: int = field(
        default=224,
        metadata={"help": "image size for training"}, )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model: str = field(
        default="EVA02-CLIP-B-16",
        metadata={
            "help":
            "model name to create, for example [EVA02-CLIP-B-16/coca_EVA02-B-16]"
        }, )
    model_name_or_path: str = field(
        default="clip",
        metadata={"help": "Path to pretrained model or model identifier"}, )
    coca_caption_loss_weight: float = field(
        default=2.0,
        metadata={"help": "coca_caption_loss_weight set, default: 2.0"}, )
    coca_contrastive_loss_weight: float = field(
        default=1.0,
        metadata={"help": "coca_contrastive_loss_weight set, default: 1.0"}, )


@dataclass
class PreTrainingArguments(TrainingArguments):
    """
    Arguments pertaining to what training options we are going to use during pretraining.
    """

    pretrained_model_path: str = field(
        default=None,
        metadata={
            "help":
            "The path to pre-trained model that we will use for pretraining."
        }, )
    text_wd: float = field(
        default=0.05, metadata={"help": "Weight decay for text tower"})
    visual_wd: float = field(
        default=0.05, metadata={"help": "Weight decay for visual tower"})
    text_lr: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate of text tower."})
    visual_lr: float = field(
        default=2e-4,
        metadata={"help": "The initial learning rate of text tower."})
    layer_decay: float = field(
        default=1.0, metadata={"help": "The basic layer decay."})
    text_ld: float = field(
        default=0.75, metadata={"help": "The layer decay of text tower."})
    visual_ld: float = field(
        default=0.75, metadata={"help": "The layer decay of text tower."})
    start_epoch: int = field(
        default=0,
        metadata={"help": " manual epoch number (useful on restarts)"}, )
    context_length: int = field(
        default=77,
        metadata={"help": " context length for text."}, )
    optimizer: str = field(
        default="lamb", metadata={"help": "optimizer setting, [lamb/adamw]"})
    dp_degree: int = field(
        default=2,
        metadata={"help": " data parallel degrees."}, )
    last_epoch: int = field(
        default=-1, metadata={"help": "the last epoch to resume"})
    accum_freq: int = field(
        default=1, metadata={"help": "accum frequency (default: 1)"})


class SelfTrainer(Trainer):
    def __init__(self, **kwargs):
        """
        自定义训练器，与Trainer的区别：
        1、自定义优化器策略
        2、支持accum_freq训练
        
        Args:
            kwargs (dict): 包含任意传入的参数及对应的值
        
        Returns:
            None
        """
        super().__init__(**kwargs)
        if self.args.accum_freq > 1:
            self.accum_features = {}
            self.accum_images = []
            self.accum_texts = []
            self.step = 0

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            1.0,
            num_training_steps - self.args.warmup_steps,
            last_epoch=self.args.last_epoch)
        if self.args.warmup_steps > 0:
            self.lr_scheduler = paddle.optimizer.lr.LinearWarmup(
                self.lr_scheduler,
                self.args.warmup_steps,
                0,
                1.0,
                last_epoch=self.args.last_epoch)
        self.optimizer = create_optimizer(self.args, self.model,
                                          self.lr_scheduler)

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
        self.accum_features.clear()
        self.accum_images.clear()
        self.accum_texts.clear()
        self.step = 0

        return loss.detach()


class Collator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.

    Args:
        processor (`paddlevlp.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, data_list):
        images = [sample["image"] for sample in data_list]
        text = [sample["text"] for sample in data_list]
        batch = self.processor(
            images=images,
            text=text,
            max_length=77,
            return_tensors="pd",
            return_attention_mask=False,
            mode="train", )
        return batch


def main_worker(training_args, model_args, data_args):
    if model_args.model.startswith("coca"):
        model = CoCa.from_pretrained(
            "EVA/" + model_args.model,
            data_world_rank=training_args.data_world_rank,
            data_world_size=training_args.data_world_size,
            ignore_mismatched_sizes=True)
    else:
        model = EVACLIP.from_pretrained(
            "EVA/" + model_args.model,
            data_world_rank=training_args.data_world_rank,
            data_world_size=training_args.data_world_size)

    if training_args.pretrained_model_path and training_args.pretrained_model_path != "None" and training_args.resume_from_checkpoint is None:
        load_model(
            training_args, model, ckpt_dir=training_args.pretrained_model_path)

    train_dataset = load_dataset('coco_clip', splits="train")

    processor = CLIPProcessor.from_pretrained(model_args.model_name_or_path)
    collator = Collator(processor)

    trainer = SelfTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator, )

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()


def setdistenv(args):
    args.dp_degree = dist.get_world_size() // (args.tensor_parallel_degree *
                                               args.sharding_parallel_degree *
                                               args.pipeline_parallel_degree)
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": args.dp_degree,
        "mp_degree": args.tensor_parallel_degree,
        "sharding_degree": args.sharding_parallel_degree,
        "pp_degree": args.pipeline_parallel_degree,
    }
    strategy.find_unused_parameters = True

    # set control in tensor parallel
    strategy.tensor_parallel_configs = {"tensor_init_seed": args.seed}

    fleet.init(is_collective=True, strategy=strategy)

    # if paddle.distributed.get_world_size() > 1:
    #     paddle.distributed.init_parallel_env()

    args.rank = dist.get_rank()
    # obtain rank message of hybrid parallel
    hcg = fleet.get_hybrid_communicate_group()
    args.mp_rank = hcg.get_model_parallel_rank()
    args.dp_rank = hcg.get_data_parallel_rank()
    args.sharding_rank = hcg.get_sharding_parallel_rank()

    args.data_world_rank = args.dp_rank * args.sharding_parallel_degree + args.sharding_rank
    args.data_world_size = dist.get_world_size() // abs(
        args.tensor_parallel_degree * args.pipeline_parallel_degree)

    # seed control in hybrid parallel
    set_hyrbid_parallel_seed(args.seed, args.data_world_rank, args.mp_rank)


if __name__ == "__main__":
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, PreTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # paddle.set_flags({'FLAGS_cudnn_deterministic': True})
    # paddle.set_flags({'FLAGS_embedding_deterministic': 1})
    training_args.hostname = socket.gethostname()
    if training_args.accum_freq > 1:
        training_args.gradient_accumulation_steps = training_args.accum_freq
    pprint.pprint(data_args)
    pprint.pprint(model_args)
    pprint.pprint(training_args)

    setdistenv(training_args)
    model_args.data_world_rank = training_args.data_world_rank
    model_args.data_world_size = training_args.data_world_size
    main_worker(training_args, model_args, data_args)

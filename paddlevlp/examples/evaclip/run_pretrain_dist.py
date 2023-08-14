# coding:utf-8
import sys
import os
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
sys.path.insert(0, parent_path)
import pprint
import socket
from dataclasses import dataclass, field
import paddle

from paddlevlp.models.evaclip.eva_clip_model import EVACLIP
from paddlevlp.optimization import create_optimizer
from paddlevlp.checkpoint import save, load_model
from paddlevlp.datasets import load_dataset
from paddlevlp.utils.env import setdistenv
from paddlevlp.trainer import CLIPTrainer
from paddlevlp.processors.clip_processing import CLIPProcessor, CLIPImageProcessor, CLIPTextProcessor
from paddlevlp.processors import SimpleTokenizer
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.trainer import (PdArgumentParser, TrainingArguments,
                               get_last_checkpoint)


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

    train_data: str = field(
        default="",
        metadata={"help": "The traindata list path."}, )

    precomputed_text_emb: str = field(
        default="open_clip_vit_g_14",
        metadata={"help": "precomputed_text_emb name."}, )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model: str = field(
        default="paddlemix/EVA/EVA02-CLIP-L-14",
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
        metadata={"help": "The initial learning rate of visual tower."})
    layer_decay: float = field(
        default=1.0, metadata={"help": "The basic layer decay."})
    text_ld: float = field(
        default=0.75, metadata={"help": "The layer decay of text tower."})
    visual_ld: float = field(
        default=0.75, metadata={"help": "The layer decay of visual tower."})
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
    gather_with_grad: bool = field(
        default=False,
        metadata={"help": "Whether to use gather_with_grad in loss."}, )
    local_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use local loss in loss."}, )
    tensorboard: bool = field(
        default=False,
        metadata={"help": "Whether to use tensorboard to record loss."}, )


class SelfTrainer(CLIPTrainer):
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
    if training_args.bf16 and training_args.fp16_opt_level == 'O2':
        paddle.set_default_dtype("bfloat16")

    config = EVACLIPConfig.from_pretrained(model_args.model)
    model = EVACLIP(
        config,
        disable_text=False,
        local_loss=training_args.local_loss,
        gather_with_grad=training_args.gather_with_grad,
        data_world_rank=training_args.data_world_rank,
        data_world_size=training_args.data_world_size)

    training_args.model = model_args.model
    if training_args.pretrained_model_path and training_args.pretrained_model_path != "None" and training_args.resume_from_checkpoint is None:
        load_model(
            training_args, model, ckpt_dir=training_args.pretrained_model_path)
    if training_args.bf16 and training_args.fp16_opt_level == 'O2':
        paddle.set_default_dtype("float32")

    train_dataset = load_dataset('coco_clip', splits="train")
    image_processor = CLIPImageProcessor.from_pretrained(
        model_args.model_name_or_path)
    text_processor = CLIPTextProcessor.from_pretrained(
        model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    processor = CLIPProcessor(image_processor, text_processor, tokenizer)
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


from paddlevlp.models.evaclip.eva_clip_model import EVACLIPConfig, EVACLIP
if __name__ == "__main__":
    parser = PdArgumentParser(
        (ModelArguments, DataArguments, PreTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.hostname = socket.gethostname()
    pprint.pprint(data_args)
    pprint.pprint(model_args)
    pprint.pprint(training_args)

    setdistenv(training_args)
    model_args.data_world_rank = training_args.data_world_rank
    model_args.data_world_size = training_args.data_world_size
    main_worker(training_args, model_args, data_args)

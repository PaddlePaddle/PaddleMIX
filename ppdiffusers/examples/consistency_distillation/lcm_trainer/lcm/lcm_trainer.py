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

import contextlib
import os
import sys
import time

import numpy as np
import paddle
import paddle.amp.auto_cast as autocast
from paddle.io import DataLoader
from paddlenlp.trainer import PrinterCallback, ProgressCallback, Trainer
from paddlenlp.trainer.integrations import (
    INTEGRATION_TO_CALLBACK,
    TrainerCallback,
    VisualDLCallback,
    rewrite_logs,
)
from paddlenlp.transformers.model_utils import _add_variant
from paddlenlp.utils import profiler
from paddlenlp.utils.log import logger

from ppdiffusers.training_utils import unwrap_model

from .text_image_pair_dataset import TextImagePair, worker_init_fn

PADDLE_WEIGHTS_NAME = "model_state.pdparams"
TRAINING_ARGS_NAME = "training_args.bin"


class VisualDLWithImageCallback(VisualDLCallback):
    def autocast_smart_context_manager(self, args):
        if args.fp16 or args.bf16:
            amp_dtype = "float16" if args.fp16 else "bfloat16"
            custom_black_list = ["reduce_sum", "c_softmax_with_cross_entropy"]
            custom_white_list = []
            if args.fp16_opt_level == "O2":
                # https://github.com/PaddlePaddle/Paddle/blob/eb97f4f0adca40b16a309b927e480178beb8ae96/python/paddle/amp/amp_lists.py#L85-L86
                # the lookup_table is in black_list, but in O2, we need it return fp16
                custom_white_list.extend(["lookup_table", "lookup_table_v2"])

            if hasattr(args, "amp_custom_white_list"):
                if args.amp_custom_white_list is not None:
                    custom_white_list.extend(args.amp_custom_white_list)
            if hasattr(args, "amp_custom_black_list"):
                if args.amp_custom_black_list is not None:
                    custom_black_list.extend(args.amp_custom_black_list)

            ctx_manager = autocast(
                True,
                custom_black_list=set(custom_black_list),
                custom_white_list=set(custom_white_list),
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
        # log image on each node
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
                max_batch = 4 if args.resolution > 256 else 8
                if model.is_sdxl:
                    max_batch = 2
                # vae reconstruction log
                image_logs["reconstruction"] = model.decode_image(
                    pixel_values=inputs["pixel_values"],
                    max_batch=max_batch,
                )
                # train sample log
                image_logs["sample"] = model.log_image(
                    input_ids=inputs["input_ids"],
                    input_ids_2=inputs.get("input_ids_2", None),
                    height=args.resolution,
                    width=args.resolution,
                    max_batch=max_batch,
                )

                # validation_prompts log
                validation_prompts = [
                    "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
                    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
                    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                    "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
                ]
                logs_prefix_list = (
                    ["eval_prompt/"] if model.is_lora else ["eval_prompt/online_unet/", "eval_prompt/target_unet/"]
                )

                for logs_prefix in logs_prefix_list:
                    for prompt in validation_prompts:
                        image_logs[logs_prefix + prompt] = model.log_image(
                            prompt=prompt,
                            height=args.resolution,
                            width=args.resolution,
                            max_batch=max_batch,
                            seed=args.seed,
                            unet=model.target_unet
                            if ("target_unet" in logs_prefix) and hasattr(model, "target_unet")
                            else None,
                        )
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
            # log images
            for k, v in image_logs.items():
                self.vdl_writer.add_image(k, v, state.global_step, dataformats="NHWC")
            self.vdl_writer.flush()


class AverageStatistical(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_cnt = 0
        self.time = 0

    def record(self, val, cnt=1):
        self.time += val
        self.total_cnt += cnt

    def get_average(self):
        if self.total_cnt == 0:
            return 0

        return self.time / self.total_cnt

    def get_average_per_sec(self):
        if self.time == 0.0:
            return 0.0

        return float(self.total_cnt) / self.time

    def get_total_cnt(self):
        return self.total_cnt

    def get_total_time(self):
        return self.time


class BenchmarkCallback(TrainerCallback):
    def __init__(self, benchmark=True, profiler_options=None):
        self.benchmark = benchmark
        self.profiler_options = profiler_options

    def on_train_begin(self, args, state, control, **kwargs):
        assert args.gradient_accumulation_steps == 1 and not args.do_eval and not args.do_predict
        if self.benchmark:
            self.reader_cost_avg = AverageStatistical()

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.benchmark:
            self.epoch_start = time.time()
            self.batch_start = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        if self.benchmark:
            self.reader_cost_avg.record(time.time() - self.batch_start)

    def on_step_end(self, args, state, control, **kwargs):
        if self.profiler_options is not None:
            profiler.add_profiler_step(self.profiler_options)

        if self.benchmark:
            self.batch_start = time.time()
            if control.should_log:
                self.maybe_log_save_evaluate_start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.benchmark:
            if logs is not None and "interval_steps_per_second" in logs:
                self.batch_start = self.batch_start + (time.time() - self.maybe_log_save_evaluate_start)
                ips = logs["interval_steps_per_second"] * args.train_batch_size
                avg_batch_cost = 1 / logs["interval_steps_per_second"]
                max_mem_reserved_msg = ""
                max_mem_allocated_msg = ""
                if paddle.device.is_compiled_with_cuda():
                    max_mem_reserved_msg = (
                        f"max_mem_reserved: {paddle.device.cuda.max_memory_reserved() // (1024 ** 2)} MB,"
                    )
                    max_mem_allocated_msg = (
                        f"max_mem_allocated: {paddle.device.cuda.max_memory_allocated() // (1024 ** 2)} MB"
                    )
                logger.info(
                    "global step %d / %d, loss: %f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, avg_samples: %.5f, ips: %.5f sample/sec, %s %s"
                    % (
                        state.global_step,
                        state.max_steps,
                        logs["loss"],
                        self.reader_cost_avg.get_average(),
                        avg_batch_cost,
                        args.train_batch_size,
                        ips,
                        max_mem_reserved_msg,
                        max_mem_allocated_msg,
                    )
                )
                self.reader_cost_avg.reset()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.benchmark:
            train_epoch_cost = time.time() - self.epoch_start
            logger.info("train epoch: %d, epoch_cost: %.5f s" % (state.epoch, train_epoch_cost))


# register visualdl_with_image
INTEGRATION_TO_CALLBACK.update({"custom_visualdl": VisualDLWithImageCallback})


def get_module_kohya_state_dict(module, prefix: str = ""):
    kohya_ss_state_dict = {}
    if prefix is not None and prefix != "" and not prefix.endswith("."):
        prefix += "."

    for peft_key, weight in module.get_trainable_state_dict().items():
        kohya_key = prefix + peft_key.rstrip(".weight")
        kohya_key = kohya_key.replace("lora_A", "lora_down.weight")
        kohya_key = kohya_key.replace("lora_B", "lora_up.weight")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        # transpose 2D weight for torch
        if weight.ndim == 2:
            weight = weight.T
        kohya_ss_state_dict[kohya_key] = np.ascontiguousarray(weight)

        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = np.ascontiguousarray(
                paddle.to_tensor(module.lora_config.lora_alpha, dtype=weight.dtype)
            )

    return kohya_ss_state_dict


class LCMTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.args.benchmark or self.args.profiler_options is not None:
            self.add_callback(
                BenchmarkCallback(
                    benchmark=self.args.benchmark,
                    profiler_options=self.args.profiler_options,
                )
            )
            if self.args.benchmark:
                if self.args.disable_tqdm:
                    self.pop_callback(PrinterCallback)
                else:
                    self.pop_callback(ProgressCallback)

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(**inputs)
        return loss

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if isinstance(self.train_dataset, TextImagePair):
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                num_workers=self.args.dataloader_num_workers,
                worker_init_fn=worker_init_fn,
            )
        else:
            return super().get_train_dataloader()

    def _save(self, output_dir=None, state_dict=None, merge_tensor_parallel=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if self.args.only_save_updated_model:
            unwraped_model = unwrap_model(self.model)
            if unwraped_model.is_lora:
                logger.info(f"Saving lcm unet lora checkpoint to `{output_dir}/lora`.")
                unwraped_model.unet.save_pretrained(os.path.join(output_dir, "lora"), save_model_config=False)
                from safetensors.numpy import save_file

                lora_kohya_state_dict = get_module_kohya_state_dict(unwraped_model.unet, prefix="lora_unet")
                save_file(
                    lora_kohya_state_dict,
                    os.path.join(output_dir, "lora", "lcm_lora.safetensors"),
                    metadata={"format": "pt"},
                )
            else:
                logger.info(f"Saving unet checkpoint to `{output_dir}/unet` and `{output_dir}/unet_target`.")
                unwraped_model.unet.save_pretrained(os.path.join(output_dir, "unet"))
                unwraped_model.target_unet.save_pretrained(os.path.join(output_dir, "unet_target"))
        else:
            logger.info(f"Saving model checkpoint to {output_dir}")
            if state_dict is None:
                state_dict = self.model.state_dict()
            paddle.save(
                state_dict,
                os.path.join(
                    output_dir,
                    _add_variant(PADDLE_WEIGHTS_NAME, self.args.weight_name_suffix),
                ),
            )
            if self.args.should_save:
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(output_dir)
                paddle.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

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

import json
import os
import time
from threading import Thread
from typing import TYPE_CHECKING, Any, Dict, Optional

import paddle
from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.peft import LoRAConfig, LoRAModel
from paddlenlp.trainer.trainer import TRAINING_ARGS_NAME, IntervalStrategy, Trainer

from ..models.qwen2_vl import MIXQwen2Tokenizer
from ..models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from ..processors.qwen2_vl_processing import Qwen2VLImageProcessor, Qwen2VLProcessor
from .common import DEFAULT_CONFIG_DIR, get_dataset, get_save_dir
from .extras.args import get_train_args, load_args, save_args
from .extras.callbacks import LogCallback
from .extras.constants import IGNORE_INDEX, PEFT_METHODS, TRAINING_STAGES
from .extras.packages import is_gradio_available
from .extras.template import get_template_and_fix_tokenizer
from .extras.training import get_eval_results, get_trainer_info
from .locales import ALERTS

if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from .manager import Manager


class Runner:
    def __init__(self, manager: "Manager", demo_mode: bool = False) -> None:
        self.manager = manager
        self.demo_mode = demo_mode
        """ Resume """
        self.trainer: Optional["Thread"] = None
        self.do_train = True
        self.running_data: Dict["Component", Any] = None
        """ State """
        self.aborted = False
        self.running = False
        self.dataset = None

    def set_abort(self, data) -> None:
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        lang = get("top.lang")

        if self.running:
            self.aborted = True
            yield ALERTS["info_aborting"][lang], gr.Slider(visible=False)
        else:
            yield ALERTS["info_aborting_error"][lang], gr.Slider(visible=False)

    def _initialize(self, data: Dict["Component", Any], do_train: bool, from_preview: bool) -> str:
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        lang, model_name, model_path = get("top.lang"), get("top.model_name"), get("top.model_path")
        dataset = get("train.dataset") if do_train else get("eval.dataset")
        config_path = get("train.config_path")
        if self.running:
            return ALERTS["err_conflict"][lang]

        if not model_name:
            return ALERTS["err_no_model"][lang]

        if not config_path:
            return ALERTS["err_no_cfg_path"][lang]

        if not model_path:
            return ALERTS["err_no_path"][lang]

        if not dataset:
            return ALERTS["err_no_dataset"][lang]

        if do_train:
            if not get("train.output_dir"):
                return ALERTS["err_no_output_dir"][lang]

            try:
                json.loads(get("train.extra_args"))
            except json.JSONDecodeError:
                return ALERTS["err_json_schema"][lang]

        else:
            if not get("eval.output_dir"):
                return ALERTS["err_no_output_dir"][lang]

        return ""

    def _finalize(self, lang: str, finish_info: str) -> str:
        finish_info = ALERTS["info_aborted"][lang] if self.aborted else finish_info
        gr.Info(finish_info)
        self.trainer = None
        self.aborted = False
        self.running = False
        self.running_data = None
        paddle.device.cuda.empty_cache()
        return finish_info

    def _parse_train_args(self, data: Dict["Component", Any]) -> Dict[str, Any]:
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        model_name, finetuning_type = get("top.model_name"), get("top.finetuning_type")
        args = dict(
            stage=TRAINING_STAGES[get("train.training_stage")],
            do_train=True,
            model_name_or_path=get("top.model_path"),
            cache_dir=None,
            preprocessing_num_workers=16,
            finetuning_type=finetuning_type,
            template=get("top.template"),
            dataset_dir=get("train.dataset_dir"),
            dataset=",".join(get("train.dataset")),
            cutoff_len=get("train.cutoff_len"),
            learning_rate=float(get("train.learning_rate")),
            num_train_epochs=float(get("train.num_train_epochs")),
            max_samples=int(get("train.max_samples")),
            per_device_train_batch_size=get("train.batch_size"),
            gradient_accumulation_steps=get("train.gradient_accumulation_steps"),
            lr_scheduler_type=get("train.lr_scheduler_type"),
            max_grad_norm=float(get("train.max_grad_norm")),
            logging_steps=get("train.logging_steps"),
            save_steps=get("train.save_steps"),
            warmup_steps=get("train.warmup_steps"),
            output_dir=get_save_dir(model_name, finetuning_type, get("train.output_dir")),
            fp16=(get("train.compute_type") == "fp16"),
            bf16=(get("train.compute_type") == "bf16"),
            pure_bf16=(get("train.compute_type") == "pure_bf16"),
            plot_loss=True,
            ddp_timeout=180000000,
            include_num_input_tokens_seen=False,
        )
        args.update(json.loads(get("train.extra_args")))

        # checkpoints
        if get("top.checkpoint_path"):
            if finetuning_type in PEFT_METHODS:  # list
                args["adapter_name_or_path"] = ",".join(
                    [get_save_dir(model_name, finetuning_type, adapter) for adapter in get("top.checkpoint_path")]
                )
            else:  # str
                args["model_name_or_path"] = get_save_dir(model_name, finetuning_type, get("top.checkpoint_path"))

        # lora config
        if args["finetuning_type"] == "lora":
            args["lora_rank"] = get("train.lora_rank")
            args["lora_alpha"] = get("train.lora_alpha")
            args["lora_dropout"] = get("train.lora_dropout")
            args["loraplus_lr_ratio"] = get("train.loraplus_lr_ratio")
            args["use_rslora"] = get("train.use_rslora")
            args["pissa_init"] = get("train.use_pissa")
            args["pissa_convert"] = get("train.use_pissa")

        # eval config
        if get("train.val_size") > 1e-6:
            args["val_size"] = get("train.val_size")
            args["eval_strategy"] = "steps"
            args["eval_steps"] = get("train.eval_steps")
            args["per_device_eval_batch_size"] = args["per_device_train_batch_size"]

        return args

    def _parse_eval_args(self, data: Dict["Component", Any]) -> Dict[str, Any]:
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        model_name, finetuning_type = get("top.model_name"), get("top.finetuning_type")

        args = dict(
            stage="sft",
            model_name_or_path=get("top.model_path"),
            cache_dir=None,
            preprocessing_num_workers=16,
            finetuning_type=finetuning_type,
            quantization_method=get("top.quantization_method"),
            template=get("top.template"),
            rope_scaling=get("top.rope_scaling") if get("top.rope_scaling") in ["linear", "dynamic"] else None,
            flash_attn="fa2" if get("top.booster") == "flashattn2" else "auto",
            use_unsloth=(get("top.booster") == "unsloth"),
            dataset_dir=get("eval.dataset_dir"),
            eval_dataset=",".join(get("eval.dataset")),
            cutoff_len=get("eval.cutoff_len"),
            max_samples=int(get("eval.max_samples")),
            per_device_eval_batch_size=get("eval.batch_size"),
            predict_with_generate=True,
            max_new_tokens=get("eval.max_new_tokens"),
            top_p=get("eval.top_p"),
            temperature=get("eval.temperature"),
            output_dir=get_save_dir(model_name, finetuning_type, get("eval.output_dir")),
        )

        if get("eval.predict"):
            args["do_predict"] = True
        else:
            args["do_eval"] = True

        # checkpoints
        if get("top.checkpoint_path"):
            if finetuning_type in PEFT_METHODS:  # list
                args["adapter_name_or_path"] = ",".join(
                    [get_save_dir(model_name, finetuning_type, adapter) for adapter in get("top.checkpoint_path")]
                )
            else:  # str
                args["model_name_or_path"] = get_save_dir(model_name, finetuning_type, get("top.checkpoint_path"))

        return args

    def _form_config_dict(self, data: Dict["Component", Any]) -> Dict[str, Any]:
        config_dict = {}
        skip_ids = ["top.lang", "top.model_path", "train.output_dir", "train.config_path"]
        for elem, value in data.items():
            elem_id = self.manager.get_id_by_elem(elem)
            if elem_id not in skip_ids:
                config_dict[elem_id] = value

        return config_dict

    def run_train_v2(self, data):
        # get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        self.running_data = data
        error = self._initialize(data, do_train=True, from_preview=True)
        if error != "":
            # lang = get("top.lang")
            output_box = self.manager.get_elem_by_id("{}.output_box".format("train"))
            progress_bar = self.manager.get_elem_by_id("{}.progress_bar".format("train"))
            yield {
                output_box: error,
                progress_bar: gr.Slider(visible=False),
            }
            return
        thread = Thread(target=self.run_train, args=(data,))
        thread.start()
        self.trainer = thread
        yield from self.monitor()

    def run_train(self, data):
        def check_aborted(self, trainer):
            while True:
                time.sleep(1)
                if self.aborted:
                    trainer.control.should_epoch_stop = True
                    trainer.control.should_training_stop = True

        callbacks = [LogCallback]
        # yield from self._launch(data, do_train=True)
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        model_path = get("top.model_path")
        model_name = get("top.model_name")
        checkpoint_path = get("top.checkpoint_path")
        finetuning_type = get("top.finetuning_type")

        tokenizer = MIXQwen2Tokenizer.from_pretrained(model_path)
        args = self._parse_train_args(data)
        model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

        # resume model
        resume_path = None
        if isinstance(checkpoint_path, str) and len(checkpoint_path) > 0:
            resume_root_path = get_save_dir(model_name, finetuning_type, checkpoint_path)
        else:
            os.makedirs(training_args.output_dir, exist_ok=True)
            resume_root_path = training_args.output_dir
        if resume_root_path is not None:
            ckpts = []
            for ckpt in os.listdir(resume_root_path):
                if "checkpoint" in ckpt:
                    ckpts.append(ckpt)
            ckpts.sort()
            if len(ckpts) > 0:
                resume_path = os.path.join(resume_root_path, ckpts[0])

        tokenizer.model_max_len = data_args.cutoff_len
        image_processor = Qwen2VLImageProcessor()
        processor = Qwen2VLProcessor(image_processor, tokenizer)
        template = get_template_and_fix_tokenizer(tokenizer, data_args)
        self.dataset = get_dataset(
            template, model_args, data_args, training_args, finetuning_args.stage, tokenizer, processor
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, dtype=model_args.compute_dtype)
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,  # for shift short attention
            label_pad_token_id=IGNORE_INDEX,
        )
        # ALERTS
        if finetuning_type == "lora":
            target = [
                "model.layers.*q_proj.*",
                "model.layers.*k_proj.*",
                "model.layers.*v_proj.*",
                "model.layers.*gate_proj.*",
                "model.layers.*up_proj.*",
                "model.layers.*down_proj.*",
                "model.layers.*o_proj.*",
            ]
            lora_cfg = LoRAConfig(
                target_modules=target,
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                merge_weights=False,
                dtype=model_args.compute_dtype,  # using str type for dtype while saving bfloat16
                rslora=finetuning_args.use_rslora,
                lora_plus_scale=finetuning_args.loraplus_lr_ratio,
                pissa=finetuning_args.pissa_init,
                tensor_parallel_degree=training_args.per_device_train_batch_size,
            )
            model = LoRAModel(model, lora_cfg)
            model.mark_only_lora_as_trainable()
            model.print_trainable_parameters()
        # len(list(filter(lambda p:p.stop_gradient,model.parameters())))
        # trainer = Trainer( model=model, args=training_args, train_dataset=self.dataset['train_dataset'], eval_dataset=None, tokenizer=tokenizer, data_collator=data_collator, callbacks=callbacks )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.dataset["train_dataset"],
            eval_dataset=self.dataset.get("eval_dataset", None),
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        # trainer.control.should_evaluate = True
        if self.dataset.get("eval_dataset") is not None:
            training_args.evaluation_strategy = IntervalStrategy.STEPS

        daemon_thread = Thread(target=check_aborted, args=(self, trainer))
        daemon_thread.daemon = True
        daemon_thread.start()
        train_result = trainer.train(resume_path)  # image token batching存在问题，无法打包成patch
        if train_result.global_step == trainer.state.max_steps:
            trainer.save_model()
            trainer.save_state()

    def monitor(self):
        self.aborted = False
        self.running = True

        get = lambda elem_id: self.running_data[self.manager.get_elem_by_id(elem_id)]
        lang, model_name, finetuning_type = get("top.lang"), get("top.model_name"), get("top.finetuning_type")
        output_dir = get("{}.output_dir".format("train"))
        output_path = get_save_dir(model_name, finetuning_type, output_dir)

        output_box = self.manager.get_elem_by_id("{}.output_box".format("train"))
        progress_bar = self.manager.get_elem_by_id("{}.progress_bar".format("train"))
        loss_viewer = self.manager.get_elem_by_id("train.loss_viewer") if self.do_train else None

        running_log = ""
        while self.trainer is not None:
            if self.aborted:
                yield {
                    output_box: ALERTS["info_aborting"][lang],
                    progress_bar: gr.Slider(visible=False),
                }
            else:
                running_log, running_progress, running_loss = get_trainer_info(output_path, self.do_train)
                return_dict = {
                    output_box: running_log,
                    progress_bar: running_progress,
                }
                if running_loss is not None:
                    return_dict[loss_viewer] = running_loss

                yield return_dict
            if self.trainer.is_alive():
                yield {
                    output_box: ALERTS["info_training"][lang],
                }
                time.sleep(5)
            else:
                yield {
                    output_box: ALERTS["info_aborting"][lang],
                    progress_bar: gr.Slider(visible=False),
                }
                self.trainer = None
                self.running = False
                print("trainer exited")

        if self.do_train:
            if os.path.exists(os.path.join(output_path, TRAINING_ARGS_NAME)):
                finish_info = ALERTS["info_finished"][lang]
            else:
                finish_info = ALERTS["err_failed"][lang]
        else:
            if os.path.exists(os.path.join(output_path, "all_results.json")):
                finish_info = get_eval_results(os.path.join(output_path, "all_results.json"))
            else:
                finish_info = ALERTS["err_failed"][lang]

        return_dict = {
            output_box: self._finalize(lang, finish_info) + "\n\n" + running_log,
            progress_bar: gr.Slider(visible=False),
        }
        yield return_dict

    def save_args(self, data):
        output_box = self.manager.get_elem_by_id("train.output_box")
        error = self._initialize(data, do_train=True, from_preview=True)
        if error:
            gr.Warning(error)
            return {output_box: error}

        lang = data[self.manager.get_elem_by_id("top.lang")]
        config_path = data[self.manager.get_elem_by_id("train.config_path")]
        os.makedirs(DEFAULT_CONFIG_DIR, exist_ok=True)
        save_path = os.path.join(DEFAULT_CONFIG_DIR, config_path)

        save_args(save_path, self._form_config_dict(data))
        return {output_box: ALERTS["info_config_saved"][lang] + save_path}

    def load_args(self, lang: str, config_path: str):
        output_box = self.manager.get_elem_by_id("train.output_box")
        config_dict = load_args(os.path.join(DEFAULT_CONFIG_DIR, config_path))
        if config_dict is None:
            gr.Warning(ALERTS["err_config_not_found"][lang])
            return {output_box: ALERTS["err_config_not_found"][lang]}

        output_dict: Dict["Component", Any] = {output_box: ALERTS["info_config_loaded"][lang]}
        for elem_id, value in config_dict.items():
            output_dict[self.manager.get_elem_by_id(elem_id)] = value

        return output_dict

    def check_output_dir(self, lang: str, model_name: str, finetuning_type: str, output_dir: str):
        if model_name and output_dir and os.path.isdir(get_save_dir(model_name, finetuning_type, output_dir)):
            gr.Warning(ALERTS["warn_output_dir_exists"][lang])
        return ALERTS["warn_output_dir_exists"][lang]

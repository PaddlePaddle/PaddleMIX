import subprocess
from typing import List

import paddlenlp

from .evaluation import run_benchmark_jobs
from .hub import push_to_hub_revision


def is_slurm_available() -> bool:
    try:
        subprocess.run(
            ["sinfo"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return True
    except FileNotFoundError:
        return False


class DummyConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


>>>>>>class PushToHubRevisionCallback(transformers.TrainerCallback):
    def __init__(self, model_config) -> None:
        self.model_config = model_config

    def on_save(
        self,
>>>>>>        args: transformers.training_args.TrainingArguments,
>>>>>>        state: transformers.trainer_callback.TrainerState,
>>>>>>        control: transformers.trainer_callback.TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            global_step = state.global_step
            dummy_config = DummyConfig(
                hub_model_id=args.hub_model_id,
                hub_model_revision=f"{args.hub_model_revision}-step-{global_step:09d}",
                output_dir=f"{args.output_dir}/checkpoint-{global_step}",
                system_prompt=args.system_prompt,
            )
            future = push_to_hub_revision(dummy_config, extra_ignore_patterns=["*.pt"])
            if is_slurm_available():
                dummy_config.benchmarks = args.benchmarks

                def run_benchmark_callback(_):
                    print(f"Checkpoint {global_step} pushed to hub.")
                    run_benchmark_jobs(dummy_config, self.model_config)

                future.add_done_callback(run_benchmark_callback)


CALLBACKS = {"push_to_hub_revision": PushToHubRevisionCallback}


>>>>>>def get_callbacks(train_config, model_config) -> List[transformers.TrainerCallback]:
    callbacks = []
    for callback_name in train_config.callbacks:
        if callback_name not in CALLBACKS:
            raise ValueError(f"Callback {callback_name} not found in CALLBACKS.")
        callbacks.append(CALLBACKS[callback_name](model_config))
    return callbacks

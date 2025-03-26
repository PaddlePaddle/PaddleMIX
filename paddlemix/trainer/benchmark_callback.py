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

from paddlenlp.trainer.trainer import TrainerCallback
from paddlenlp.trainer.trainer_utils import ShardingOption
from paddlenlp.utils.log import logger


"""utils for log"""
import os
import time
import numpy as np
import paddle
import paddle.distributed.fleet as fleet
from paddlenlp.trainer.plugins.timer import get_timers

def get_memory_info():
    """get_memory_info"""
    divisor = 2**30
    return (
        paddle.device.cuda.memory_allocated() / divisor,
        paddle.device.cuda.max_memory_allocated() / divisor,
        paddle.device.cuda.memory_reserved() / divisor,
        paddle.device.cuda.max_memory_reserved() / divisor,
    )


class Statistical(object):
    """Statistical
    """
    def __init__(self, buffer_size, skip_step, approximate=False):
        self.step = 0
        self.local_step = 0
        self.is_first_period = True
        print('buffer size is %d and skip step is %d.'%(buffer_size, skip_step))

        assert skip_step < buffer_size
        self.buffer_size = buffer_size
        self.skip_step = skip_step
        self.approximate = approximate

        self.local_tokens = np.zeros([buffer_size], dtype=np.int64)
        self.local_samples = np.zeros([buffer_size], dtype=np.int64)
        self.efficient_tokens = np.zeros([buffer_size], dtype=np.int64)
        self.tokens = np.zeros([buffer_size], dtype=np.int64)
        self.samples = np.zeros([buffer_size], dtype=np.int64)
        self.durations = np.zeros([buffer_size], dtype=np.float64)
        if int(os.getenv("PADDLE_TRAINERS_NUM", "1")) > 1:
            assert (
                paddle.distributed.is_initialized()
            ), "please call fleet.init() before"
            if hasattr(fleet.fleet, "_hcg"):
                hcg = fleet.get_hybrid_communicate_group()
    
                dp_size = hcg.get_data_parallel_world_size() 
                sharding_size = hcg.get_sharding_parallel_world_size()
                self.dp_size = dp_size * sharding_size 
                self.world_size = paddle.distributed.get_world_size()
                self.groups = []
                if dp_size > 1:
                    self.groups.append(hcg.get_data_parallel_group())
                if sharding_size > 1:
                    self.groups.append(hcg.get_sharding_parallel_group())
            else:
                self.dp_size = paddle.distributed.get_world_size()
                self.world_size = self.dp_size
                self.groups = []

        else:
            self.dp_size = 1
            self.world_size = 1
            self.groups = []

    def reset(self):
        """reset
        """
        if self.local_step > 0:
            self.is_first_period = False
        self.local_step = 0
        self.local_tokens.fill(0)
        self.local_samples.fill(0)
        self.efficient_tokens.fill(0)
        self.tokens.fill(0)
        self.samples.fill(0)
        self.durations.fill(0.0)

    def _get_global(self, tokens_num, batch_size):
        if self.dp_size == 1 or self.approximate:
            return tokens_num * self.dp_size, batch_size * self.dp_size

        timers = get_timers()
        if timers:
            timers("all-reduce-token-bz").start()
        x = paddle.to_tensor([tokens_num, batch_size], dtype=paddle.int64)
        for g in self.groups:  
            paddle.distributed.stream.all_reduce(x, group=g, sync_op=True, use_calc_stream=True)
        if timers:
            timers("all-reduce-token-bz").stop()
        return x.numpy().tolist() 

    def add(self, start_time, efficient_tokens_num, tokens_num, batch_size):
        """add """
        assert (
            self.local_step < self.buffer_size
        ), "the step number exceeds the ckpt saving interval"
        global_tokens_num, global_batch_size = self._get_global(tokens_num, batch_size)
        global_efficient_tokens_num, _ = self._get_global(efficient_tokens_num, batch_size)

        duration = time.time() - start_time
        self.durations[self.local_step] = duration
        self.local_tokens[self.local_step] = tokens_num
        self.local_samples[self.local_step] = batch_size
        self.efficient_tokens[self.local_step] = global_efficient_tokens_num
        self.tokens[self.local_step] = global_tokens_num
        self.samples[self.local_step] = global_batch_size
        self.step += 1
        self.local_step += 1
        return global_tokens_num, global_batch_size

    def get_tokens_per_sec_per_card(self):
        """get_tokens_per_sec_per_card"""
        return (
            self.tokens[self.local_step - 1]
            / self.durations[self.local_step - 1]
            / self.world_size
        )
        
    def get_efficient_tokens_per_sec_per_card(self):
        """get_efficient_tokens_per_sec_per_card"""
        return (
            self.efficient_tokens[self.local_step - 1]
            / self.durations[self.local_step - 1]
            / self.world_size
        )

    def get_avg_tokens_per_sec_per_card(self):
        """get_avg_tokens_per_sec_per_card"""
        if self.step <= self.skip_step:
            return self.get_tokens_per_sec_per_card()

        start = self.skip_step if self.is_first_period else 0
        return (
            np.sum(self.tokens[start : self.local_step])
            / np.sum(self.durations[start : self.local_step])
            / self.world_size
        )
    
    def get_avg_efficient_tokens_per_sec_per_card(self):
        """get_avg_tokens_per_sec_per_card"""
        if self.step <= self.skip_step:
            return self.get_efficient_tokens_per_sec_per_card()

        start = self.skip_step if self.is_first_period else 0
        return (
            np.sum(self.efficient_tokens[start : self.local_step])
            / np.sum(self.durations[start : self.local_step])
            / self.world_size
        )

    def get_samples_per_sec_per_card(self):
        """get_samples_per_sec_per_card"""
        return (
            self.samples[self.local_step - 1]
            / self.durations[self.local_step - 1]
            / self.world_size
        )

    def get_avg_samples_per_sec_per_card(self):
        """get_avg_samples_per_sec_per_card"""
        if self.step <= self.skip_step:
            return self.get_samples_per_sec_per_card()

        start = self.skip_step if self.is_first_period else 0
        return (
            np.sum(self.samples[start : self.local_step])
            / np.sum(self.durations[start : self.local_step])
            / self.world_size
        )

    def get_tokens(self):
        """get_tokens"""
        return self.tokens[: self.local_step]

    def get_efficient_tokens(self):
        """get_efficient_tokens"""
        return self.efficient_tokens[: self.local_step]

    def get_samples(self):
        """get_samples"""
        return self.samples[: self.local_step]

    def get_durations(self):
        """get_durations"""
        return self.durations[: self.local_step]

    def get_total_tokens_per_card(self):
        """get_total_tokens_per_card"""
        return np.sum(self.tokens[: self.local_step]) / self.world_size
    
    def get_total_efficient_tokens_per_card(self):
        """get_total_efficient_tokens_per_card"""
        return np.sum(self.efficient_tokens[: self.local_step]) / self.world_size

    def get_total_samples_per_card(self):
        """get_total_samples_per_card"""
        return np.sum(self.samples[: self.local_step]) / self.world_size

    def get_skip_duration(self):
        """get_skip_duration"""
        if self.is_first_period:
            return np.sum(self.durations[: min(self.local_step, self.skip_step)])
        else:
            return 0.0


    def get_result(self):
        """get_result"""
        runtime = np.sum(self.durations[: self.local_step])
        local_samples = np.sum(self.local_samples[: self.local_step])
        local_tokens = np.sum(self.local_tokens[: self.local_step])
        global_samples = np.sum(self.samples[: self.local_step])
        global_tokens = np.sum(self.tokens[: self.local_step])
        global_efficient_tokens = np.sum(self.efficient_tokens[: self.local_step])

        alloc, max_alloc, reserved, max_reserved = get_memory_info()
        result = {
            f'runtime': runtime,
            f"local_samples": local_samples,
            f"global_samples": global_samples,
            f"local_tokens": local_tokens,
            f"global_tokens": global_tokens,
            f"global_efficient_tokens": global_efficient_tokens,
            f"samples_per_sec_per_card": self.get_samples_per_sec_per_card(),
            f"avg_samples_per_sec_per_card": self.get_avg_samples_per_sec_per_card(),
            f"efficient_tokens_per_sec_per_card": self.get_efficient_tokens_per_sec_per_card(),
            f"tokens_per_sec_per_card": self.get_tokens_per_sec_per_card(),
            f"avg_tokens_per_sec_per_card": self.get_avg_tokens_per_sec_per_card(),
            f"avg_efficient_tokens_per_sec_per_card": self.get_avg_efficient_tokens_per_sec_per_card(),
            "memory_allocated_gb": alloc,
            "max_memory_allocated_gb": max_alloc,
            "memory_reserved_gb": reserved,
            "max_memory_reserved_gb": max_reserved,
        }
        
        return result


class BenchmarkCallback(TrainerCallback):
    """
    used to benchmark the training process.
    """

    ACC_SAMPLES = "acc_global_samples"
    ACC_TOKENS = "acc_global_tokens"

    def __init__(self, trainer, save_steps, skip_step, benchmark_mode):
        super().__init__()
        self.trainer = trainer
        self.state = Statistical(save_steps, skip_step)
        self.efficient_token_count = 0
        self.cur_tokens = 0
        self.cur_samples = 0

    def set_save_time(self, save_time):
        """
        set the time to save model.
        """
        self.save_time = save_time

    def on_train_begin(self, args, state, control, **kwargs):
        """
        record the start time of training.
        """
        if state.trial_params is None:
            state.trial_params = {}

        if self.ACC_SAMPLES not in state.trial_params:
            state.trial_params[self.ACC_SAMPLES] = 0
        if self.ACC_TOKENS not in state.trial_params:
            state.trial_params[self.ACC_TOKENS] = 0

        if paddle.distributed.is_initialized():
            if hasattr(fleet.fleet, "_hcg"):
                dp_degree = fleet.get_hybrid_communicate_group().get_data_parallel_world_size()
            else:
                dp_degree = paddle.distributed.get_world_size()
            assert dp_degree <= 1, f"data_parallel_degree should be 1 but got {dp_degree}"
            paddle.distributed.barrier()
        self.end_save_time = time.time()

    def on_epoch_begin(self, args, state, control, **kwargs):
        """
        record the start time of epoch.
        """
        self.epoch_start = time.time()
        self.batch_start = time.time()

    def on_substep_end(self, args, state, control, **kwargs):
        """
        record the start time of batch.
        """
        model = kwargs["model"]
        batch_size, seq_length, _ = model.input_shape
        self.efficient_token_count = model.efficient_token_count
        self.cur_tokens += batch_size * seq_length
        self.cur_samples += batch_size

    def on_step_end(self, args, state, control, **kwargs):
        """
        record the start time of sub-batch.
        """
        self.on_substep_end(args, state, control, **kwargs)

        tokens, batches = self.state.add(
            start_time=self.batch_start, efficient_tokens_num=self.efficient_token_count, tokens_num=self.cur_tokens, batch_size=self.cur_samples
        )
        state.trial_params[self.ACC_SAMPLES] += batches
        state.trial_params[self.ACC_TOKENS] += tokens

        self.efficient_token_count = 0
        self.cur_tokens = 0
        self.cur_samples = 0

        self.batch_start = time.time()
        if control.should_log:
            self.maybe_log_save_evaluate_start = time.time()

    def on_save(self, args, state, control, **kwargs):
        """
        record the infomation of saving model.
        """
        pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        record the infomation of a logging step.
        """
        if benchmark_mode:
            logs.update(self.state.get_result())
            logs[self.ACC_SAMPLES] = state.trial_params[self.ACC_SAMPLES]
            logs[self.ACC_TOKENS] = state.trial_params[self.ACC_TOKENS]

            max_mem_reserved_msg = (
                f"max_mem_reserved: {logs['max_memory_reserved_gb']} GB,"
            )
            max_mem_allocated_msg = (
                f"max_mem_allocated: {logs['max_memory_allocated_gb']} GB"
            )
            
            logger.info(
                "global step %d, loss: %.5f, interval_samples_per_second: %.5f, ips: %.5f, %s %s"
                % (
                    state.global_step,
                    logs["loss"],
                    logs["interval_samples_per_second"],
                    logs["avg_efficient_tokens_per_sec_per_card"],
                    max_mem_reserved_msg,
                    max_mem_allocated_msg,
                )
            )

    def _log(self, logs):
        """
        record the information accurately and neatly.
        """
        logs_str = []
        logs_str = []
        for k, v in logs.items():
            if isinstance(v, float):
                if abs(v) < 1e-3:
                    v = f"{v:e}"
                elif abs(v) > 100:
                    v = f"{v:.04f}"
                else:
                    v = f"{v:.06f}"
            logs_str.append(f"{k}: {v}")
        logger.info(", ".join(logs_str))
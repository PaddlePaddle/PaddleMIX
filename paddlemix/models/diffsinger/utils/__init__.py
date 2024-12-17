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

from __future__ import annotations

import pathlib
import re
import time
import types
from collections import OrderedDict

import numpy as np
import paddle

from paddlemix.models.diffsinger.basics.base_module import CategorizedModule
from paddlemix.models.diffsinger.utils import paddle_aux
from paddlemix.models.diffsinger.utils.hparams import hparams

def tensors_to_scalars(metrics):
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, paddle.Tensor):
            v = v.item()
        if type(v) is dict:
            v = tensors_to_scalars(v)
        new_metrics[k] = v
    return new_metrics


def collate_nd(values, pad_value=0, max_len=None):
    """
    Pad a list of Nd tensors on their first dimension and stack them into a (N+1)d tensor.
    """
    size = max(v.shape[0] for v in values) if max_len is None else max_len, *tuple(values[0].shape)[1:]
    res = paddle.full(shape=(len(values), *size), fill_value=pad_value, dtype=values[0].dtype)
    for i, v in enumerate(values):
        res[i, : len(v), ...] = v
    return res


def random_continuous_masks(*shape: int, dim: int, device: (str | (paddle.CPUPlace, paddle.CUDAPlace, str)) = "cpu"):  # type: ignore
    start, end = (
        paddle.sort(
            x=paddle.randint(
                low=0, high=shape[dim] + 1, shape=(*shape[:dim], 2, *((1,) * (len(shape) - dim - 1)))
            ).expand(shape=[*((-1,) * (dim + 1)), *shape[dim + 1 :]]),
            axis=dim,
        ),
        paddle.argsort(
            x=paddle.randint(
                low=0, high=shape[dim] + 1, shape=(*shape[:dim], 2, *((1,) * (len(shape) - dim - 1)))
            ).expand(shape=[*((-1,) * (dim + 1)), *shape[dim + 1 :]]),
            axis=dim,
        ),
    )[0].split(1, dim=dim)
    idx = paddle.arange(start=0, end=shape[dim], dtype="int64").reshape(
        *((1,) * dim), shape[dim], *((1,) * (len(shape) - dim - 1))
    )
    masks = (idx >= start) & (idx < end)
    return masks


def _is_batch_full(batch, num_frames, max_batch_frames, max_batch_size):
    if len(batch) == 0:
        return 0
    if len(batch) == max_batch_size:
        return 1
    if num_frames > max_batch_frames:
        return 1
    return 0


def batch_by_size(indices, num_frames_fn, max_batch_frames=80000, max_batch_size=48, required_batch_size_multiple=1):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_frames_fn (callable): function that returns the number of frames at
            a given index
        max_batch_frames (int, optional): max number of frames in each batch
            (default: 80000).
        max_batch_size (int, optional): max number of sentences in each
            batch (default: 48).
        required_batch_size_multiple: require the batch size to be multiple
            of a given number
    """
    bsz_mult = required_batch_size_multiple
    if isinstance(indices, types.GeneratorType):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)
    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_frames = num_frames_fn(idx)
        sample_lens.append(num_frames)
        sample_len = max(sample_len, num_frames)
        assert (
            sample_len <= max_batch_frames
        ), "sentence at index {} of size {} exceeds max_batch_samples limit of {}!".format(
            idx, sample_len, max_batch_frames
        )
        num_frames = (len(batch) + 1) * sample_len
        if _is_batch_full(batch, num_frames, max_batch_frames, max_batch_size):
            mod_len = max(bsz_mult * (len(batch) // bsz_mult), len(batch) % bsz_mult)
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    mask = tensor.not_equal(y=paddle.to_tensor(padding_idx)).astype(dtype="int32")
    return (paddle.cumsum(x=mask, axis=1).astype(dtype=mask.dtype) * mask).astype(dtype="int64") + padding_idx


def softmax(x, dim):
    return paddle.nn.functional.softmax(x=x, axis=dim, dtype="float32")


def unpack_dict_to_list(samples):
    samples_ = []
    bsz = samples.get("outputs").shape[0]
    for i in range(bsz):
        res = {}
        for k, v in samples.items():
            try:
                res[k] = v[i]
            except:
                pass
        samples_.append(res)
    return samples_


def filter_kwargs(dict_to_filter, kwarg_obj):
    import inspect

    sig = inspect.signature(kwarg_obj)
    if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
        return dict_to_filter.copy()
    filter_keys = [
        param.name
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD or param.kind == param.KEYWORD_ONLY
    ]
    filtered_dict = {
        filter_key: dict_to_filter[filter_key] for filter_key in filter_keys if filter_key in dict_to_filter
    }
    return filtered_dict


def load_ckpt(
    cur_model,
    ckpt_base_dir,
    ckpt_steps=None,
    prefix_in_ckpt="model",
    ignored_prefixes=None,
    key_in_ckpt="state_dict",
    strict=True,
    device="cpu",
):
    if ignored_prefixes is None:
        ignored_prefixes = ["model.fs2.encoder.embed_tokens"]
    if not isinstance(ckpt_base_dir, pathlib.Path):
        ckpt_base_dir = pathlib.Path(ckpt_base_dir)
    if ckpt_base_dir.is_file():
        checkpoint_path = [ckpt_base_dir]
    elif ckpt_steps is not None:
        checkpoint_path = [ckpt_base_dir / f"model_ckpt_steps_{int(ckpt_steps)}.ckpt"]
    else:
        base_dir = ckpt_base_dir
        checkpoint_path = sorted(
            [
                ckpt_file
                for ckpt_file in base_dir.iterdir()
                if ckpt_file.is_file() and re.fullmatch("model_ckpt_steps_\\d+\\.ckpt", ckpt_file.name)
            ],
            key=lambda x: int(re.search("\\d+", x.name).group(0)),
        )
    assert len(checkpoint_path) > 0, f"| ckpt not found in {ckpt_base_dir}."
    checkpoint_path = checkpoint_path[-1]
    ckpt_loaded = paddle.load(path=str(checkpoint_path))
    if isinstance(cur_model, CategorizedModule):
        cur_model.check_category(ckpt_loaded.get("category"))
    if key_in_ckpt is None:
        state_dict = ckpt_loaded
    else:
        state_dict = ckpt_loaded[key_in_ckpt]
    if prefix_in_ckpt is not None:
        state_dict = OrderedDict(
            {
                k[len(prefix_in_ckpt) + 1 :]: v
                for k, v in state_dict.items()
                if k.startswith(f"{prefix_in_ckpt}.")
                if all(not k.startswith(p) for p in ignored_prefixes)
            }
        )
    if not strict:
        cur_model_state_dict = cur_model.state_dict()
        unmatched_keys = []
        for key, param in state_dict.items():
            if key in cur_model_state_dict:
                new_param = cur_model_state_dict[key]
                if tuple(new_param.shape) != tuple(param.shape):
                    unmatched_keys.append(key)
                    print("| Unmatched keys: ", key, tuple(new_param.shape), tuple(param.shape))
        for key in unmatched_keys:
            del state_dict[key]
    cur_model.set_state_dict(state_dict=state_dict)
    shown_model_name = "state dict"
    if prefix_in_ckpt is not None:
        shown_model_name = f"'{prefix_in_ckpt}'"
    elif key_in_ckpt is not None:
        shown_model_name = f"'{key_in_ckpt}'"
    print(f"| load {shown_model_name} from '{checkpoint_path}'.")


def remove_padding(x, padding_idx=0):
    if x is None:
        return None
    assert len(tuple(x.shape)) in [1, 2]
    if len(tuple(x.shape)) == 2:
        return x[np.abs(x).sum(-1) != padding_idx]
    elif len(tuple(x.shape)) == 1:
        return x[x != padding_idx]


class Timer:
    timer_map = {}

    def __init__(self, name, print_time=False):
        if name not in Timer.timer_map:
            Timer.timer_map[name] = 0
        self.name = name
        self.print_time = print_time

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        Timer.timer_map[self.name] += time.time() - self.t
        if self.print_time:
            print(self.name, Timer.timer_map[self.name])


def print_arch(model, model_name="model"):
    print(f"| {model_name} Arch: ", model)


def num_params(model, print_out=True, model_name="model"):
    parameters = filter(lambda p: not p.stop_gradient, model.parameters())
    parameters = sum([np.prod(tuple(p.shape)) for p in parameters]) / 1000000
    if print_out:
        print(f"| {model_name} Trainable Parameters: %.3fM" % parameters)
    return parameters


def build_object_from_class_name(cls_str, parent_cls, *args, **kwargs):
    import importlib

    pkg = ".".join(cls_str.split(".")[:-1])
    cls_name = cls_str.split(".")[-1]
    cls_type = getattr(importlib.import_module(pkg), cls_name)
    if parent_cls is not None:
        assert issubclass(cls_type, parent_cls), f"| {cls_type} is not subclass of {parent_cls}."
    return cls_type(*args, **filter_kwargs(kwargs, cls_type))


def build_lr_scheduler_from_config(optimizer, scheduler_args):
    # try:
    # except ImportError:
    from paddle.optimizer.lr import LRScheduler as LRScheduler

    def helper(params):
        if isinstance(params, list):
            return [helper(s) for s in params]
        elif isinstance(params, dict):
            resolved = {k: helper(v) for k, v in params.items()}
            if "cls" in resolved:
                if (
                    resolved["cls"] == "torch.optim.lr_scheduler.ChainedScheduler"
                    and scheduler_args["scheduler_cls"] == "torch.optim.lr_scheduler.SequentialLR"
                ):
                    raise ValueError(f"ChainedScheduler cannot be part of a SequentialLR.")
                resolved["optimizer"] = optimizer
                obj = build_object_from_class_name(resolved["cls"], LRScheduler, **resolved)
                return obj
            return resolved
        else:
            return params

    resolved = helper(scheduler_args)
    resolved["optimizer"] = optimizer
    return build_object_from_class_name(scheduler_args["scheduler_cls"], LRScheduler, **resolved)


def simulate_lr_scheduler(optimizer_args, scheduler_args, step_count, num_param_groups=1):
    optimizer = build_object_from_class_name(
        optimizer_args["optimizer_cls"],
        paddle.optimizer.Optimizer,
        [
            {
                "params": paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.to_tensor([])),
                "initial_lr": optimizer_args["lr"],
            }
            for _ in range(num_param_groups)
        ],
        **optimizer_args,
    )
    scheduler = build_lr_scheduler_from_config(optimizer, scheduler_args)
    scheduler.optimizer._step_count = 1
    for _ in range(step_count):
        scheduler.step()
    return scheduler.state_dict()


def remove_suffix(string: str, suffix: str):
    if string.endswith(suffix):
        string = string[: -len(suffix)]
    return string

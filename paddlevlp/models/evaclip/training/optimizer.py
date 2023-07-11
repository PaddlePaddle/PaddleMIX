# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict, abc as container_abcs
from copy import deepcopy
from itertools import chain
import warnings
import paddle


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class Optimizer(object):
    def __init__(self, params, defaults):

        if isinstance(params, paddle.Tensor):
            raise TypeError(
                "`parameters` argument given to the optimizer should be "
                "an iterable of paddle Tensors, but got argument type is `{}`.".
                format(type(params)))

        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def add_param_group(self, param_group):
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, paddle.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError(
                'optimizer parameters need to be organized in ordered collections, but '
                'the ordering of tensors in sets will change between runs. Please use a list instead.'
            )
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, paddle.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + format(
                                    type(params)))
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    "parameter group didn't specify a value of required optimization parameter "
                    + name)
            else:
                if name == 'lr':
                    param_group.setdefault(name, deepcopy(default))
                else:
                    param_group.setdefault(name, default)
        params = param_group['params']
        if len(params) != len(set(params)):
            warnings.warn(
                "optimizer contains a parameter group with duplicate parameters",
                stacklevel=3)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError(
                "some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

    @property
    def _param_groups(self):
        return self.param_groups

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    @staticmethod
    def _get_lr(param_group):
        lr_t = param_group["lr"]
        if isinstance(lr_t, paddle.optimizer.lr.LRScheduler):
            lr_t = lr_t.get_lr()
        if 'lr_scale' in param_group:
            lr_t *= param_group['lr_scale']
        return lr_t

    def state_dict(self):
        def pack_group(group):
            packed = {k: v for k, v in group.items() if k != 'params'}
            packed['params'] = [p.name for p in group['params']]

            return packed

        param_groups = [pack_group(g) for g in self.param_groups]

        state = {k: v for k, v in self.state.items()}

        return {
            'state': state,
            'param_groups': param_groups,
        }

    def set_state_dict(self, state_dict):
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "loaded state dict contains a parameter group "
                "that doesn't match the size of optimizer's group")

        # Update the state
        param_name_map = {
            name: p
            for name, p in zip(
                chain.from_iterable((g['params'] for g in saved_groups)),
                chain.from_iterable((g['params'] for g in groups)))
        }

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, paddle.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                if paddle.is_floating_point(
                        param) and param.dtype != value.dtype:
                    value = paddle.cast(value, param.dtype)
                value.value().get_tensor().set(
                    value.numpy(),
                    paddle.fluid.framework._current_expected_place())
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in param_name_map:
                param = param_name_map[k]
                # TODO(GuoxiaWang): fix type cast
                # state[k] = cast(param, v)
                state[k] = v
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group

        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)
        ]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    @paddle.no_grad()
    def clear_grad(self, set_to_zero=True):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.clear_gradient(set_to_zero)

    @paddle.no_grad()
    def lr_step(self, step=None):
        for group in self.param_groups:
            lr = group['lr']
            if isinstance(lr, paddle.optimizer.lr.LRScheduler):
                lr.step(step)
            elif 'lr_func' in group and callable(group['lr_func']):
                group['lr_func'](group, step)

    @paddle.no_grad()
    def get_lr(self, group_id=0):
        lr = self.param_groups[group_id]['lr']
        if isinstance(lr, paddle.optimizer.lr.LRScheduler):
            lr = lr.get_lr()
        return lr

    @paddle.no_grad()
    def step(self):
        raise NotImplementedError

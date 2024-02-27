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

import paddle


class EasyDict:
    def __init__(self, sub_dict):
        for k, v in sub_dict.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return paddle.mean(x, axis=list(range(1, len(x.shape))))


def log_state(state):
    result = []

    sorted_state = dict(sorted(state.items()))
    for key, value in sorted_state.items():
        # Check if the value is an instance of a class
        if "<object" in str(value) or "object at" in str(value):
            result.append(f"{key}: [{value.__class__.__name__}]")
        else:
            result.append(f"{key}: {value}")
    return "\n".join(result)


def none_or_str(value):
    if value == "None":
        return None
    return value


def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)


def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument(
        "--sampling-method",
        type=str,
        default="dopri5",
        help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq",
    )
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")


def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
    group.add_argument(
        "--diffusion-form",
        type=str,
        default="sigma",
        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],
        help="form of diffusion coefficient in the SDE",
    )
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument(
        "--last-step",
        type=none_or_str,
        default="Mean",
        choices=[None, "Mean", "Tweedie", "Euler"],
        help="form of last step taken in the SDE",
    )
    group.add_argument("--last-step-size", type=float, default=0.04, help="size of the last step taken")

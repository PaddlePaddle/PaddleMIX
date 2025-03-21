# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest
from argparse import ArgumentTypeError


def strtobool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def get_bool_from_env(key, default_value=False):
    if key not in os.environ:
        return default_value
    value = os.getenv(key)
    try:
        value = strtobool(value)
    except ValueError:
        raise ValueError(f"If set, {key} must be yes, no, true, false, 0 or 1 (case insensitive).")
    return value


_run_slow_test = get_bool_from_env("RUN_SLOW_TEST")


def slow(test):
    """
    Mark a test which spends too much time.
    Slow tests are skipped by default. Execute the command `export RUN_SLOW_TEST=True` to run them.
    """
    if not _run_slow_test:
        return unittest.skip("test spends too much time")(test)
    else:
        return test

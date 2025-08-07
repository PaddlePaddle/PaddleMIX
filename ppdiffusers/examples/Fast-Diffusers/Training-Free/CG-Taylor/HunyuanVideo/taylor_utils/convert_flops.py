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

import re


def convert_flops(flops_str):
    """
    Convert a FLOPS string (e.g., '12.34 GFLOPS', '1.2 TFLOPS') into the corresponding numerical value.
    """
    # Use regular expressions to match numbers and units
    match = re.match(r"([\d.]+)\s*([GT]?FLOPS)", flops_str.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Unable to parse FLOPS string: {flops_str}")

    # Extract the numeric value and unit
    value = float(match.group(1))
    unit = match.group(2).upper()

    # Convert based on the unit
    if unit == "GFLOPS":
        return value * 10**9
    elif unit == "TFLOPS":
        return value * 10**12
    else:
        raise ValueError(f"Unknown FLOPS unit: {unit}")

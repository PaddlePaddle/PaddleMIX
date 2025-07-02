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

# =======================================================
NOISE_SCHEDULES = {
    "linear",
    "scaled_linear",
    "squaredcos_cap_v2",
}

PREDICT_TYPE = {
    "epsilon",
    "sample",
    "v_prediction",
}

# =======================================================
NEGATIVE_PROMPT = "错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺，"

# =======================================================
TRT_MAX_BATCH_SIZE = 1
TRT_MAX_WIDTH = 1280
TRT_MAX_HEIGHT = 1280

# =======================================================
# Constants about models
# =======================================================

SAMPLER_FACTORY = {
    "ddpm": {
        "scheduler": "DDPMScheduler",
        "name": "DDPM",
        "kwargs": {
            "steps_offset": 1,
            "clip_sample": False,
            "clip_sample_range": 1.0,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "beta_end": 0.03,
            "prediction_type": "v_prediction",
        },
    },
    "ddim": {
        "scheduler": "DDIMScheduler",
        "name": "DDIM",
        "kwargs": {
            "steps_offset": 1,
            "clip_sample": False,
            "clip_sample_range": 1.0,
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "beta_end": 0.03,
            "prediction_type": "v_prediction",
        },
    },
    "dpmms": {
        "scheduler": "DPMSolverMultistepScheduler",
        "name": "DPMMS",
        "kwargs": {
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "beta_end": 0.03,
            "prediction_type": "v_prediction",
            "trained_betas": None,
            "solver_order": 2,
            "algorithm_type": "dpmsolver++",
        },
    },
}

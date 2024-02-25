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

from .trainer_model import SiTDiffusionModel
from .sit import DiT

from .transport import Transport, ModelType, WeightType, PathType, Sampler


def create_transport(
    path_type='Linear',
    prediction="velocity",
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
):
    """function for creating Transport object
    **Note**: model prediction defaults to velocity
    Args:
    - path_type: type of path to use; default to linear
    - learn_score: set model prediction to score
    - learn_noise: set model prediction to noise
    - velocity_weighted: weight loss by velocity weight
    - likelihood_weighted: weight loss by likelihood weight
    - train_eps: small epsilon for avoiding instability during training
    - sample_eps: small epsilon for avoiding instability during sampling
    """

    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }

    path_type = path_choice[path_type]

    if (path_type in [PathType.VP]):
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    elif (path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY):
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    else: # velocity & [GVP, LINEAR] is stable everywhere
        train_eps = 0
        sample_eps = 0
    
    # create flow state
    state = Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
    )
    return state

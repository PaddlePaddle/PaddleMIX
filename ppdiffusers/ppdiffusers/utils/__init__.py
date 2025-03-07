# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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


import os

from packaging import version

from ..version import VERSION as __version__
from . import initializer_utils
from .accelerate_utils import apply_forward_hook
from .constants import TEXT_ENCODER_ATTN_MODULE  # deprecate
from .constants import (  # fastdeploy; NEW; DIFFUSERS; PPDIFFUSERS; TRANSFORMERS; PADDLENLP
    CONFIG_NAME,
    DEPRECATED_REVISION_ARGS,
    DIFFUSERS_CACHE,
    DIFFUSERS_DYNAMIC_MODULE_NAME,
    DOWNLOAD_SERVER,
    FASTDEPLOY_MODEL_NAME,
    FASTDEPLOY_WEIGHTS_NAME,
    FROM_AISTUDIO,
    FROM_DIFFUSERS,
    FROM_HF_HUB,
    HF_MODULES_CACHE,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    LOW_CPU_MEM_USAGE_DEFAULT,
    MIN_PEFT_VERSION,
    NEG_INF,
    ONNX_EXTERNAL_WEIGHTS_NAME,
    ONNX_WEIGHTS_NAME,
    PADDLE_INFER_MODEL_NAME,
    PADDLE_INFER_WEIGHTS_NAME,
    PADDLE_SAFETENSORS_WEIGHTS_NAME,
    PADDLE_SAFETENSORS_WEIGHTS_NAME_INDEX_NAME,
    PADDLE_WEIGHTS_NAME,
    PADDLE_WEIGHTS_NAME_INDEX_NAME,
    PPDIFFUSERS_CACHE,
    PPDIFFUSERS_DYNAMIC_MODULE_NAME,
    PPDIFFUSERS_MODULES_CACHE,
    PPNLP_BOS_RESOLVE_ENDPOINT,
    PPNLP_PADDLE_WEIGHTS_INDEX_NAME,
    PPNLP_PADDLE_WEIGHTS_NAME,
    PPNLP_SAFE_WEIGHTS_INDEX_NAME,
    PPNLP_SAFE_WEIGHTS_NAME,
    TEST_DOWNLOAD_SERVER,
    TO_DIFFUSERS,
    TORCH_SAFETENSORS_WEIGHTS_NAME,
    TORCH_SAFETENSORS_WEIGHTS_NAME_INDEX_NAME,
    TORCH_WEIGHTS_NAME,
    TORCH_WEIGHTS_NAME_INDEX_NAME,
    TRANSFORMERS_SAFE_WEIGHTS_INDEX_NAME,
    TRANSFORMERS_SAFE_WEIGHTS_NAME,
    TRANSFORMERS_TORCH_WEIGHTS_INDEX_NAME,
    TRANSFORMERS_TORCH_WEIGHTS_NAME,
    USE_PEFT_BACKEND,
    get_map_location_default,
    str2bool,
)
from .deprecation_utils import deprecate
from .doc_utils import replace_example_docstring
from .download_utils import (
    SaveToAistudioMixin,
    _add_variant,
    _get_model_file,
    aistudio_download,
    bos_aistudio_hf_download,
    get_checkpoint_shard_files,
    ppdiffusers_url_download,
)
from .dynamic_modules_utils import get_class_from_dynamic_module
from .export_utils import (
    export_to_gif,
    export_to_obj,
    export_to_ply,
    export_to_video,
    export_to_video_2,
)
from .hub_utils import (
    HF_HUB_OFFLINE,
    PushToHubMixin,
    extract_commit_hash,
    http_user_agent,
)
from .import_utils import (
    BACKENDS_MAPPING,
    ENV_VARS_TRUE_AND_AUTO_VALUES,
    ENV_VARS_TRUE_VALUES,
    PPDIFFUSERS_SLOW_IMPORT,
    DummyObject,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_accelerate_available,
    is_accelerate_version,
    is_bs4_available,
    is_einops_available,
    is_fastdeploy_available,
    is_ftfy_available,
    is_inflect_available,
    is_k_diffusion_available,
    is_k_diffusion_version,
    is_librosa_available,
    is_note_seq_available,
    is_omegaconf_available,
    is_onnx_available,
    is_paddle_available,
    is_paddle_version,
    is_paddlenlp_available,
    is_paddlenlp_version,
    is_paddlesde_available,
    is_peft_available,
    is_pp_invisible_watermark_available,
    is_ppxformers_available,
    is_safetensors_available,
    is_scipy_available,
    is_tensorboard_available,
    is_torch_available,
    is_torch_version,
    is_transformers_available,
    is_transformers_version,
    is_unidecode_available,
    is_visualdl_available,
    is_wandb_available,
    recompute_use_reentrant,
    requires_backends,
    use_old_recompute,
)

# custom load_utils
from .load_utils import is_torch_file, safetensors_load, smart_load, torch_load
from .loading_utils import load_image
from .logging import get_logger
from .outputs import BaseOutput
from .paddle_utils import (
    apply_freeu,
    fourier_filter,
    get_rng_state_tracker,
    maybe_allow_in_graph,
    rand_tensor,
    randint_tensor,
    randn_tensor,
    dim2perm
)
from .peft_utils import (
    check_peft_version,
    delete_adapter_layers,
    get_adapter_name,
    get_peft_kwargs,
    recurse_remove_peft_layers,
    scale_lora_layers,
    set_adapter_layers,
    set_weights_and_activate_adapters,
    unscale_lora_layers,
)
from .pil_utils import (
    PIL_INTERPOLATION,
    make_image_grid,
    numpy_to_pil,
    pd_to_pil,
    pt_to_pil,
)

image_grid = make_image_grid

from .state_dict_utils import (
    convert_state_dict_to_peft,
    convert_state_dict_to_ppdiffusers,
    convert_unet_state_dict_to_peft,
)
from .testing_utils import (  # load_image,
    floats_tensor,
    get_examples_pipeline,
    image_grid,
    load_hf_numpy,
    load_numpy,
    load_pd,
    load_ppnlp_numpy,
    nightly,
    paddle_all_close,
    paddle_device,
    parse_flag_from_env,
    print_tensor_test,
    require_paddle_gpu,
    slow,
)

logger = get_logger(__name__)


def check_min_version(min_version):
    if __version__ == "0.0.0":
        return
    if version.parse(__version__) < version.parse(min_version):
        if "dev" in min_version:
            error_message = "This example requires a source install from PaddleMIX ppdiffusers"
        else:
            error_message = f"This example requires a minimum version of {min_version},"
        error_message += f" but the version found is {__version__}.\n"
        raise ImportError(error_message)

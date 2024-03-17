# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""
Import utilities: Utilities related to imports and our lazy inits.
"""
import importlib.util
import operator as op
import os
import sys
from collections import OrderedDict
from itertools import chain
from types import ModuleType
from typing import Any, Union

from huggingface_hub.utils import is_jinja_available  # noqa: F401
from packaging.version import Version, parse

from . import logging


def str2bool(v):
    if isinstance(v, bool):
        return v
    if not isinstance(v, str):
        v = str(v)
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Not supported value: {}".format(v))


# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_PADDLE = os.environ.get("USE_PADDLE", "AUTO").upper()
USE_SAFETENSORS = os.environ.get("USE_SAFETENSORS", "AUTO").upper()
PPDIFFUSERS_SLOW_IMPORT = os.environ.get("PPDIFFUSERS_SLOW_IMPORT", "FALSE").upper()
PPDIFFUSERS_SLOW_IMPORT = PPDIFFUSERS_SLOW_IMPORT in ENV_VARS_TRUE_VALUES

STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}

_paddle_version = "N/A"
if USE_PADDLE in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _paddle_available = importlib.util.find_spec("paddle") is not None
    _ppxformers_available = False
    if _paddle_available:
        try:
            import paddle

            _paddle_version = paddle.__version__
            logger.info(f"Paddle version {_paddle_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _paddle_available = False

        if _paddle_available:
            try:
                from paddle.incubate.nn.memory_efficient_attention import (  # noqa
                    memory_efficient_attention,
                )

                _ = memory_efficient_attention(
                    paddle.ones((1, 1, 2, 40), dtype=paddle.float16),
                    paddle.ones((1, 1, 2, 40), dtype=paddle.float16),
                    paddle.ones((1, 1, 2, 40), dtype=paddle.float16),
                )
                _ppxformers_available = True
            except Exception:
                _ppxformers_available = False

else:
    logger.info("Disabling Paddle because USE_PADDLE is set")
    _paddle_available = False
    _ppxformers_available = False

_torch_version = "N/A"
_torch_available = importlib.util.find_spec("torch") is not None
if _torch_available:
    try:
        _torch_version = importlib_metadata.version("torch")
        logger.info(f"PyTorch version {_torch_version} available.")
    except importlib_metadata.PackageNotFoundError:
        _torch_available = False

if USE_SAFETENSORS in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _safetensors_available = importlib.util.find_spec("safetensors") is not None
    if _safetensors_available:
        try:
            _safetensors_version = importlib_metadata.version("safetensors")
            logger.info(f"Safetensors version {_safetensors_version} available.")
        except importlib_metadata.PackageNotFoundError:
            _safetensors_available = False
else:
    logger.info("Disabling Safetensors because USE_TF is set")
    _safetensors_available = False

_transformers_available = importlib.util.find_spec("transformers") is not None
try:
    _transformers_version = importlib_metadata.version("transformers")
    logger.debug(f"Successfully imported transformers version {_transformers_version}")
except importlib_metadata.PackageNotFoundError:
    _transformers_available = False

_inflect_available = importlib.util.find_spec("inflect") is not None
try:
    _inflect_version = importlib_metadata.version("inflect")
    logger.debug(f"Successfully imported inflect version {_inflect_version}")
except importlib_metadata.PackageNotFoundError:
    _inflect_available = False

_unidecode_available = importlib.util.find_spec("unidecode") is not None
try:
    _unidecode_version = importlib_metadata.version("unidecode")
    logger.debug(f"Successfully imported unidecode version {_unidecode_version}")
except importlib_metadata.PackageNotFoundError:
    _unidecode_available = False


_onnxruntime_version = "N/A"
_onnx_available = importlib.util.find_spec("onnxruntime") is not None
if _onnx_available:
    candidates = (
        "onnxruntime",
        "onnxruntime-gpu",
        "ort_nightly_gpu",
        "onnxruntime-directml",
        "onnxruntime-openvino",
        "ort_nightly_directml",
        "onnxruntime-rocm",
        "onnxruntime-training",
    )
    _onnxruntime_version = None
    # For the metadata, we have to look for both onnxruntime and onnxruntime-gpu
    for pkg in candidates:
        try:
            _onnxruntime_version = importlib_metadata.version(pkg)
            break
        except importlib_metadata.PackageNotFoundError:
            pass
    _onnx_available = _onnxruntime_version is not None
    if _onnx_available:
        logger.debug(f"Successfully imported onnxruntime version {_onnxruntime_version}")

_fastdeploy_version = "N/A"
_fastdeploy_available = importlib.util.find_spec("fastdeploy") is not None
if _fastdeploy_available:
    candidates = ("fastdeploy_gpu_python", "fastdeploy_python")
    # For the metadata, we have to look for both fastdeploy_python and fastdeploy_gpu_python
    for pkg in candidates:
        try:
            _fastdeploy_version = importlib_metadata.version(pkg)
            break
        except importlib_metadata.PackageNotFoundError:
            pass
    _fastdeploy_available = _fastdeploy_version != "N/A"
    if _fastdeploy_available:
        logger.debug(f"Successfully imported fastdeploy version {_fastdeploy_version}")

_paddlenlp_available = importlib.util.find_spec("paddlenlp") is not None
try:
    _paddlenlp_version = importlib_metadata.version("paddlenlp")
    logger.debug(f"Successfully imported paddlenlp version {_paddlenlp_version}")
except importlib_metadata.PackageNotFoundError:
    if _paddlenlp_available:
        _paddlenlp_version = "0.0.0"

# (sayakpaul): importlib.util.find_spec("opencv-python") returns None even when it's installed.
# _opencv_available = importlib.util.find_spec("opencv-python") is not None
try:
    candidates = (
        "opencv-python",
        "opencv-contrib-python",
        "opencv-python-headless",
        "opencv-contrib-python-headless",
    )
    _opencv_version = None
    for pkg in candidates:
        try:
            _opencv_version = importlib_metadata.version(pkg)
            break
        except importlib_metadata.PackageNotFoundError:
            pass
    _opencv_available = _opencv_version is not None
    if _opencv_available:
        logger.debug(f"Successfully imported cv2 version {_opencv_version}")
except importlib_metadata.PackageNotFoundError:
    _opencv_available = False

_scipy_available = importlib.util.find_spec("scipy") is not None
try:
    _scipy_version = importlib_metadata.version("scipy")
    logger.debug(f"Successfully imported scipy version {_scipy_version}")
except importlib_metadata.PackageNotFoundError:
    _scipy_available = False

_librosa_available = importlib.util.find_spec("librosa") is not None
try:
    _librosa_version = importlib_metadata.version("librosa")
    logger.debug(f"Successfully imported librosa version {_librosa_version}")
except importlib_metadata.PackageNotFoundError:
    _librosa_available = False

_accelerate_available = True

_k_diffusion_available = importlib.util.find_spec("k_diffusion") is not None
try:
    _k_diffusion_version = importlib_metadata.version("k_diffusion")
    logger.debug(f"Successfully imported k-diffusion version {_k_diffusion_version}")
except importlib_metadata.PackageNotFoundError:
    _k_diffusion_available = False

_note_seq_available = importlib.util.find_spec("note_seq") is not None
try:
    _note_seq_version = importlib_metadata.version("note_seq")
    logger.debug(f"Successfully imported note-seq version {_note_seq_version}")
except importlib_metadata.PackageNotFoundError:
    _note_seq_available = False

_wandb_available = importlib.util.find_spec("wandb") is not None
try:
    _wandb_version = importlib_metadata.version("wandb")
    logger.debug(f"Successfully imported wandb version {_wandb_version }")
except importlib_metadata.PackageNotFoundError:
    _wandb_available = False

_omegaconf_available = importlib.util.find_spec("omegaconf") is not None
try:
    _omegaconf_version = importlib_metadata.version("omegaconf")
    logger.debug(f"Successfully imported omegaconf version {_omegaconf_version}")
except importlib_metadata.PackageNotFoundError:
    _omegaconf_available = False

_tensorboard_available = importlib.util.find_spec("tensorboard") is not None
try:
    _tensorboard_version = importlib_metadata.version("tensorboard")
    logger.debug(f"Successfully imported tensorboard version {_tensorboard_version}")
except importlib_metadata.PackageNotFoundError:
    _tensorboard_available = False

_visualdl_available = importlib.util.find_spec("visualdl") is not None
try:
    _visualdl_version = importlib_metadata.version("visualdl")
    logger.debug(f"Successfully imported visualdl version {_visualdl_version}")
except importlib_metadata.PackageNotFoundError:
    _visualdl_available = False

_einops_available = importlib.util.find_spec("einops") is not None
try:
    try:
        import einops
        import einops.layers.paddle

        einops.layers.paddle
        logger.debug(f"Successfully imported einops version {einops.__version__}")
    except ImportError:
        _einops_available = False
except importlib_metadata.PackageNotFoundError:
    _einops_available = False

_compel_available = importlib.util.find_spec("compel") is not None
try:
    _compel_version = importlib_metadata.version("compel")
    logger.debug(f"Successfully imported compel version {_compel_version}")
except importlib_metadata.PackageNotFoundError:
    _compel_available = False


_ftfy_available = importlib.util.find_spec("ftfy") is not None
try:
    _ftfy_version = importlib_metadata.version("ftfy")
    logger.debug(f"Successfully imported ftfy version {_ftfy_version}")
except importlib_metadata.PackageNotFoundError:
    _ftfy_available = False


_bs4_available = importlib.util.find_spec("bs4") is not None
try:
    # importlib metadata under different name
    _bs4_version = importlib_metadata.version("beautifulsoup4")
    logger.debug(f"Successfully imported ftfy version {_bs4_version}")
except importlib_metadata.PackageNotFoundError:
    _bs4_available = False

_paddlesde_available = importlib.util.find_spec("paddlesde") is not None
try:
    _paddlesde_version = importlib_metadata.version("paddlesde")
    logger.debug(f"Successfully imported paddlesde version {_paddlesde_version}")
except importlib_metadata.PackageNotFoundError:
    _paddlesde_available = False

_pp_invisible_watermark_available = importlib.util.find_spec("pp-invisible-watermark") is not None
try:
    _invisible_watermark_version = importlib_metadata.version("pp-invisible-watermark")
    logger.debug(f"Successfully imported pp-invisible-watermark version {_invisible_watermark_version}")
except importlib_metadata.PackageNotFoundError:
    _pp_invisible_watermark_available = False


_peft_available = str2bool(os.getenv("USE_PEFT_BACKEND", False))


def is_paddle_available():
    return _paddle_available


def is_paddlenlp_available():
    return _paddlenlp_available


def is_visualdl_available():
    return _visualdl_available


def is_fastdeploy_available():
    return _fastdeploy_available


def is_torch_available():
    USE_TORCH = str2bool(os.getenv("USE_TORCH", True))
    if USE_TORCH:
        return _torch_available
    else:
        return False


def is_safetensors_available():
    return _safetensors_available


def is_transformers_available():
    return _transformers_available


def is_inflect_available():
    return _inflect_available


def is_unidecode_available():
    return _unidecode_available


def is_onnx_available():
    return _onnx_available


def is_opencv_available():
    return _opencv_available


def is_scipy_available():
    return _scipy_available


def is_librosa_available():
    return _librosa_available


def is_ppxformers_available():
    USE_PPXFORMERS = str2bool(os.getenv("USE_PPXFORMERS", True))
    if USE_PPXFORMERS:
        return _ppxformers_available
    else:
        return False


# NOTE this is paddle accelerate
def is_accelerate_available():
    return _accelerate_available


def is_einops_available():
    return _einops_available


def is_k_diffusion_available():
    return False  # _k_diffusion_available


def is_note_seq_available():
    return _note_seq_available


def is_wandb_available():
    return _wandb_available


def is_omegaconf_available():
    return _omegaconf_available


def is_tensorboard_available():
    return _tensorboard_available


def is_compel_available():
    return _compel_available


def is_ftfy_available():
    return _ftfy_available


def is_bs4_available():
    return _bs4_available


def is_paddlesde_available():
    return _paddlesde_available


# This is paddle packge
def is_pp_invisible_watermark_available():
    return _pp_invisible_watermark_available


def is_peft_available():
    return _peft_available


# docstyle-ignore
FASTDEPLOY_IMPORT_ERROR = """
{0} requires the fastdeploy library but it was not found in your environment. You can install it with pip: `pip install
fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html`
"""

# docstyle-ignore
PADDLE_IMPORT_ERROR = """
{0} requires the Paddle library but it was not found in your environment. Checkout the instructions on the
installation page: https://www.paddlepaddle.org.cn/install/quick and follow the ones that match your environment.
"""

# docstyle-ignore
PPXFORMERS_IMPORT_ERROR = """
{0} requires the scaled_dot_product_attention but your PaddlePaddle donot have this. Checkout the instructions on the
installation page: https://www.paddlepaddle.org.cn/install/quick and follow the ones that match your environment.
"""

# docstyle-ignore
PADDLENLP_IMPORT_ERROR = """
{0} requires the paddlenlp library but it was not found in your environment. You can install it with pip: `pip
install paddlenlp`
"""

# docstyle-ignore
TENSORBOARD_IMPORT_ERROR = """
{0} requires the tensorboard library but it was not found in your environment. You can install it with pip: `pip
install tensorboard`
"""

# docstyle-ignore
VISUALDL_IMPORT_ERROR = """
{0} requires the visualdl library but it was not found in your environment. You can install it with pip: `pip
install visualdl`
"""

# docstyle-ignore
INFLECT_IMPORT_ERROR = """
{0} requires the inflect library but it was not found in your environment. You can install it with pip: `pip install
inflect`
"""

# docstyle-ignore
PYTORCH_IMPORT_ERROR = """
{0} requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
"""

# docstyle-ignore
ONNX_IMPORT_ERROR = """
{0} requires the onnxruntime library but it was not found in your environment. You can install it with pip: `pip
install onnxruntime`
"""

# docstyle-ignore
OPENCV_IMPORT_ERROR = """
{0} requires the OpenCV library but it was not found in your environment. You can install it with pip: `pip
install opencv-python`
"""

# docstyle-ignore
SCIPY_IMPORT_ERROR = """
{0} requires the scipy library but it was not found in your environment. You can install it with pip: `pip install
scipy`
"""

# docstyle-ignore
LIBROSA_IMPORT_ERROR = """
{0} requires the librosa library but it was not found in your environment.  Checkout the instructions on the
installation page: https://librosa.org/doc/latest/install.html and follow the ones that match your environment.
"""

# docstyle-ignore
TRANSFORMERS_IMPORT_ERROR = """
{0} requires the transformers library but it was not found in your environment. You can install it with pip: `pip
install transformers`
"""

# docstyle-ignore
UNIDECODE_IMPORT_ERROR = """
{0} requires the unidecode library but it was not found in your environment. You can install it with pip: `pip install
Unidecode`
"""

# docstyle-ignore
K_DIFFUSION_IMPORT_ERROR = """
{0} requires the k-diffusion library but it was not found in your environment. You can install it with pip: `pip
install k-diffusion`
"""

# docstyle-ignore
NOTE_SEQ_IMPORT_ERROR = """
{0} requires the note-seq library but it was not found in your environment. You can install it with pip: `pip
install note-seq`
"""

# docstyle-ignore
WANDB_IMPORT_ERROR = """
{0} requires the wandb library but it was not found in your environment. You can install it with pip: `pip
install wandb`
"""

# docstyle-ignore
OMEGACONF_IMPORT_ERROR = """
{0} requires the omegaconf library but it was not found in your environment. You can install it with pip: `pip
install omegaconf`
"""

# docstyle-ignore
TENSORBOARD_IMPORT_ERROR = """
{0} requires the tensorboard library but it was not found in your environment. You can install it with pip: `pip
install tensorboard`
"""

# docstyle-ignore
COMPEL_IMPORT_ERROR = """
{0} requires the compel library but it was not found in your environment. You can install it with pip: `pip install compel`
"""

# docstyle-ignore
EINOPS_IMPORT_ERROR = """
{0} requires the einops[paddle] library but it was not found in your environment. You can update it with pip: `pip
install -U einops or pip install git+https://github.com/arogozhnikov/einops.git `
"""

# docstyle-ignore
BS4_IMPORT_ERROR = """
{0} requires the Beautiful Soup library but it was not found in your environment. You can install it with pip:
`pip install beautifulsoup4`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
FTFY_IMPORT_ERROR = """
{0} requires the ftfy library but it was not found in your environment. Checkout the instructions on the
installation section: https://github.com/rspeer/python-ftfy/tree/master#installing and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
PADDLESDE_IMPORT_ERROR = """
{0} requires the paddlesde library but it was not found in your environment. You can install it with pip: `pip install paddlesde`
"""

# docstyle-ignore
PP_INVISIBLE_WATERMARK_IMPORT_ERROR = """
{0} requires the pp-invisible-watermark library but it was not found in your environment. You can install it with pip: `pip install pp-invisible-watermark>=0.2.0`
"""


BACKENDS_MAPPING = OrderedDict(
    [
        ("bs4", (is_bs4_available, BS4_IMPORT_ERROR)),
        ("fastdeploy", (is_fastdeploy_available, FASTDEPLOY_IMPORT_ERROR)),
        ("paddle", (is_paddle_available, PADDLE_IMPORT_ERROR)),
        ("paddlenlp", (is_paddlenlp_available, PADDLENLP_IMPORT_ERROR)),
        ("visualdl", (is_visualdl_available, VISUALDL_IMPORT_ERROR)),
        ("inflect", (is_inflect_available, INFLECT_IMPORT_ERROR)),
        ("onnx", (is_onnx_available, ONNX_IMPORT_ERROR)),
        ("opencv", (is_opencv_available, OPENCV_IMPORT_ERROR)),
        ("scipy", (is_scipy_available, SCIPY_IMPORT_ERROR)),
        ("torch", (is_paddle_available, PYTORCH_IMPORT_ERROR)),
        ("transformers", (is_paddlenlp_available, TRANSFORMERS_IMPORT_ERROR)),
        ("unidecode", (is_unidecode_available, UNIDECODE_IMPORT_ERROR)),
        ("librosa", (is_librosa_available, LIBROSA_IMPORT_ERROR)),
        ("k_diffusion", (is_k_diffusion_available, K_DIFFUSION_IMPORT_ERROR)),
        ("note_seq", (is_note_seq_available, NOTE_SEQ_IMPORT_ERROR)),
        ("wandb", (is_wandb_available, WANDB_IMPORT_ERROR)),
        ("omegaconf", (is_omegaconf_available, OMEGACONF_IMPORT_ERROR)),
        ("tensorboard", (is_tensorboard_available, TENSORBOARD_IMPORT_ERROR)),
        ("einops", (is_einops_available, EINOPS_IMPORT_ERROR)),
        ("note_seq", (is_note_seq_available, NOTE_SEQ_IMPORT_ERROR)),
        ("compel", (is_compel_available, COMPEL_IMPORT_ERROR)),
        ("ftfy", (is_ftfy_available, FTFY_IMPORT_ERROR)),
        ("paddlesde", (is_paddlesde_available, PADDLESDE_IMPORT_ERROR)),
        ("pp_invisible_watermark", (is_pp_invisible_watermark_available, PP_INVISIBLE_WATERMARK_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError("".join(failed))

    if name in [
        "VersatileDiffusionTextToImagePipeline",
        "VersatileDiffusionPipeline",
        "VersatileDiffusionDualGuidedPipeline",
        "StableDiffusionImageVariationPipeline",
        "UnCLIPPipeline",
    ] and is_paddlenlp_version("<", "2.6.0"):
        raise ImportError(
            f"You need to install `paddlenlp>=2.6.0` in order to use {name}: \n```\n pip install"
            " --upgrade paddlenlp \n```"
        )

    if name in ["StableDiffusionDepth2ImgPipeline", "StableDiffusionPix2PixZeroPipeline"] and is_paddlenlp_version(
        "<", "2.6.1"  # TODO version
    ):
        raise ImportError(
            f"You need to install `paddlenlp>=2.6.0` in order to use {name}: \n```\n pip install"
            " --upgrade paddlenlp \n```"
        )


class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    def __getattr__(cls, key):
        if key.startswith("_") and key not in ["_load_connected_pipes", "_is_onnx", "_is_fastdeploy"]:
            return super().__getattr__(cls, key)
        requires_backends(cls, cls._backends)


# This function was copied from: https://github.com/huggingface/accelerate/blob/874c4967d94badd24f893064cc3bef45f57cadf7/src/accelerate/utils/versions.py#L319
def compare_versions(library_or_version: Union[str, Version], operation: str, requirement_version: str):
    """
    Args:
    Compares a library version to some requirement using a given operation.
        library_or_version (`str` or `packaging.version.Version`):
            A library name or a version to check.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`.
        requirement_version (`str`):
            The version to compare the library version against
    """
    if operation not in STR_OPERATION_TO_FUNC.keys():
        raise ValueError(f"`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}")
    operation = STR_OPERATION_TO_FUNC[operation]
    if isinstance(library_or_version, str):
        library_or_version = parse(importlib_metadata.version(library_or_version))
    return operation(library_or_version, parse(requirement_version))


# This function was copied from: https://github.com/huggingface/accelerate/blob/874c4967d94badd24f893064cc3bef45f57cadf7/src/accelerate/utils/versions.py#L338
def is_torch_version(operation: str, version: str):
    """
    Args:
    Compares the current PyTorch version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A string version of PyTorch
    """
    return compare_versions(parse(_torch_version), operation, version)


def is_paddle_version(operation: str, version: str):
    """
    Args:
    Compares the current Paddle version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A string version of Paddle
    """
    if not _paddle_available:
        return False
    if _paddle_version == "0.0.0":
        return True
    return compare_versions(parse(_paddle_version), operation, version)


def is_paddlenlp_version(operation: str, version: str):
    """
    Args:
    Compares the current paddlenlp version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _paddlenlp_available:
        return False
    if _paddlenlp_version == "0.0.0" or "post" in _paddlenlp_version:
        return True
    return compare_versions(parse(_paddlenlp_version), operation, version)


def is_transformers_version(operation: str, version: str):
    """
    Args:
    Compares the current Transformers version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _paddlenlp_available:
        return False
    return compare_versions(parse(_transformers_version), operation, version)


def is_accelerate_version(operation: str, version: str):
    """
    Args:
    Compares the current Accelerate version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _accelerate_available:
        return False
    from ppdiffusers.accelerate import __version__

    return compare_versions(parse(__version__), operation, version)


def is_k_diffusion_version(operation: str, version: str):
    """
    Args:
    Compares the current k-diffusion version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _k_diffusion_available:
        return False
    return compare_versions(parse(_k_diffusion_version), operation, version)


def use_old_recompute():
    return str2bool(os.getenv("FLAG_USE_OLD_RECOMPUTE", "False"))


def recompute_use_reentrant():
    if use_old_recompute():
        return True
    # if paddle 2.5.0 or higher, recompute_use_reentrant is False by default
    if is_paddle_version(">=", "2.5.0") or is_paddle_version("==", "0.0.0"):
        return str2bool(os.getenv("FLAG_RECOMPUTE_USE_REENTRANT", "False"))
    return True


def get_objects_from_module(module):
    """
    Args:
    Returns a dict of object names and values in a module, while skipping private/internal objects
        module (ModuleType):
            Module to extract the objects from.

    Returns:
        dict: Dictionary of object names and corresponding values
    """

    objects = {}
    for name in dir(module):
        if name.startswith("_"):
            continue
        objects[name] = getattr(module, name)

    return objects


class OptionalDependencyNotAvailable(BaseException):
    """An error indicating that an optional dependency of Diffusers was not found in the environment."""


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))

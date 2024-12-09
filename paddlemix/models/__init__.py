# copyright (c) 2023 paddlepaddle authors. all rights reserved.
# copyright 2023 the salesforce team authors and the huggingface team. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

from .audioldm2.configuration import *
from .audioldm2.modeling import *
from .blip2.modeling import *
from .cogvlm.configuration import *
from .cogvlm.modeling import *
from .imagebind.modeling import *
from .imagebind.multimodal_preprocessors import *
from .internlm_xcomposer2 import *
from .llava import *
from .minigpt4.configuration import *
from .minigpt4.modeling import *
from .visualglm.configuration import *
from .visualglm.modeling import *
from .qwen_vl import *
from .minicpm_v import *
from .janus import *
from .diffsinger import *

import pkg_resources

version = pkg_resources.get_distribution("paddlenlp").version   
try:
    if version.startswith('3'):
        from .internvl2 import *
        from .qwen2_vl import *

    else:
        print(f"paddlenlp version {version} is not 3.x, skipping import internvl2 and qwen2_vl.")

except ImportError:
    print("paddlenlp is not installed.")



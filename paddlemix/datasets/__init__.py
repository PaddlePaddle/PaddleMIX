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

# Standard imports

import pkg_resources

# Local imports
from .caption_dataset import *
from .chatml_dataset import *
from .coco_caption import *
from .coco_clip import *
from .collator import *
from .dataset import *
from .mixtoken_dataset import *
from .vg_caption import *

version = pkg_resources.get_distribution("paddlenlp").version
try:
    if version.startswith("3"):
        from .internvl_dataset import *
    else:
        print(f"paddlenlp version {version} is not 3.x, skipping import internvl2 datasets.")

except ImportError:
    print("paddlenlp is not installed.")

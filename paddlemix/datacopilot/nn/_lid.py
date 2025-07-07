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


import tempfile
from typing import Optional

from ..misc import download_url_to_file

LID_MODEL_URL = {
    "lid.176.bin": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
    "lid.176.ftz": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
}


class FastTextLIDModel(object):
    def __init__(self, path: Optional[str] = None, name: str = "lid.176.bin") -> None:
        self._name = name
        self._path = path
        self._model = None

    @property
    def names(
        self,
    ):
        return list(LID_MODEL_URL.keys())

    @property
    def model(
        self,
    ):
        import fasttext

        if self._model is None:
            if self._path is None:
                url = LID_MODEL_URL[self._name]
                with tempfile.NamedTemporaryFile(delete=True) as f:
                    download_url_to_file(url, f.name)
                    self._model = fasttext.load_model(f.name)
            else:
                self._model = fasttext.load_model(self._path)

        return self._model

    def predict(self, text: str, k: int = 1, threshold: float = 0):
        return self.model.predict(text, k=k, threshold=threshold)

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


import time
import random
import requests

class ErnieEval(object):
    """
    ErnieEval class for evaluating Ernie model.
    """
    def __init__(self, 
                model_name="ernie-speed-128k", 
                access_token="", 
                ak="", sk="", 
                api_type="aistudio", 
                max_retries=1):
        super().__init__()
        config = {
            "api_type": api_type,
            "max_retries": max_retries,
        }
        if access_token:
            config["access_token"] = access_token
        else:
            config["ak"] = ak
            config["sk"] = sk
        self.model_name = model_name
        self.config = config
    
    def predict(self, prompts, temperature=0.001):
        import erniebot
        chat_completion = erniebot.ChatCompletion.create(
            _config_=self.config,
            model=self.model_name,
            messages=[{"role": "user", "content": prompts}],
            temperature=float(temperature),
        )
        res = chat_completion.get_result()
        return res


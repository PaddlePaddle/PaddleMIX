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
import erniebot

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
        chat_completion = erniebot.ChatCompletion.create(
            _config_=self.config,
            model=self.model_name,
            messages=[{"role": "user", "content": prompts}],
            temperature=float(temperature),
        )
        res = chat_completion.get_result()
        return res

class PaddleGPT4o:
    def __init__(self, ):
        pass

    def get_output(self, prompts, ):
        while True:
            rst = self.get_result(
                text=prompts,
                max_tokens=2048,
                temperature=0.9,
                topp=0,
                penalty_score=1,
            )
            if len(rst) > 1:
                break
            time.sleep(1)
        return rst

    def gen_session_id(self, prefix="1"):
        # 生成新Session ID
        return str(prefix) + "session_id_time_%f_rand_%f" % (time.time(), random.random())

    def request_gpt4(self, json_data):    
        response = requests.request("POST", "http://10.88.94.144:8900/generate", json=json_data).json()
        return response
    
    def get_result(self, text, max_tokens, temperature, topp, penalty_score, history=None):
        request_data = {
            "context": [
                {
                    "role": "system",
                    "utterance": [{
                        "type": "text",
                        "text": text[0],
                        },
                    ],
                },
                {
                    "role": "user", 
                    "utterance": [{
                        "type": "text",
                        "text": text[1],
                        },
                    ],
                }
            ],
            "top_p": topp,
            "temperature": temperature,
            "penalty_score": penalty_score,
            "frequency_score": 0,
            "presence_score": 0,
            "min_dec_len": 2,
            "max_dec_len": max_tokens,
        }
    
        SUCCESS = False
        session_id = self.gen_session_id()

        try:
            rst = self.request_gpt4(request_data)
            if 'result' in rst:
                if 'response' in rst['result']:
                    if 'utterance' in rst['result']['response']:
                        SUCCESS = True
            result = rst['result']['response']['utterance']
        
        except Exception as e:
            print("error:", e, "session_id:", session_id)
        if SUCCESS:
            return result
        else:
            return ""

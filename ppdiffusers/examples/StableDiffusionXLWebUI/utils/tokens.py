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

import requests

API_URL = "https://vaxeldh1n0cdaah0.aistudio-hub.baidu.com/chat/completions"


def query(payload, token="your access token"):
    headers = {
        # 请前往 https://aistudio.baidu.com/index/accessToken 查看 访问令牌
        "Authorization": f"token {token}",
        "Content-Type": "application/json",
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def prompt_opt(prompt, lang="英文"):
    output = query(
        {"messages": [{"role": "user", "content": f"{prompt}，优化并添加细节，然后仅给出{lang}结果，不要任何多余内容，控制77tokens以内。"}]}
    )
    return output["result"].replace("Optimized Prompt: ", "")

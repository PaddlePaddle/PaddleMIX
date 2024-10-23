# -*- coding: utf-8 -*-

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

# This code shows an example of text translation from English to Simplified-Chinese.
# This code runs on Python 2.7.x and Python 3.x.
# You may install `requests` to run this code: pip install requests
# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document

import random
import re
import time

# import json
from hashlib import md5

import requests

# import string

# Set your own appid/appkey.
appid = None or "20230825001792923"  # '你自己申请的'
appkey = None or "s4stlEqXE7t3zdSmImyI"  # '你自己申请的'


def translate_a_to_b(query, from_lang="zh", to_lang="en"):
    # For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
    from_lang = from_lang
    to_lang = to_lang

    endpoint = "http://api.fanyi.baidu.com"
    path = "/api/trans/vip/translate"
    url = endpoint + path

    # Generate salt and sign
    def make_md5(s, encoding="utf-8"):
        return md5(s.encode(encoding)).hexdigest()

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = {"appid": appid, "q": query, "from": from_lang, "to": to_lang, "salt": salt, "sign": sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()
    # print(result)
    return result.get("trans_result", {"dst": query})[0].get("dst", query)

    # # Show response
    # return json.dumps(result, indent=4, ensure_ascii=False)


def translate_chinese_in_sentence(sentence):
    # punctuation = string.punctuation
    # print(punctuation)
    # 使用正则表达式找到所有中文文本
    chinese_pattern = re.compile(r"([\u4e00-\u9fff]+(?:[\w])+)")
    chinese_texts = chinese_pattern.findall(sentence)
    # print(chinese_texts)
    if chinese_texts:
        try:
            translation_texts = translate_a_to_b(str(chinese_texts), from_lang="zh", to_lang="en")
        except:
            print("翻译失败，请检查翻译密钥是否正确。")
            return sentence
        translation_texts = translation_texts.strip("[").rstrip("]").split(",")  # 对生成的字符串列表str去除“[”和“]”，再以“,”分割，生成列表
        translation_texts = [i.strip("'").rstrip("'").replace("\\'", "'") for i in translation_texts]  # 去除多余“'”，成新列表
        # print(translation_texts)
        # 替换原句中的中文为英文翻译
        for original, translated in zip(chinese_texts, translation_texts):
            sentence = sentence.replace(original, translated)
    return sentence


def multi_tasks_translate(sentence, *args):
    """
    # 示例
    from baidufanyi import multi_tasks_translate
    st1, st2, st3, st4 = (
        "", # 空字符
        "hello,我喜欢吃苹果，898，jenny，它很甜,jeny,它很甜，我喜欢吃苹果，我喜欢吃苹果", # 中英数字混合
        "This code shows an example of text translation", # 纯英文
        "我喜欢吃苹果", # 纯中文
    )
    results = multi_tasks_translate(st1, st2, st3, st4) # 输出为list
    tuple(results) # 转成元组
    """

    result = [translate_chinese_in_sentence(sentence)]
    for sentence in args:
        time.sleep(0.9)
        result.append(translate_chinese_in_sentence(sentence))
    return result

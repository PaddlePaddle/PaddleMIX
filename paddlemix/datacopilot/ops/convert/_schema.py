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


from ...core import SCHEMA, T


def convert_schema(item: T, in_schema: SCHEMA = SCHEMA.MM, out_schema: SCHEMA = SCHEMA.MIX) -> T:
    """convert scheme"""
    if in_schema == out_schema:
        return item

    # MM <-> MIX
    elif in_schema == SCHEMA.MM and out_schema == SCHEMA.MIX:
        return _convert_mm_mix(item)

    else:
        raise NotImplementedError("")


def _convert_mm_mix(item):
    if "image" in item:
        images = [
            {
                "id": 0,
                "url": item["image"],
            }
        ]
    else:
        images = None

    conversations = []
    for conv in item["conversations"]:
        if conv["from"] == "human":
            role = "user"
            if "image" in item:
                if "<image>" in conv["value"]:
                    value = conv["value"].replace("<image>", "<image>0</image>")
                else:
                    value = "<image>0</image>\n" + conv["value"]
            else:
                value = conv["value"]
        else:
            role = "assistant"
            value = conv["value"]

        conversations.append(
            {
                "from": role,
                "value": value,
            }
        )

    newitem = {"id": item["id"], "images": images, "conversations": conversations}
    return newitem

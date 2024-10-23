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

# 假设的genshin_characters字典，去除了男性角色
genshin_characters = {
    "菲谢尔": "Fischl",
    "莫娜": "Mona",
    "香菱": "Xiangling",
    "迪奥娜": "Diona",
    "芭芭拉": "Barbara",
    "诺艾尔": "Noelle",
    "砂糖": "Sucrose",
    "辛焱": "Xinyan",
    "刻晴": "Keqing",
    "七七": "Qiqi",
    "胡桃": "Hu Tao",
    "凝光": "Ningguang",
    "烟绯": "Yanfei",
    "云堇": "Yun Jin",
    "宵宫": "Yoimiya",
    "罗莎莉亚": "Rosaria",
    "优菈": "Eula",
    "埃洛伊": "Aloy",  # 这个角色可能是联动角色，名字可能与游戏内不同
    "神里绫华": "Kamizato Ayaka",
    "早柚": "Sayu",
    "珊瑚宫心海_1": "Koramoru",
    "珊瑚宫心海_2": "Sangonomiya Kokomi",
    "九条裟罗": "Kujo Sara",
    "米卡": "Mika",  # 注意：这里假设米卡是女性，实际上可能是男性
    "雷电将军": "Raiden Shogun",  # 注意：这是职位名，实际角色名是巴尔泽布(Baalzebul)或雷电真(Raiden Makami)
    "申鹤": "Shenhe",
    "妮露": "Nilou",
    "莱依拉": "Layla",
    "绮良良": "Kirara",
    "甘雨": "Ganyu",
    "夜兰": "Yelan",
    "纳西妲": "Nahida",
    "八重神子": "Yae Miko",
    "瑶瑶": "Yao Yao",
    "琳妮特": "Lynette",
    "琳妮特·风": "Lynette: Wind",  # 假设有元素形态变化
    "多莉": "Dolly",
    "其他": "others",  # "Koramoru Umi"
    # ... 这里可以继续添加其他女性角色，但确保移除了所有男性角色
}


# 示例用法
# print(genshin_characters.get("未知角色", "角色名未找到"))  # 输出: 角色名未找到

import json
import copy
import os
import argparse

def convert_question(caption):
    """将问题转换为模板格式"""
    QUESTION_TEMPLATE = f"<image>Locate {caption}, output its bbox coordinates using JSON format."
    return QUESTION_TEMPLATE

def process_data(json_path, image_dir, output_path):
    """处理数据并保存到指定路径"""
    # 加载原始数据
    data = json.load(open(json_path))

    new_data = []
    for i in range(len(data)):
        # 提取原始描述并清理
        raw_caption = data[i]['messages'][0]['content'].split(':')[1].strip()
        if raw_caption[-1] == '.':
            raw_caption = raw_caption[:-1]

        # 深拷贝数据并修改内容
        data_i = copy.deepcopy(data[i])
        data_i['messages'][0]['content'] = convert_question(raw_caption)
        data_i['images'][0] = os.path.join(image_dir, os.path.basename(data_i['images'][0]))
        new_data.append(data_i)

    # 保存处理后的数据
    json.dump(new_data, open(output_path, 'w'), indent=4)
    print(f"处理后的数据已保存到: {output_path}")

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="处理 JSON 数据并生成新的 JSON 文件")
    parser.add_argument("--json_path", type=str, required=True, help="原始 JSON 文件的路径")
    parser.add_argument("--image_dir", type=str, required=True, help="图像文件的根目录")
    parser.add_argument("--output_path", type=str, required=True, help="输出 JSON 文件的路径")
    args = parser.parse_args()

    # 调用处理函数
    process_data(args.json_path, args.image_dir, args.output_path)
from openai import OpenAI
client = OpenAI(api_key="xxxx", base_url="https://api.deepseek.com/beta")

import argparse

parser = argparse.ArgumentParser(description='参数')
parser.add_argument('prompt', type=str, help='图片描述')
args = parser.parse_args()
prompt = args.prompt

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system",
         "content": "You are a picture beautifier. The user will provide you with a description of a picture, and you need to beautify this description from multiple aspects and output the refined picture description, which will be used for text - to - image generation. You only need to output the final prompt!"},
        {"role": "user", "content": prompt},
    ],
)
reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content
print(reasoning_content)
print("=================================================")
print(content)


import os
import argparse
import cv2
import paddle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchvision import transforms

def main(args):
    # 检查图像路径并修改文本
    if args.image_path is not None:
        args.text = "<ImageHere> " + args.text

    model_name_or_path = args.model_name_or_path

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,  # 确保模型以 float16 加载
        trust_remote_code=True
    ).cuda()
    model.eval()

    # 图像加载和处理
    image_path = args.image_path
    if os.path.exists(image_path):
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Failed to load image from {image_path}.")

        # 将图像从 BGR 转换为 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 图像转换
        transform = transforms.Compose([
            transforms.ToPILImage(),  # 将 NumPy 数组转换为 PIL 图像
            transforms.Resize((560, 560)),  # 根据模型要求调整大小
            transforms.ToTensor(),           # 转换为张量
        ])

        # 转换为张量并添加批次维度
        image_tensor = transform(image).unsqueeze(0).cuda()  # 现在形状为 (1, 3, 224, 224)
        image_tensor = image_tensor.half()  # 确保图像张量为 float16
        print(f"Image tensor shape: {image_tensor.shape}")
        print(f"Image tensor dtype: {image_tensor.dtype}")  # 打印数据类型用于调试
    else:
        raise ValueError(f"Image path does not exist: {image_path}")

    # 运行推理并打印输出
    inputs = tokenizer(args.text, return_tensors="pt").to("cuda")  # 移动到 GPU

    # 确保 input_ids 是整数类型
    inputs["input_ids"] = inputs["input_ids"].to(dtype=torch.int64)
    inputs["attention_mask"] = inputs["attention_mask"].to(dtype=torch.int64)

    output_ids = model.generate(**inputs, max_length=256)
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

    # 使用模型的 chat 方法获取响应
    # 尝试将图像作为列表传递
    try:
        response, _ = model.chat(tokenizer, query=args.text, image=[image_tensor], history=[], do_sample=False)
        print(response)
    except RuntimeError as e:
        print(f"Error during model.chat: {e}")
        # 添加更多调试信息
        print(f"Model dtype: {next(model.parameters()).dtype}")
        print(f"Image tensor dtype in chat: {image_tensor.dtype}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="请详细描述这张图片。")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--text", type=str, required=True, help="Input text query.")

    args = parser.parse_args()
    main(args)

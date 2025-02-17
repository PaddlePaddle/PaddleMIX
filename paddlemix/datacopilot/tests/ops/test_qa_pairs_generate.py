import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from paddlemix.datacopilot.core import MMDataset
from paddlemix.datacopilot.ops.generate._qa_pairs_generate import generate_qna_for_images



# 调用分析函数
dataset = generate_qna_for_images(image_folder_path="paddlemix/demo_images")

dataset.export_json('test_qwen2_vl_1w.json')
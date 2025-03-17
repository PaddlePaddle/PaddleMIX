import argparse
import json
import json
import os
import random
import re
import sys
from pprint import pprint

import paddle
from paddlenlp.generation import GenerationConfig
from paddlenlp.utils.import_utils import import_module
from tqdm import tqdm

sys.path.append('paddlemix/examples/r1_mllm')
from r1_mllm.utils.tokenizer import get_processor
from r1_mllm.utils.constant import TEMPLATE_MAPPING, MODEL_MAPPING, SUPPORTED_MODELS

def parse_args():
    parser = argparse.ArgumentParser(description="Run VLM-R1 evaluation.")
    parser.add_argument("--model_name", type=str,default="Qwen2.5-VL-3B-Instruct", required=True, help="Name of the model.")
    parser.add_argument("--model_path", type=str,default="Qwen/Qwen2.5-VL-3B-Instruct", required=True, help="Path to the model checkpoint.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the dataset.")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory of the images.")
    parser.add_argument("--test_datasets", type=str, nargs="+", default=["refgta_subsample"], help="List of datasets to evaluate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--sample_num", type=int, default=500, help="Number of samples to evaluate.")
    parser.add_argument("--steps", type=int, default=100, help="Checkpoint steps for logging.")
    parser.add_argument("--method", type=str, default="r1", help="Choose test r1 or baseline")
    parser.add_argument("--seed", type=int, default=42, help="Seed for inference.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type for inference.")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="Attention type for inference.")
    return parser.parse_args()

def extract_bbox_answer(method,content):
    bbox_match = False
    if method == "baseline":
        bbox_pattern = r'(-?\d*\.?\d+),\s*(-?\d*\.?\d+),\s*(-?\d*\.?\d+),\s*(-?\d*\.?\d+)'
        bbox_match = re.search(bbox_pattern, content)
    else:
        answer_tag_pattern = "\s*<answer>(.*?)</answer>"
        bbox_pattern = r"\[([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+)\]"

        content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            bbox_match = re.search(bbox_pattern, content_answer)
    if bbox_match:
        bbox = [
            float(bbox_match.group(1)),
            float(bbox_match.group(2)),
            float(bbox_match.group(3)),
            float(bbox_match.group(4)),
        ]
        x1, y1, x2, y2 = bbox
        return bbox, False
    return [0, 0, 0, 0], False


def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2] - 1, box2[2] - 1)
    inter_y2 = min(box1[3] - 1, box2[3] - 1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = (
        (box1[2] - box1[0]) * (box1[3] - box1[1])
        + (box2[2] - box2[0]) * (box2[3] - box2[1])
        - inter
    )
    return float(inter) / union

def get_test_template(method):
    # TODO: Support other prompt
    if method =="r1":
        QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
    else:
        QUESTION_TEMPLATE = "Locate {Question}, output its bbox coordinates using JSON format."
    
    return QUESTION_TEMPLATE

def main(args):
    steps = args.steps
    print("Steps: ", steps)
    MODEL_PATH = args.model_path
    os.makedirs("logs",exist_ok=True)
    OUTPUT_PATH = "./logs/rec_results_{DATASET}_{MODEL}_{METHOD}_{STEPS}.json"
    BSZ = args.batch_size
    DATA_ROOT = args.data_root
    TEST_DATASETS = args.test_datasets
    IMAGE_ROOT = args.image_root
    random.seed(args.seed)

    template_name = TEMPLATE_MAPPING[args.model_name]
    process_vision_info = import_module(f"paddlemix.processors.{template_name}_processing.process_vision_info")
    
    # register model
    model_cls = import_module(f"paddlemix.models.{MODEL_MAPPING[args.model_name]}")
    model = model_cls.from_pretrained(
        MODEL_PATH,
        dtype=args.dtype,
    )
    processor, tokenizer = get_processor(args.model_name,SUPPORTED_MODELS[args.model_name])

    sample_num = args.sample_num
    for ds in TEST_DATASETS:
        print(f"Processing {ds}...")
        ds_path = os.path.join(DATA_ROOT, f"{ds}.json")
        data = json.load(open(ds_path, "r"))
        random.shuffle(data)
        QUESTION_TEMPLATE = get_test_template(args.method)
        data = data[:sample_num]
        messages = []
        for x in data:
            image_path = os.path.join(IMAGE_ROOT, x["image"])
            question = x['normal_caption'] if args.method=="baseline" else x['problem'] 
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question),
                        },
                    ],
                }
            ]
            messages.append(message)
        
        all_outputs = []
        for i in tqdm(range(0, len(messages), BSZ)):
            batch_messages = messages[i : i + BSZ]
            text = [
                processor.tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in batch_messages
            ]
            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pd",
            )
            text_inputs = processor.tokenizer(
                text=text,
                padding=True,
                padding_side="left",
                return_tensors="pd",
            )
            # TODO
            # padding side bug 
            inputs.update(text_inputs)
            generation_config = GenerationConfig(
                use_cache=True,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=model.config.eos_token_id,
                pad_token_id=model.config.pad_token_id,
            )
            with paddle.no_grad():
                generated_ids = model.generate(
                    **inputs,generation_config=generation_config
                )[0]
            generated_ids_trimmed = [
                out_ids
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            batch_output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            # processor.batch_decode(text_inputs['input_ids'],skip_special_tokens=True,)
            all_outputs.extend(batch_output_text)
        final_output = []
        correct_number = 0
        for input_example, model_output in zip(data, all_outputs):
            original_output = model_output
            ground_truth = input_example["solution"]
            ground_truth_normalized = input_example["normalized_solution"]
            model_answer, normalized = extract_bbox_answer(args.method,original_output)
            correct = 0
            if model_answer is not None:
                if not normalized and iou(model_answer, ground_truth) > 0.5:
                    correct = 1
                elif normalized and iou(model_answer, ground_truth_normalized) > 0.5:
                    correct = 1
            correct_number += correct
            result = {
                "question": QUESTION_TEMPLATE.format(Question=x["problem"]),
                "ground_truth": ground_truth,
                "model_output": original_output,
                "extracted_answer": model_answer,
                "correct": correct,
            }
            final_output.append(result)
        accuracy = correct_number / len(data) * 100
        print(f"\nAccuracy of {ds}: {accuracy:.2f}%")
        output_path = OUTPUT_PATH.format(DATASET=ds,MODEL=args.model_name.lower(), METHOD=args.method, STEPS=steps)
        with open(output_path, "w") as f:
            json.dump({"accuracy": accuracy, "results": final_output}, f, indent=2)
        print(f"Results saved to {output_path}")
        print("-" * 100)

if __name__=="__main__":
    args = parse_args()
    main(args)
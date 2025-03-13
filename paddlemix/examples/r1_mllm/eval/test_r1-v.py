
import argparse
import json
import json
import os
import random
import re
import sys
from pprint import pprint

from tqdm import tqdm
import paddle
from paddlenlp.generation import GenerationConfig
from paddlenlp.utils.import_utils import import_module

sys.path.append('paddlemix/examples/r1_mllm')
from r1_mllm.utils.tokenizer import get_processor
from r1_mllm.utils.constant import TEMPLATE_MAPPING, MODEL_MAPPING, SUPPORTED_MODELS

def parse_args():
    parser = argparse.ArgumentParser(description="Run R1-V evaluation.")
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
    return parser.parse_args()


def get_test_template(method):
    # TODO: Support other prompt
    if method =="r1":
        QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
    else:
        QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."
    
    return QUESTION_TEMPLATE

def extract_number_answer(output_str):
    # Try to find the number within <answer> tags, if can not find, return None
    answer_pattern = r'<answer>\s*(\d+)\s*</answer>'
    match = re.search(answer_pattern, output_str)
    
    if match:
        return int(match.group(1))
    return None

def main(args):
    steps = args.steps
    print("Steps: ", steps)
    MODEL_PATH = args.model_path
    os.makedirs("logs",exist_ok=True)
    OUTPUT_PATH = "./logs/r1-v_results_{DATASET}_{MODEL}_{METHOD}_{STEPS}.json"
    BSZ = args.batch_size
    DATA_ROOT = args.data_root
    TEST_DATASETS = args.test_datasets
    IMAGE_ROOT = args.image_root
    random.seed(args.seed)
    template_name = TEMPLATE_MAPPING[args.model_name]
    process_vision_info = import_module(f"paddlemix.processors.{template_name}_processing.process_vision_info")

    model_cls = import_module(f"paddlemix.models.{MODEL_MAPPING[args.model_name]}")
    model = model_cls.from_pretrained(
        MODEL_PATH,
        dtype=args.dtype,
    )

    processor, tokenizer = get_processor(args.model_name,SUPPORTED_MODELS[args.model_name])
    sample_num = args.sample_num
    for ds in TEST_DATASETS:
        print(f"Processing {ds}...")
        ds_path = os.path.join(DATA_ROOT, f"{ds}.jsonl")
        # data = json.load(open(ds_path, "r"))
        with open(ds_path, 'r') as file:
            data = [json.loads(line) for line in file]
        QUESTION_TEMPLATE = get_test_template(args.method)
        messages = []
        for x in data:
            if x["image_path"].startswith("./"):
                x["image_path"] = x["image_path"][2:]
            image_path = os.path.join(IMAGE_ROOT, x["image_path"])
            # question = x['normal_caption'] if args.method=="baseline" else x['problem'] 
            question = x['question']
            # SYSTEM_PROMPT = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
            message = [
                # {"role": "system", "content": SYSTEM_PROMPT},
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
            # import pdb;pdb.set_trace()
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
            ground_truth = input_example['ground_truth']
            model_answer = extract_number_answer(original_output)
            
            # Create a result dictionary for this example
            result = {
                'question': input_example,
                'ground_truth': ground_truth,
                'model_output': original_output,
                'extracted_answer': model_answer
            }
            final_output.append(result)
            
            # Count correct answers
            if model_answer is not None and model_answer == ground_truth:
                correct_number += 1

        # Calculate and print accuracy
        accuracy = correct_number / len(data) * 100
        print(f"\nAccuracy: {accuracy:.2f}%")

        output_path = OUTPUT_PATH.format(DATASET=ds,MODEL=args.model_name.lower(), METHOD=args.method, STEPS=steps)
        with open(output_path, "w") as f:
            json.dump({"accuracy": accuracy, "results": final_output}, f, indent=2)
        print(f"Results saved to {output_path}")
        print("-" * 100)

if __name__=="__main__":
    args = parse_args()
    main(args)
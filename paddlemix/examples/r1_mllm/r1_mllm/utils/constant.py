SUPPORTED_MODELS = {
    "PPDocBee-2B-1129": "PaddleMIX/PPDocBee-2B-1129",
    "Qwen2-VL-2B-Instruct": "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen2-VL-7B-Instruct": "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-72B-Instruct": "Qwen/Qwen2.5-VL-72B-Instruct",
}


MODEL_MAPPING = {
    "PPDocBee-2B-1129": "Qwen2VLForConditionalGeneration",
    "Qwen2-VL-2B-Instruct": "Qwen2VLForConditionalGeneration",
    "Qwen2-VL-7B-Instruct": "Qwen2VLForConditionalGeneration",
    "Qwen2.5-VL-3B-Instruct": "Qwen2_5_VLForConditionalGeneration",
    "Qwen2.5-VL-7B-Instruct": "Qwen2_5_VLForConditionalGeneration",
    "Qwen2.5-VL-72B-Instruct": "Qwen2_5_VLForConditionalGeneration",
}

TOKENIZER_MAPPING = {
    "PPDocBee-2B-1129": "MIXQwen2Tokenizer",
    "Qwen2-VL-2B-Instruct": "MIXQwen2Tokenizer",
    "Qwen2-VL-7B-Instruct": "MIXQwen2Tokenizer",
    "Qwen2.5-VL-3B-Instruct": "MIXQwen2_5_Tokenizer",
    "Qwen2.5-VL-7B-Instruct": "MIXQwen2_5_Tokenizer",
    "Qwen2.5-VL-72B-Instruct": "MIXQwen2_5_Tokenizer",
}

VL_PROCESSOR_MAPPING = {
    "PPDocBee-2B-1129": "Qwen2VLProcessor",
    "Qwen2-VL-2B-Instruct": "Qwen2VLProcessor",
    "Qwen2-VL-7B-Instruct": "Qwen2VLProcessor",
    "Qwen2.5-VL-3B-Instruct": "Qwen2_5_VLProcessor",
    "Qwen2.5-VL-7B-Instruct": "Qwen2_5_VLProcessor",
    "Qwen2.5-VL-72B-Instruct": "Qwen2_5_VLProcessor",
}

IMAGE_PROCESSOR_MAPPING = {
    "PPDocBee-2B-1129": "Qwen2VLImageProcessor",
    "Qwen2-VL-2B-Instruct": "Qwen2VLImageProcessor",
    "Qwen2-VL-7B-Instruct": "Qwen2VLImageProcessor",
    "Qwen2.5-VL-3B-Instruct": "Qwen2_5_VLImageProcessor",
    "Qwen2.5-VL-7B-Instruct": "Qwen2_5_VLImageProcessor",
    "Qwen2.5-VL-72B-Instruct": "Qwen2_5_VLImageProcessor",
}

TEMPLATE_MAPPING = {
    "PPDocBee-2B-1129": "qwen2_vl",
    "Qwen2-VL-2B-Instruct": "qwen2_vl",
    "Qwen2-VL-7B-Instruct": "qwen2_vl",
    "Qwen2.5-VL-3B-Instruct": "qwen2_5_vl",
    "Qwen2.5-VL-7B-Instruct": "qwen2_5_vl",
    "Qwen2.5-VL-72B-Instruct": "qwen2_5_vl",
}

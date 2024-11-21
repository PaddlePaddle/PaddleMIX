import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from paddlenlp.transformers import Qwen2Tokenizer
from paddlemix.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from paddlemix.processors.qwen2_vl_processing import (
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    process_vision_info
)

class ServerConfig:
    def __init__(self, host="0.0.0.0", port=8001, model_path="Qwen/Qwen2-VL-2B-Instruct"):
        self.host = host
        self.port = port
        self.model_path = model_path

class GenerateRequest(BaseModel):
    messages: List[Dict[str, Any]]
    max_new_tokens: Optional[int] = 128

def create_app(config: ServerConfig):
    app = FastAPI()
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(config.model_path, dtype="bfloat16")
    image_processor = Qwen2VLImageProcessor()
    tokenizer = Qwen2Tokenizer.from_pretrained(config.model_path)
    processor = Qwen2VLProcessor(image_processor, tokenizer)

    @app.post("/generate")
    async def generate(request: GenerateRequest):
        try:
            image_inputs, video_inputs = process_vision_info(request.messages)
            question = request.messages[-1]["content"][-1]["text"]
            image_pad_token = "<|vision_start|><|image_pad|><|vision_end|>"
            text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{image_pad_token}{question}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pd"
            )
            
            generated_ids = model.generate(**inputs, max_new_tokens=request.max_new_tokens)
            output_text = processor.batch_decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            return {
                "error_code": 200,
                "error_msg": "success",
                "result": {
                    "response": {
                        "role": "assistant",
                        "utterance": output_text[0]
                    }
                }
            }
        except Exception as e:
            return {
                "error_code": 500,
                "error_msg": str(e),
                "result": None
            }
    
    return app

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--model-path", default="Qwen/Qwen2-VL-2B-Instruct",
                      help="Path to model weights")
    
    args = parser.parse_args()
    config = ServerConfig(host=args.host, port=args.port, model_path=args.model_path)
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)

if __name__ == "__main__":
    main()
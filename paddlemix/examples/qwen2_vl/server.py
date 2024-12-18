import os
import base64
from threading import Thread
from typing import List, Dict, Any, Optional, Union
import time
import json
import shortuuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from paddlenlp.transformers import Qwen2Tokenizer
from paddlenlp.generation import TextIteratorStreamer
from paddlemix.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from paddlemix.processors.qwen2_vl_processing import (
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    process_vision_info
)

from openai.types.chat.chat_completion import ChatCompletion, Choice as ChatCompletionChoice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChatStreamChoice, ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.95

class ServerConfig:
    def __init__(self, host="0.0.0.0", port=8001, model_path="Qwen/Qwen2-VL-2B-Instruct"):
        self.host = host
        self.port = port
        self.model_path = model_path

def create_app(config: ServerConfig):
    app = FastAPI()

    model = Qwen2VLForConditionalGeneration.from_pretrained(config.model_path, dtype="bfloat16")
    image_processor = Qwen2VLImageProcessor()
    tokenizer = Qwen2Tokenizer.from_pretrained(config.model_path)
    processor = Qwen2VLProcessor(image_processor, tokenizer)

    def process_messages(messages: List[Message]):
        system_message = ""
        user_message = ""
        
        # Convert messages to the format expected by process_vision_info
        converted_messages = []
        
        for msg in messages:    
            if msg.role == "system":
                if isinstance(msg.content, list):
                    for content in msg.content:
                        if isinstance(content, dict) and content.get("text"):
                            system_message = content["text"]
                            break
                else:
                    system_message = msg.content
                    
            elif msg.role == "user":
                # Handle user message
                converted_msg = {"role": msg.role}
                if isinstance(msg.content, list):
                    # Convert content to the format expected by process_vision_info
                    converted_content = []
                    for content in msg.content:
                        content_type = content.get("type")
                        
                        if content_type == "text":
                            user_message = content["text"]
                            converted_content.append(content)
                        elif content_type == "image_url":
                            converted_content.append({
                                "type": "image",
                                "image_url": content["image_url"]["url"]
                            })
                    converted_msg["content"] = converted_content
                    converted_messages.append(converted_msg)
                else:
                    user_message = msg.content
                    return system_message, user_message, [], []
        # Process images and videos using the existing function
        image_inputs, video_inputs = process_vision_info(converted_messages)
        return system_message, user_message, image_inputs, video_inputs


    def create_chat_completion_response(text: str, request_id: str, model: str):
        return ChatCompletion(
            id=f"chatcmpl-{shortuuid.random()}",
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=text
                    ),
                    finish_reason="stop"
                )
            ],
            created=int(time.time()),
            object="chat.completion",
            usage=CompletionUsage(
                prompt_tokens=0,  # You may want to calculate these
                completion_tokens=0,
                total_tokens=0
            )
        )

    def stream_generator(streamer, request_id: str, model: str):
        for new_text in streamer:
            if new_text:
                chunk = ChatCompletionChunk(
                    id=request_id,
                    model=model,
                    choices=[
                        ChatStreamChoice(
                            index=0,
                            delta=ChoiceDelta(content=new_text),
                            finish_reason=None
                        )
                    ],
                    created=int(time.time()),
                    object="chat.completion.chunk"
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
        
        # Send the final [DONE] message
        yield "data: [DONE]\n\n"

    @app.post("/v1/chat/completions/chat/completions") 
    async def create_chat_completion(request: ChatCompletionRequest):
        try:
            system_message, user_message, image_inputs, video_inputs = process_messages(request.messages)
            
            # Prepare the prompt
            image_pad_token = "<|vision_start|><|image_pad|><|vision_end|>"
            text = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{image_pad_token}{user_message}<|im_end|>\n<|im_start|>assistant\n"

            # Process inputs
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pd"
            )

            request_id = f"chatcmpl-{shortuuid.random()}"

            if request.stream:
                streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_special_tokens=True)
                
                generation_kwargs = {
                    "streamer": streamer,
                    "max_new_tokens": request.max_tokens,
                    "top_p": request.top_p,
                    "temperature": request.temperature,
                    "num_beams": 1,
                }
                
                generation_kwargs.update(inputs)
                
                thread = Thread(
                    target=model.generate,
                    kwargs=generation_kwargs
                )
                thread.start()

                return StreamingResponse(
                    stream_generator(streamer, request_id, request.model),
                    media_type="text/event-stream"
                )
            else:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    top_p=request.top_p,
                    temperature=request.temperature
                )
                
                output_text = processor.batch_decode(
                    generated_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

                response = create_chat_completion_response(
                    text=output_text,
                    request_id=request_id,
                    model=request.model
                )
                
                return JSONResponse(response.model_dump())

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": str(e),
                        "type": "internal_error",
                        "code": 500
                    }
                }
            )
    return app

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--model-path", default="Qwen/Qwen2-VL-2B-Instruct", help="Path to model weights")

    args = parser.parse_args()
    config = ServerConfig(host=args.host, port=args.port, model_path=args.model_path)
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)

if __name__ == "__main__":
    main()
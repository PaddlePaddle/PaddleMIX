import soundfile as sf
import json
import os

import paddle
from paddlemix.processors.qwen2_vl_processing import Qwen2VLImageProcessor
from paddlemix.models.qwen2_vl.mix_qwen2_tokenizer import MIXQwen2Tokenizer
from paddlemix.models.qwen2_5_omni import Qwen2_5OmniModel, Qwen2_5OmniProcessor, WhisperFeatureExtractor
from paddlemix.models.qwen2_5_omni.mm_process import process_mm_info

compute_dtype = "bfloat16"
model_path = "Qwen/Qwen2.5-Omni-7B_pd"
model = Qwen2_5OmniModel.from_pretrained(model_path, dtype="bfloat16").eval()
tokenizer = MIXQwen2Tokenizer.from_pretrained(model_path)
processor_config = json.load(open(os.path.join(model_path,"preprocessor_config.json")))
whisper_proc = WhisperFeatureExtractor(**processor_config)
img_proc = Qwen2VLImageProcessor(**processor_config)
processor = Qwen2_5OmniProcessor(img_proc,whisper_proc,tokenizer)

conversation = [
    {
        "role": "system",
        "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "input/draw.mp4",
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
        ],
    },
]

# set use audio in video
USE_AUDIO_IN_VIDEO = True

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
print(text)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pd", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
# inputs = inputs.to(model.device).to(model.dtype)

# convert dtype
inputs.pixel_values_videos = inputs.pixel_values_videos.astype(compute_dtype)
inputs.input_features = inputs.input_features.astype(compute_dtype)
# inputs.pixel_values_videos = inputs.pixel_values_videos.astype(compute_dtype)

# Inference: Generation of the output text and audio
with paddle.no_grad():
    text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)
sf.write(
    "output.wav",
    audio.reshape([-1]).detach().cpu().numpy(),
    samplerate=24000,
)
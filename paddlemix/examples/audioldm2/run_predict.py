# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
import paddle
from paddlenlp.trainer import PdArgumentParser
import os
import time
import soundfile as sf
from paddlemix.models.audioldm2.modeling import AudioLDM2Model
from paddlemix.models.audioldm2.encoders.phoneme_encoder import text as text 
import random
import numpy as np
import re

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def text2phoneme(data):
    return text._clean_text(re.sub(r'<.*?>', '', data), ["english_cleaners2"])

def text_to_filename(text):
    return text.replace(" ", "_").replace("'", "_").replace('"', "_")

CACHE = {
    "get_vits_phoneme_ids":{
        "PAD_LENGTH": 310,
        "_pad": '_',
        "_punctuation": ';:,.!?¡¿—…"«»“” ',
        "_letters": 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
        "_letters_ipa": "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ",
        "_special": "♪☎☒☝⚠"
    }
}

CACHE["get_vits_phoneme_ids"]["symbols"] = [CACHE["get_vits_phoneme_ids"]["_pad"]] + list(CACHE["get_vits_phoneme_ids"]["_punctuation"]) + list(CACHE["get_vits_phoneme_ids"]["_letters"]) + list(CACHE["get_vits_phoneme_ids"]["_letters_ipa"]) + list(CACHE["get_vits_phoneme_ids"]["_special"])
CACHE["get_vits_phoneme_ids"]["_symbol_to_id"] = {s: i for i, s in enumerate(CACHE["get_vits_phoneme_ids"]["symbols"])}

def get_vits_phoneme_ids_no_padding(phonemes):
    pad_token_id = 0
    pad_length = CACHE["get_vits_phoneme_ids"]["PAD_LENGTH"]
    _symbol_to_id = CACHE["get_vits_phoneme_ids"]["_symbol_to_id"]
    batchsize = len(phonemes)

    clean_text = phonemes[0] + "⚠"
    sequence = []

    for symbol in clean_text:
        if(symbol not in _symbol_to_id.keys()):
            print("%s is not in the vocabulary. %s" % (symbol, clean_text))
            symbol = "_"
        symbol_id = _symbol_to_id[symbol]
        sequence += [symbol_id]

    def _pad_phonemes(phonemes_list):
        return phonemes_list + [pad_token_id] * (pad_length-len(phonemes_list))
    
    sequence = sequence[:pad_length]

    return {"phoneme_idx": paddle.to_tensor(_pad_phonemes(sequence), dtype="int64").unsqueeze(0).expand([batchsize, -1])}


def make_batch_for_text_to_audio(text, transcription="", waveform=None, fbank=None, batchsize=1):
    text = [text] * batchsize
    if(transcription):
        transcription = text2phoneme(transcription)
    transcription = [transcription] * batchsize

    if batchsize < 1:
        print("Warning: Batchsize must be at least 1. Batchsize is set to .")

    if fbank is None:
        fbank = paddle.zeros(
            (batchsize, 1024, 64)
        )  # Not used, here to keep the code format
    else:
        fbank = paddle.to_tensor(fbank, dtype="float32")
        fbank = fbank.expand([batchsize, 1024, 64])
        assert fbank.shape[0] == batchsize

    stft = paddle.zeros((batchsize, 1024, 512))  # Not used
    phonemes = get_vits_phoneme_ids_no_padding(transcription)

    waveform = paddle.zeros((batchsize, 160000))  # Not used
    ta_kaldi_fbank = paddle.zeros((batchsize, 1024, 128))

    batch = {
        "text": text,  # list
        "fname": [text_to_filename(t) for t in text],  # list
        "waveform": waveform,
        "stft": stft,
        "log_mel_spec": fbank,
        "ta_kaldi_fbank": ta_kaldi_fbank,
    }
    batch.update(phonemes)
    return batch

def get_time():
    t = time.localtime()
    return time.strftime("%d_%m_%Y_%H_%M_%S", t)

def save_wave(waveform, savepath, name="outwav", samplerate=16000):
    if type(name) is not list:
        name = [name] * waveform.shape[0]

    for i in range(waveform.shape[0]):
        if waveform.shape[0] > 1:
            fname = "%s_%s.wav" % (
                    os.path.basename(name[i])
                    if (not ".wav" in name[i])
                    else os.path.basename(name[i]).split(".")[0],
                    i,
                )
        else:
            fname = "%s.wav" % os.path.basename(name[i]) if (not ".wav" in name[i]) else os.path.basename(name[i]).split(".")[0]
            # Avoid the file name too long to be saved
            if len(fname) > 255:
                fname = f"{hex(hash(fname))}.wav"

        path = os.path.join(
            savepath, fname
        )
        print("Save audio to %s" % path)
        sf.write(path, waveform[i, 0], samplerate=samplerate)

def read_list(fname):
    result = []
    with open(fname, "r", encoding="utf-8") as f:
        for each in f.readlines():
            each = each.strip('\n')
            result.append(each)
    return result

def text_to_audio(
        model,
        text,
        transcription="",
        seed=42,
        ddim_steps=200,
        duration=10,
        batchsize=1,
        guidance_scale=3.5,
        n_candidate_gen_per_text=3,
        latent_t_per_second=25.6,
    ):

        seed_everything(int(seed))
        waveform = None

        batch = make_batch_for_text_to_audio(text, transcription=transcription, waveform=waveform, batchsize=batchsize)

        model.latent_t_size = int(duration * latent_t_per_second)

        waveform = model(
            batch,
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            n_gen=n_candidate_gen_per_text,
            duration=duration,
        )

        return waveform


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    text: str = field(default="", metadata={"help": "Text prompt to the model for audio generation."})
    transcription: str = field(default="", metadata={"help": "Transcription for Text-to-Speech."})
    text_list: str = field(default="", metadata={"help": "A file (utf-8 encoded) that contains text prompt to the model for audio generation."})

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="haoheliu/audioldm2-full",
        metadata={"help": "Path to pretrained model or model identifier"},
    )
    save_path: str = field(
        default="./output",
        metadata={"help": "The path to save model output."},
    )
    device: str = field(
        default="gpu",
        metadata={"help": "The device for computation. If not specified, the script will automatically choose gpu."},
    )
    batchsize: int = field(
        default=1,
        metadata={"help": "Generate how many samples at the same time."},
    )
    ddim_steps: int = field(
        default=200,
        metadata={"help": "The sampling step for DDIM."},
    )
    guidance_scale: float = field(
        default=3.5,
        metadata={"help": "Guidance scale (Large => better quality and relavancy to text; Small => better diversity)."},
    )
    duration: float = field(
        default=10.0,
        metadata={"help": "The duration of the samples."},
    )
    n_candidate_gen_per_text: int = field(
        default=3,
        metadata={"help": "Automatic quality control. This number control the number of candidates (e.g., generate three audios and choose the best to show you). A Larger value usually lead to better quality with heavier computation."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Change this value (any integer number) will lead to a different generation result."},
    )

def main():
    parser = PdArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # process args
    text = data_args.text
    transcription = data_args.transcription
    text_list = data_args.text_list

    save_path = os.path.join(model_args.save_path, get_time())
    random_seed = model_args.seed
    duration = model_args.duration
    sample_rate = 16000
    latent_t_per_second = 25.6

    print("Warning: For AudioLDM2 we currently only support 10s of generation. Please use audioldm_48k or audioldm_16k_crossattn_t5 if you want a different duration.")
    duration = 10
    
    guidance_scale = model_args.guidance_scale
    n_candidate_gen_per_text = model_args.n_candidate_gen_per_text

    if transcription:
        if "speech" not in model_args.model_name_or_path:
            print("Warning: You choose to perform Text-to-Speech by providing the transcription. However you do not choose the correct model name (audioldm2-speech-gigaspeech or audioldm2-speech-ljspeech).")
            print("Warning: We will use audioldm2-speech-gigaspeech by default")
            model_args.model_name_or_path = "audioldm2-speech-gigaspeech"
        if not text:
            print("Warning: You should provide text as a input to describe the speaker. Use default (A male reporter is speaking).")
            text = "A female reporter is speaking full of emotion"
    
    if text_list:
        print("Generate audio based on the text prompts in %s" % text_list)
        prompt_todo = read_list(text_list)
    else: 
        prompt_todo = [text]
        
    # build audioldm2 model
    paddle.set_device(model_args.device)
    audioldm2 = AudioLDM2Model.from_pretrained(model_args.model_name_or_path)
    
    # predict
    os.makedirs(save_path, exist_ok=True)
    for text in prompt_todo:
        if "|" in text:
            text, name = text.split("|")
        else:
            name = text[:128]

        if transcription:
            name += "-TTS-%s" % transcription

        waveform = text_to_audio(
            audioldm2,
            text,
            transcription=transcription, # To avoid the model to ignore the last vocab
            seed=random_seed,
            duration=duration,
            guidance_scale=guidance_scale,
            ddim_steps=model_args.ddim_steps,
            n_candidate_gen_per_text=n_candidate_gen_per_text,
            batchsize=model_args.batchsize,
            latent_t_per_second=latent_t_per_second
        )
        
        save_wave(waveform, save_path, name=name, samplerate=sample_rate)

if __name__ == "__main__":
    main()

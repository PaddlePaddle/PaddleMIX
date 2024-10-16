import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
import paddle
import torch
import random
from tqdm import tqdm
from whisper.model import Whisper, ModelDimensions, Whisper_torch, AudioEncoder_torch2paddle
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram_torch, log_mel_spectrogram


checkpoint_dims = {
    'n_mels': 80,
    'n_vocab': 51865,
    'n_audio_ctx': 1500,
    'n_audio_state': 1280,
    'n_audio_head': 20,
    'n_audio_layer': 32,
    'n_text_ctx': 448,
    'n_text_state': 1280,
    'n_text_head': 20,
    'n_text_layer': 32,
}


def load_model_torch(path) -> Whisper_torch:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location="cpu")
    dims = ModelDimensions(**checkpoint_dims)
    print(dims)
    model = Whisper_torch(dims)
    # del model.decoder
    cut = len(model.encoder.blocks) // 4
    cut = -1 * cut
    del model.encoder.blocks[cut:]
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    # model.half()
    model.to(device)
    return model

def load_model(path) -> Whisper:
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = paddle.load(path)

    # dims = ModelDimensions(**checkpoint["dims"])
    dims = ModelDimensions(**checkpoint_dims)

    print(dims)
    model = Whisper(dims)
    # del model.decoder
    cut = len(model.encoder.blocks) // 4
    cut = -1 * cut
    del model.encoder.blocks[cut:]
    model.set_state_dict(checkpoint)
    model.eval()
    # model.half()
    # model.to(device)
    return model


# --------------------------------------------------------------

def pred_ppg(whisper: Whisper, wavPath, ppgPath):
    # wavPath = "~/Desktop/whisper-vits-svc/data_svc/waves-16k/435_all/000000.wav"
    wavPath = os.path.expanduser(wavPath)
    audio = load_audio(wavPath)
    audln = audio.shape[0]
    ppgln = audln // 320
    audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio)
    # return mel.cpu().numpy()
    with paddle.no_grad():
        ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().numpy()
        ppg = ppg[:ppgln,]  # [length, dim=1280]
        # print(ppg.shape)
        np.save(ppgPath, ppg, allow_pickle=False)

        return ppg


# def pred_ppg_torch(whisper: Whisper_torch, wavPath, ppgPath):
#     audio = load_audio(wavPath)
#     audln = audio.shape[0]
#     ppgln = audln // 320
#     audio = pad_or_trim(audio)
#     mel = log_mel_spectrogram_torch(audio).cuda()
#     # return mel.cpu().numpy()
#     with torch.no_grad():
#         ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
#         ppg = ppg[:ppgln,]  # [length, dim=1280]
#         # print(ppg.shape)
#         # np.save(ppgPath, ppg, allow_pickle=False)

#         return ppg

# --------------------------------------------------------------

if __name__ == "__main__":

    # path_tc = "~/Desktop/whisper-vits-svc/data_svc/whisper/434_all/000002.ppg.npy"
    # path_tc = os.path.expanduser(path_tc)
    # y_tc = np.load(path_tc)

    # path_pd = "~/Desktop/PaddleMIX/paddlemix/models/vits-svc/data_svc/whisper/434_all/000002.ppg.npy"
    # path_pd = os.path.expanduser(path_pd)
    # y_pd = np.load(path_pd)

    # print(
    #     "MAX:", abs(y_tc - y_pd).max(),
    #     "MEAN", abs(y_tc - y_pd).mean(),
    # )

    # ----------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", default="data_svc/waves-16k")
    parser.add_argument("-p", "--ppg", help="ppg", dest="ppg", default="data_svc/whisper")
    args = parser.parse_args()
    print(args.wav)
    print(args.ppg)

    os.makedirs(args.ppg, exist_ok=True)
    wavPath = args.wav
    ppgPath = args.ppg

    whisper = load_model(os.path.join("whisper_pretrain", "large-v2.pdparam"))

    # ------------ torch 旧模型转化 ------------
    # whisper_torch = load_model_torch(os.path.join("whisper_pretrain", "large-v2.pt"))
    # AudioEncoder_torch2paddle(whisper_torch.encoder, whisper.encoder)
    
    # paddle.save(
    #     whisper.state_dict(), "whisper_pretrain/large-v2.pdparam"
    # )

    # ------------ 测试 paddle 和 torch 模型 ------------
    # import numpy as np
    # x = np.random.rand(1, 80, 520).astype("float32")
    # x_tc = torch.from_numpy(x).cuda()
    # x_pd = paddle.to_tensor(x)

    # ---------------------------
    # dims = ModelDimensions(**checkpoint_dims)

    # whisper_torch = Whisper_torch(dims).cuda()
    # whisper = Whisper(dims)
    # ---------------------------

    # 转化模型参数
    # AudioEncoder_torch2paddle(whisper_torch.encoder, whisper.encoder)

    # y_tc = whisper_torch.encoder( x_tc ).detach().cpu().numpy()
    # y_pd = whisper.encoder( x_pd ).detach().cpu().numpy()

    # print(
    #     abs(y_tc - y_pd).max()
    # )
    # ----------------------------------------------------

    spkPaths = os.listdir(wavPath)
    # random.shuffle(spkPaths) # why shuffle
    spkPaths = sorted(spkPaths)

    for spks in spkPaths:
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{ppgPath}/{spks}", exist_ok=True)

            files = [f for f in os.listdir(f"./{wavPath}/{spks}") if f.endswith(".wav")]
            for file in tqdm(files, desc=f'Processing ppg {spks}'):
                if file.endswith(".wav"):
                    # print(file)
                    file = file[:-4]
                    path_wav = f"{wavPath}/{spks}/{file}.wav"
                    path_ppg = f"{ppgPath}/{spks}/{file}.ppg"
                    # if os.path.isfile(f"{path_ppg}.npy"):
                    #     continue
                    # y_tc = pred_ppg_torch(whisper_torch, path_wav, path_ppg)
                    y_pd = pred_ppg(whisper, path_wav, path_ppg)

                    # print(
                    #     abs(y_tc - y_pd).max()
                    # )

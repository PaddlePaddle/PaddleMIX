import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
import paddle
import librosa

from tqdm import tqdm
from hubert import hubert_model


def load_audio(file: str, sr: int = 16000):
    x, sr = librosa.load(file, sr=sr)
    return x


def load_model(path, device=None):
    model = hubert_model.hubert_soft(path)
    model.eval()
    # model.half()
    # model.to(device)
    return model


def pred_vec(model, wavPath, vecPath):
    feats = load_audio(wavPath)
    feats = paddle.to_tensor(feats)
    # feats = feats[None, None, :].half()
    feats = feats[None, None, :]
    with paddle.no_grad():
        with paddle.amp.auto_cast():
            vec = model.units(feats).squeeze().numpy()
        # print(vec.shape)   # [length, dim=256] hop=320
        np.save(vecPath, vec, allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", default="data_svc/waves-16k/")
    parser.add_argument("-v", "--vec", help="vec", dest="vec", default="data_svc/hubert")
    
    args = parser.parse_args()
    print(args.wav)
    print(args.vec)
    os.makedirs(args.vec, exist_ok=True)

    wavPath = args.wav
    vecPath = args.vec

    hubert = load_model( os.path.join("hubert_pretrain", "hubert-soft.pdparam") )

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{vecPath}/{spks}", exist_ok=True)

            files = [f for f in os.listdir(f"./{wavPath}/{spks}") if f.endswith(".wav")]
            for file in tqdm(files, desc=f'Processing vec {spks}'):
                file = file[:-4]
                pred_vec(hubert, f"{wavPath}/{spks}/{file}.wav", f"{vecPath}/{spks}/{file}.vec")

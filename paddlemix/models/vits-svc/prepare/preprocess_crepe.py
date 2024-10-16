import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import librosa
import paddle
import crepe
import argparse
from tqdm import tqdm

def paddle_randn_like(t):
    return paddle.randn(shape=t.shape, dtype=t.dtype)

def compute_f0(filename, save, device):

    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    # Load audio
    audio = paddle.to_tensor(audio)[None]
    audio = audio + paddle_randn_like(audio) * 0.001

    # Here we'll use a 10 millisecond hop length
    hop_length = 160
    # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
    # This would be a reasonable range for speech
    fmin = 50
    fmax = 1000
    # Select a model capacity--one of "tiny" or "full"
    model = "full"
    # Pick a batch size that doesn't cause memory errors on your gpu
    batch_size = 512
    # Compute pitch using first gpu
    pitch, periodicity = crepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model,
        batch_size=batch_size,
        device=device,
        return_periodicity=True,
    )
    # CREPE was not trained on silent audio. some error on silent need filter.pitPath
    periodicity = crepe.filter.median(periodicity, 7)
    pitch = crepe.filter.mean(pitch, 5)
    pitch[periodicity < 0.5] = 0
    pitch = pitch.squeeze(0)
    np.save(save, pitch, allow_pickle=False)


if __name__ == "__main__":

    # torch npy 路径
    tc_npy = "~/Desktop/whisper-vits-svc/data_svc/pitch/421_all/000002.pit.npy"
    tc_npy = os.path.expanduser(tc_npy)

    # paddle npy 路径
    pd_npy = "~/Desktop/PaddleMIX/paddlemix/models/vits-svc/data_svc/pitch/421_all/000002.pit.npy"
    pd_npy = os.path.expanduser(pd_npy)

    tc_arr = np.load(tc_npy)
    pd_arr = np.load(pd_npy)

    print(
        abs(tc_arr - pd_arr).max(),
        abs(tc_arr - pd_arr).mean(),
        tc_arr.std() - pd_arr.std(),
        tc_arr.mean() - pd_arr.mean(),
    )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", default="data_svc/waves-16k")
    parser.add_argument("-p", "--pit", help="pit", dest="pit", default="data_svc/pitch")

    args = parser.parse_args()
    print(args.wav)
    print(args.pit)

    os.makedirs(args.pit, exist_ok=True)
    wavPath = args.wav
    pitPath = args.pit

    device = None

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{pitPath}/{spks}", exist_ok=True)

            files = [f for f in os.listdir(f"./{wavPath}/{spks}") if f.endswith(".wav")]
            for file in tqdm(files, desc=f'Processing crepe {spks}'):
                file = file[:-4]
                compute_f0(f"{wavPath}/{spks}/{file}.wav", f"{pitPath}/{spks}/{file}.pit", device)

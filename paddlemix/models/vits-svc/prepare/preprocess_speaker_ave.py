import os
import paddle
import argparse
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':

    pd_file = "~/Desktop/PaddleMIX/paddlemix/models/vits-svc/data_svc/singer/422_all.spk.npy"
    pd_file = os.path.expanduser(pd_file)

    tc_file = "~/Desktop/whisper-vits-svc/data_svc/singer/422_all.spk.npy"
    tc_file = os.path.expanduser(pd_file)

    pd_npy = np.load(pd_file)
    tc_npy = np.load(tc_file)

    print(
        abs(pd_npy - tc_npy).max()
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_speaker", type=str, default="data_svc/speaker/")
    parser.add_argument("--dataset_singer", type=str, default="data_svc/singer")

    data_speaker = parser.parse_args().dataset_speaker
    data_singer = parser.parse_args().dataset_singer

    os.makedirs(data_singer, exist_ok=True)

    for speaker in os.listdir(data_speaker):
        subfile_num = 0
        speaker_ave = 0

        for file in tqdm(os.listdir(os.path.join(data_speaker, speaker)), desc=f"average {speaker}"):
            if not file.endswith(".npy"):
                continue
            source_embed = np.load(os.path.join(data_speaker, speaker, file))
            source_embed = source_embed.astype(np.float32)
            speaker_ave = speaker_ave + source_embed
            subfile_num = subfile_num + 1
        if subfile_num == 0:
            continue
        speaker_ave = speaker_ave / subfile_num

        np.save(os.path.join(data_singer, f"{speaker}.spk.npy"),
                speaker_ave, allow_pickle=False)

        # rewrite timbre code by average, if similarity is larger than cmp_val
        rewrite_timbre_code = False
        if not rewrite_timbre_code:
            continue
        cmp_src = paddle.to_tensor(speaker_ave, dtype="float32")
        cmp_num = 0
        cmp_val = 0.85
        for file in tqdm(os.listdir(os.path.join(data_speaker, speaker)), desc=f"rewrite {speaker}"):
            if not file.endswith(".npy"):
                continue
            cmp_tmp = np.load(os.path.join(data_speaker, speaker, file))
            cmp_tmp = cmp_tmp.astype(np.float32)
            cmp_tmp = paddle.to_tensor(cmp_tmp, dtype="float32")
            cmp_cos = paddle.nn.functional.cosine_similarity(cmp_src, cmp_tmp, axis=0)
            if (cmp_cos > cmp_val):
                cmp_num += 1
                np.save(os.path.join(data_speaker, speaker, file),
                        speaker_ave, allow_pickle=False)
        print(f"rewrite timbre for {speaker} with :", cmp_num)

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

import math

import numpy as np
import paddle


def img_psnr(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr = 20 * math.log10(1 / math.sqrt(mse))
    return psnr


def calculate_psnr(videos1, videos2):
    # videos [batch_size, timestamps, channel, h, w]

    assert videos1.shape == videos2.shape

    psnr_results = []

    for video_num in range(videos1.shape[0]):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        psnr_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w] numpy

            img1 = video1[clip_timestamp].numpy()
            img2 = video2[clip_timestamp].numpy()

            # calculate psnr of a video
            psnr_results_of_a_video.append(img_psnr(img1, img2))

        psnr_results.append(psnr_results_of_a_video)

    psnr_results = np.array(psnr_results)

    psnr = {}
    psnr_std = {}

    for clip_timestamp in range(len(video1)):
        psnr[clip_timestamp] = np.mean(psnr_results[:, clip_timestamp])
        psnr_std[clip_timestamp] = np.std(psnr_results[:, clip_timestamp])

    result = {
        "value": psnr,
        "value_std": psnr_std,
        "video_setting": video1.shape,
        "video_setting_name": "time, channel, heigth, width",
    }

    return result


def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 50
    CHANNEL = 3
    SIZE = 64
    videos1 = paddle.zeros(shape=[NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE])
    videos2 = paddle.zeros(shape=[NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE])
    paddle.set_device("gpu")

    import json

    result = calculate_psnr(videos1, videos2)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()

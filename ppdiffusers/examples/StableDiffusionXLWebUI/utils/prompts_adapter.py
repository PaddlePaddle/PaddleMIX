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

examples = """
gr.Examples(
    label='Face Adapter',
    examples=[[os.path.join(STATIC_DIR, 'img_ipadapter.png')]],
    inputs= [adapter_face_image]
)

gr.Examples(
    label='文生图',
    examples=[
                ["Beautiful girl, delicate, peerless face, delicate eyes, off-the-shoulder Sun Dress, light green, blossoms, detailed light effect rendering, high-definition picture, beautiful two-dimensional.",
                 -1,
                 'nsfw, 2 people',
                 4,30,1024,1800
                ],
                ["A full body shot of pretty Ju Jingyi, Mint-Green Wavy  Hair, solo, slim, perfect face, looking, Mint-Green Bloom Dress, masterpiece, highly detailed.",
                 -1,
                 'nsfw, 2 people',
                 4,30,1280,904
                ],
                ['Beautiful girl, delicate, peerless face, delicate eyes, off-the-shoulder Sun Dress, pink, cherry blossoms, detailed light effect rendering, high-definition picture, beautiful two-dimensional.',
                 -1,
                 'nsfw',
                 4,30,1024,1800
                ],
                ['masterpiece, 1 beautiful girl,  pinkish green dress, 8k super UHD',
                 '-1',
                 'nsfw, lowres,text,signature, watermark,username, blurry',
                 5,30,1440,720
                ],
                ['masterpiece, beautiful girl, yellowish green, delicate color of pink',
                 '-1',
                 '',
                 3,30,1560,760
                ],
                ['((best quality)), ((masterpiece)), ((scenic) scenery), in the sunlight, edge of the sea, slanting sun, extremely detailed 8K UHD',
                '-1',
                    'nsfw, lowres,text,signature, watermark,username, blurry',
                    4,30,1440,1024
                ],
    ],
    inputs=[
        text2img_prompt,
        text2img_seed,
        text2img_negative_prompt,
        text2img_cfg_scale,
        text2img_steps,
        text2img_height,
        text2img_width
    ]
)

gr.Examples(
    label='图生图',
    examples=[
                ['best quality, masterpiece, 1 beautiful girl, beautiful cherry lips, beautiful eyes, highlight hair, slim, extremely detail skin, skin pores, wearing pink dress), extremely detailed 8K UHD',
                'nsfw, lowres,text,signature, watermark,username, blurry',
                os.path.join(STATIC_DIR, 'text_691436129_ddpm_30_4_0_1712919483.194813.png'),
                '-1', 5, 30, 0.3, 1024, 8
                ],
    ],
    inputs=[
        img2img_prompt,
        img2img_negative_prompt,
        img2img_img, img2img_seed,
        img2img_cfg_scale,
        img2img_steps,
        img2img_strength,
        img2img_width,
        img2img_height
    ]
)

gr.Examples(
    label='局部绘图',
    examples=[
                ['{{best quality}}, {{masterpiece}}, {{ultra-detailed}},(highlight skin, extreme detail skin, skin pores, wearing long dress)',
                os.path.join(STATIC_DIR, 'inpaint.png'),
                '((watermark)) nsfw, lowres,text, signature, username, blurry'
                ]
    ],
    inputs=[
        inp2img_prompt,
        inp2img_img,
        inp2img_negative_prompt
    ]
)

gr.Examples(
    label='放大4倍',
    examples=[[os.path.join(STATIC_DIR, "bcsz_0.png")]],
    inputs= [svr_img_path]
)
"""

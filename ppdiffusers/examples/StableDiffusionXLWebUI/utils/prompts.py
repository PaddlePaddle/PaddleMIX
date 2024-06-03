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
    label='文生图',
    examples=[
                ['GenShin Impact roler NingGuang:0.8, solo, slim, perfect face, standing, bangs, off-shoulders, yellow:1.1, dress:1.2, masterpiece, best quality, highly detailed',
                 '-1',
                 'navel:1.4,' + negative_prompt,
                 5,18,1024,1024,"ddim"
                ],
                ['Nilou, GenShin, solo, sleek, curvy, smoldering eyes, bangs, off-shoulder, vibrant yellow dress, masterpiece.',
                 '-1',
                 'navel:1.4,' + negative_prompt,
                 5,18,1280,904,"ddim"
                ],
                ['A full body shot of pretty Ju Jingyi, Mint-Green Wavy  Hair, solo, slim, perfect face, looking, Mint-Green Bloom Dress, masterpiece, highly detailed.',
                 '-1',
                 'navel:1.4,' + negative_prompt,
                 5,18,1280,904,"ddim"
                ],
                ['Manga Style:0, Realistic Style:1.0, Anime:0, Watercolor Style:0, Abstract Style:0, NingGuang, solo, slim, perfect face, standing, bangs, off-shoulders, yellow:1.1, dress:1.2, masterpiece, best quality, highly detailed',
                 '-1',
                 'navel:1.4,' + negative_prompt,
                 5,18,1280,904,"ddim"
                ],
                ['Charming mature woman of GenShin`s roler Kamizato Ayaka, Wavy Curly lemon-yellow Hair, perfect face, standing, baggy attire Fringe peach-pink Skirt, masterpiece, best quality, highly detailed.',
                 '-1',
                 'navel:1.2,' + negative_prompt,
                 5,18,1280,904,"ddim"
                ],
                ['Charming mature woman of GenShin`s roler Kamizato Ayaka, Wavy Curly pink Hair, perfect face, standing, baggy attire Fringe vividly yellow Skirt, masterpiece, best quality, highly detailed.',
                 '-1',
                 'navel:1.2,' + negative_prompt,
                 5,18,1280,904,"ddim"
                ],
                ['Pretty GenShin Impact roler Nilou, Mint-Green Wavy Curly Hair, solo, perfect face, lying, baggy attire Mint-Green Bloom Dress, masterpiece, best quality, highly detailed.',
                 '-1',
                 'navel:1.4,' + negative_prompt,
                 5,18,1280,904,"ddim"
                ],
                ['Pretty GenShin Impact roler Nilou, Cherry Blossom Pink Wavy Curly Hair, solo, perfect face, lying, baggy attire Cherry Blossom Pink Bloom Dress, dress, masterpiece, best quality, highly detailed.',
                 '-1',
                 'navel:1.4,' + negative_prompt,
                 5,18,1280,904,"ddim"
                ],
                ['Charming mature woman of GenShin`s roler Sangonomiya Kokomi, Orange-Yellow Pink Wavy Curly Hair, solo, perfect face, standing, baggy attire Orange-Yellow Bowknot Dress, masterpiece, best quality, highly detailed.',
                 '-1',
                 'navel:1.4,' + negative_prompt,
                 5,18,1280,904,"ddim"
                ],
                ['Attractive GenShin Impact roler Yae Miko, Wavy Curly Hair, sleek, solo, slim, perfect face, standing, Maroon color, in baggy attire Lace Cut-out Dress, masterpiece, best quality, highly detailed.',
                 '-1',
                 'navel:1.4,' + negative_prompt,
                 5,18,1280,904,"ddim"
                ],
                ['Pretty GenShin Impact roler Yae Miko, Mint-Green Wavy Curly Hair, solo, slim, perfect face, lying, baggy attire Mint-Green Bloom Dress, masterpiece, best quality, highly detailed.',
                 '-1',
                 'navel:1.4,' + negative_prompt,
                 5,18,1280,904,"ddim"
                ],
                ['GenShin Impact roler Kamizato Ayaka, solo, woman, perfect face, reclining on the lake surface, bangs, off-shoulders, Dress, masterpiece, best quality, highly detailed',
                 '-1',
                 negative_prompt,
                 5,18,888,1184,"ddim"
                ],
                ['Attractive GenShin Impact roler Yoimiya, solo, slim, perfect face, standing, off-shoulders, 橘红色:1.1, QiPao:1.2, masterpiece, best quality, highly detailed.',
                 '-1',
                 negative_prompt,
                 5,18,1024,1024,"ddim"
                ],
                ['Attractive GenShin Impact roler Yoimiya, 温柔，甜美，迷人， solo, slim, perfect face, standing, off-shoulders, 鲜亮的黄色:1.2, Sun Dress:1.1, masterpiece, best quality, highly detailed.',
                 '-1',
                 negative_prompt,
                 5.1,18,1144,912,"ddim"
                ],
                ['GenShin Impact roler NingGuang, solo, woman, perfect face, standing, bangs, off-shoulders, fire red :1.1, dress:1.2, masterpiece, best quality, highly detailed',
                 '-1',
                 negative_prompt,
                 5, 18, 768, 1360,"ddim"
                ],
                ['GenShin Impact roler Sangonomiya Kokomi:1.0, solo, slim, perfect face, look at viewer, standing, bangs, single hair bun, long hair, grey eyes, parted lips, jewelry, off-shoulders, yellow:1.1, bikini:1.0, , masterpiece, best quality, highly detailed',
                 '-1',
                 negative_prompt,
                 5,30,1024,1024,"ddpm"
                ],
                ['Beautiful girl, delicate, peerless face, delicate eyes, off-the-shoulder dress, light delicate green-yellow, cherry blossoms, detailed light effect rendering, high-definition picture, beautiful two-dimensional.',
                 '-1',
                 'nsfw, bad hands, extral hands, extral fingers',
                 5,30,1024,1024,"ddpm"
                ],
                ["Beautiful girl, delicate, peerless face, delicate eyes, off-the-shoulder Sun Dress, light green, blossoms, detailed light effect rendering, high-definition picture, beautiful two-dimensional.",
                 -1,
                 'nsfw, 2 people',
                 4,30,1024,1800,"ddpm"
                ],
                ['Beautiful girl, delicate, peerless face, delicate eyes, off-the-shoulder Sun Dress, pink, cherry blossoms, detailed light effect rendering, high-definition picture, beautiful two-dimensional.',
                 -1,
                 'nsfw',
                 4,30,1024,1800,"ddpm"
                ],
                ['masterpiece, 1 beautiful girl,  pinkish green dress, 8k super UHD',
                 '-1',
                 'nsfw, lowres,text,signature, watermark,username, blurry',
                 5,30,1440,720,"ddpm"
                ],
                ['masterpiece, beautiful girl, yellowish green, delicate color of pink',
                 '-1',
                 '',
                 3,30,1560,760,"ddpm"
                ],
    ],
    inputs=[
        text2img_prompt,
        text2img_seed,
        text2img_negative_prompt,
        text2img_cfg_scale,
        text2img_steps,
        text2img_height,
        text2img_width,
        scheduler_type
    ]
)

gr.Examples(
    label='文生图-风景',
    examples=[
                ['((best quality)), ((masterpiece)), ((scenic) scenery), in the sunlight, edge of the sea, slanting sun, extremely detailed 8K UHD',
                 '-1',
                 'people, nsfw, lowres,text,signature, watermark,username, blurry',
                 4,30,768,1360,"ddpm"
                ],
                ['A breathtaking scenery comes to life with soft, gentle lighting that casts a magical glow. Vibrant colors burst forth, creating a vibrant and lively atmosphere that captivates the senses. Immerse yourself in this enchanting natural wonder, where every detail whispers serenity and harmony, GoPro camera with wide-angle perspective.',
                 '-1',
                 'people, nsfw, lowres,text,signature, watermark,username, blurry',
                 4,30,768,1360,"ddpm"
                ],
                ['A breathtaking scenery comes to life with soft, gentle lighting that casts a magical glow. Vibrant colors burst forth, creating a vibrant and lively atmosphere that captivates the senses. Immerse yourself in this enchanting natural wonder, where every detail whispers serenity and harmony.',
                 '-1',
                 'people, nsfw, lowres,text,signature, watermark,username, blurry',
                 4,30,768,1360,"ddpm"
                ],
                ['photorealistic, a breathtaking scenery, gentle lighting that casts a glow. Vibrant colors burst forth, creating a vibrant and lively atmosphere that captivates the senses. Enchant natural wonder, where every detail whispers serenity and harmony, GoPro camera with wide-angle perspective.',
                 '-1',
                 'people, nsfw, lowres,text,signature, watermark,username, blurry',
                 4,30,768,1360,"ddpm"
                ],
                ['a breathtaking scenery, gentle lighting that casts a glow. Vibrant colors burst forth, creating a vibrant and lively atmosphere that captivates the senses. Enchant natural wonder, where every detail whispers serenity and harmony, wide-angle perspective.',
                 '-1',
                 'people, nsfw, lowres,text,signature, watermark,username, blurry',
                 7,30,768,1360,"ddpm"
                ],
    ],
    inputs=[
        text2img_prompt,
        text2img_seed,
        text2img_negative_prompt,
        text2img_cfg_scale,
        text2img_steps,
        text2img_height,
        text2img_width,
        scheduler_type
    ]
)

gr.Examples(
    label='文生图',
    examples=[
                [
                 'Pony_Pencil-Xl-V1.0.2',
                 'madebyollin/sdxl-vae-fp16-fix',
                 'Charming mature woman of GenShin`s roler HuTao, Wavy Curly pink Hair, perfect face, solo, slim, standing, peach-pink, baggy attire Fringe-Skirt, masterpiece, best quality, highly detailed.',
                 '-1',
                 'short body, navel:1.2, nsfw:1.2,' + negative_prompt,
                 5,18,1280,904,"ddim"
                ],
                [
                 'Pony_Pencil-Xl-V1.0.2',
                 'madebyollin/sdxl-vae-fp16-fix',
                 'Charming mature woman of GenShin`s roler HuTao, Wavy Curly pink Hair, perfect face, solo, slim, standing, baggy attire cyan-blue Fringe-Skirt, masterpiece, best quality, highly detailed.',
                 '-1',
                 'short body, navel:1.2, nsfw:1.2,' + negative_prompt,
                 5,18,1280,904,"ddim"
                ],
                [
                 'Pony_Pencil-Xl-V1.0.2',
                 'madebyollin/sdxl-vae-fp16-fix',
                 'Charming mature woman of GenShin`s roler HuTao, Wavy curly-yellow Hair, perfect face, solo, slim, standing, curly-yellow top, baggy attire curly-white Fringe-Skirt, masterpiece, best quality, highly detailed.',
                 '-1',
                 'short body, navel:1.2, nsfw:1.2,' + negative_prompt,
                 5,18,1280,904,"ddim"
                ],
                [
                 'Pony_Pencil-Xl-V1.0.2',
                 'madebyollin/sdxl-vae-fp16-fix',
                 'Charming mature woman of GenShin`s roler HuTao, hands behind her back, Wavy curly-yellow Hair, perfect face, solo, slim, standing, curly-yellow Sweet-Camisole-Skirt, masterpiece, best quality, highly detailed.',
                 '-1',
                 'short body, navel:1.2, nsfw:1.2,' + negative_prompt,
                 5,18,1280,904,"ddim"
                ],
    ],
    inputs=[
        model_name,
        vae_dir,
        text2img_prompt,
        text2img_seed,
        text2img_negative_prompt,
        text2img_cfg_scale,
        text2img_steps,
        text2img_height,
        text2img_width,
        scheduler_type
    ]
)

gr.Examples(
    label='图生图',
    examples=[
                ['best quality, masterpiece, 1 beautiful girl, beautiful cherry lips, beautiful eyes, highlight hair, slim, extremely detail skin, skin pores, wearing pink dress), extremely detailed 8K UHD',
                'nsfw, lowres,text,signature, watermark,username, blurry',
                os.path.join(STATIC_DIR, 'adpt1.png'),
                '-1', 5, 30, 0.8, 1024, 800
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
    examples=[[os.path.join(STATIC_DIR, 'bcsz_0.png')]],
    inputs= [svr_img_path]
)
"""

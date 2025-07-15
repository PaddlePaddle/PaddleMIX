# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# code is heavily based on https://github.com/tianweiy/DMD2

import paddle

from ppdiffusers.transformers import CLIPTextModel, CLIPTextModelWithProjection


class SDXLTextEncoder(paddle.nn.Layer):
    def __init__(self, args, accelerator, dtype=paddle.float32) -> None:
        super().__init__()
        print("dddebug:", args.model_id)
        self.text_encoder_one = (
            CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder", revision=args.revision)
            .to(accelerator.device)
            .to(dtype=dtype)
        )

        self.text_encoder_two = (
            CLIPTextModelWithProjection.from_pretrained(
                args.model_id, subfolder="text_encoder_2", revision=args.revision
            )
            .to(accelerator.device)
            .to(dtype=dtype)
        )

        self.accelerator = accelerator

    def forward(self, batch):
        text_input_ids_one = batch["text_input_ids_one"].to(self.accelerator.device).squeeze(1)
        text_input_ids_two = batch["text_input_ids_two"].to(self.accelerator.device).squeeze(1)
        prompt_embeds_list = []

        for text_input_ids, text_encoder in zip(
            [text_input_ids_one, text_input_ids_two], [self.text_encoder_one, self.text_encoder_two]
        ):
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]

            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view([bs_embed, seq_len, -1])
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = paddle.concat(prompt_embeds_list, axis=-1)
        # use the second text encoder's pooled prompt embeds (overwrite in for loop)
        pooled_prompt_embeds = pooled_prompt_embeds.view([len(text_input_ids_one), -1])

        return prompt_embeds, pooled_prompt_embeds

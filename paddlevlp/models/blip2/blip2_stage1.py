# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The Salesforce Team Authors and The HuggingFace Team. All rights reserved.
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

import paddle
import paddle.distributed as dist
from paddlenlp.utils.log import logger
from paddlevlp.models.blip2.modeling import Blip2PretrainedModel,Blip2ForStage1ModelOutput
from .configuration import Blip2Config
from paddlevlp.models.blip2.modeling_utils import disabled_train, all_gather_with_grad, concat_all_gather

def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)

class Blip2Qformer(Blip2PretrainedModel):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    def __init__(self, config: Blip2Config):
        super().__init__(config)
        self.tokenizer = self.init_tokenizer()
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
           "eva_clip_g",config.vision_config.image_size,config.vision_config.dropout,
           config.vision_config.mp_degree if hasattr(config.vision_config, "mp_degree") else 1,
           gradient_checkpointing=config.vision_config.gradient_checkpointing if hasattr(config.vision_config, "gradient_checkpointing") else False
        )
        self.freeze_vit = config.freeze_vit
        if self.freeze_vit:
            # freeze vit except the post layer norm layer.
            for name, param in self.visual_encoder.named_parameters():
                if "post_layernorm" not in name:
                    param.stop_gradient = True
            self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logger.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            config.num_query_tokens, config.vision_config.hidden_size,config.qformer_config.cross_attention_frequency,
            mp_degree=config.qformer_config.mp_degree if hasattr(config.qformer_config, "mp_degree") else 1,
            gradient_checkpointing=config.qformer_config.gradient_checkpointing if hasattr(config.qformer_config, "gradient_checkpointing") else False
        )

        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        embed_dim=256
        max_txt_len=32
        #import pdb;pdb.set_trace()
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if '_query' in name:
                key_orig = name.replace('_query', '')
                param.copy_(state_dict[key_orig], False) ### problem
        self.vision_proj = paddle.nn.Linear(in_features=config.qformer_config
            .hidden_size, out_features=embed_dim)
        self.text_proj = paddle.nn.Linear(in_features=config.qformer_config.hidden_size, out_features=embed_dim)
        self.itm_head = paddle.nn.Linear(in_features=self.Qformer.config.hidden_size, out_features=2)
        self.temp = self.create_parameter(
            shape=(1, ), default_initializer=paddle.nn.initializer.Constant(value=0.07))
        self.max_txt_len = max_txt_len

    def forward(self, pixel_values,text_input):
        text = text_input

        image = pixel_values
        image_embeds = self.ln_vision(self.visual_encoder(image))

        image_atts = paddle.ones(image_embeds.shape[:-1], dtype="int64")
        query_tokens = self.query_tokens.expand(shape=[image_embeds.shape[0], -1, -1])

        query_output = self.Qformer.bert(query_embeds=query_tokens,
            encoder_hidden_states=image_embeds, encoder_attention_mask=
            image_atts, use_cache=True, return_dict=True)
        image_feats = paddle.nn.functional.normalize(x=self.vision_proj(
            query_output.last_hidden_state), axis=-1)

        text_tokens = self.tokenizer(text,
                                     padding='max_length',
                                     truncation=True,
                                     max_length=self.max_txt_len,
                                     return_attention_mask=True,
                                     return_tensors="pd"
                        )
        text_output = self.Qformer.bert(text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask, return_dict=True)
        text_feat = paddle.nn.functional.normalize(self.text_proj(
            text_output.last_hidden_state[:, 0, :]), axis=-1)

        ###============== Image-text Contrastive ===================###
        # image_feats_all = image_feats
        # text_feat_all = text_feat
        image_feats_all = concat_all_gather(image_feats)
        text_feat_all = concat_all_gather(text_feat)
        sim_q2t = paddle.matmul(image_feats.unsqueeze(axis=1), text_feat_all.unsqueeze(axis=-1)).squeeze()
        sim_i2t = sim_q2t.max(axis=-1)
        sim_i2t = sim_i2t / self.temp
        sim_t2q = paddle.matmul(x=text_feat.unsqueeze(axis=1).unsqueeze(
            axis=1), y=image_feats_all.transpose(perm=[0, 2, 1])).squeeze()
        sim_t2i = sim_t2q.max(axis=-1)
        sim_t2i = sim_t2i / self.temp

        rank = dist.get_rank()
        bs = image.shape[0]

        targets = paddle.linspace(start=rank * bs, stop=rank * bs + bs - 1,
            num=bs).astype(int)
        one_hot_label = paddle.nn.functional.one_hot(targets, num_classes=sim_i2t.shape[1])
        smooth_label = paddle.nn.functional.label_smooth(label=one_hot_label, epsilon=0.1)
        loss_itc = (paddle.nn.functional.cross_entropy(
            input=sim_i2t, label=smooth_label, soft_label=True) +
                    paddle.nn.functional.cross_entropy(
            input=sim_t2i, label=smooth_label, soft_label=True)) / 2
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        image_embeds_world = all_gather_with_grad(image_embeds)
        with paddle.no_grad():
            weights_t2i = paddle.nn.functional.softmax(x=sim_t2i, axis=1) + 0.0001
            weights_t2i_list= paddle.chunk(weights_t2i,chunks=paddle.distributed.get_world_size(), axis=-1)
            weights_t2i_list[rank].fill_diagonal_(value=0)
            weights_t2i = paddle.concat(weights_t2i_list,axis=-1)
            weights_i2t = paddle.nn.functional.softmax(x=sim_i2t, axis=1) + 0.0001
            weights_i2t_list= paddle.chunk(weights_i2t,chunks=paddle.distributed.get_world_size(), axis=-1)
            weights_i2t_list[rank].fill_diagonal_(value=0)
            weights_i2t = paddle.concat(weights_i2t_list,axis=-1)
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = paddle.multinomial(x=weights_t2i[b], num_samples=1).item(
                )
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = paddle.stack(x=image_embeds_neg, axis=0)
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = paddle.multinomial(x=weights_i2t[b], num_samples=1).item(
                )
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])
        text_ids_neg = paddle.stack(x=text_ids_neg, axis=0)
        text_atts_neg = paddle.stack(x=text_atts_neg, axis=0)
        text_ids_all = paddle.concat(x=[text_tokens.input_ids, text_tokens.
            input_ids, text_ids_neg], axis=0)
        text_atts_all = paddle.concat(x=[text_tokens.attention_mask,
            text_tokens.attention_mask, text_atts_neg], axis=0)
        query_tokens_itm = self.query_tokens.expand(shape=[text_ids_all.
            shape[0], -1, -1])
        query_atts_itm = paddle.ones(shape=query_tokens_itm.shape[:-1],
            dtype='int64')
        attention_mask_all = paddle.concat(x=[query_atts_itm, text_atts_all
            ], axis=1)
        image_embeds_all = paddle.concat(x=[image_embeds, image_embeds_neg,
            image_embeds], axis=0)
        image_atts_all = paddle.ones(shape=image_embeds_all.shape[:-1],
            dtype='int64')
        output_itm = self.Qformer.bert(text_ids_all, query_embeds=
            query_tokens_itm, attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all, encoder_attention_mask=
            image_atts_all, return_dict=True)
        vl_embeddings = output_itm.last_hidden_state[:, :query_tokens_itm.
            shape[1], :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(axis=1)

        itm_labels = paddle.concat([paddle.ones([bs], dtype='int64'),
            paddle.zeros([2 * bs], dtype='int64')], axis=0)
        loss_itm = paddle.nn.functional.cross_entropy(input=logits, label=
            itm_labels)
        ##================= Image Captioning ========================##

        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, (0)] = self.tokenizer.bos_token_id
        labels = masked_fill(decoder_input_ids, decoder_input_ids == self.tokenizer.pad_token_id, -100)
        query_atts = paddle.ones(shape=query_tokens.shape[:-1], dtype='int64')
        attention_mask = paddle.concat(x=[query_atts, text_tokens.
            attention_mask], axis=1)
        #import pdb;pdb.set_trace()
        lm_output = self.Qformer(decoder_input_ids, attention_mask=
            attention_mask, past_key_values=query_output.past_key_values,
            return_dict=True, labels=labels)
        loss_lm = lm_output.loss
        return Blip2ForStage1ModelOutput(loss=loss_itc + loss_itm + loss_lm, loss_itc=
            loss_itc, loss_itm=loss_itm, loss_lm=loss_lm)

    @paddle.no_grad()
    def generate(self, samples, use_nucleus_sampling=False, num_beams=3,
        max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples['image']
        image_embeds = self.ln_vision(self.visual_encoder(image))
        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, axis=0)
        else:
            num_beams = 1
        image_atts = paddle.ones(shape=image_embeds.shape[:-1], dtype='int64')
        model_kwargs = {'encoder_hidden_states': image_embeds,
            'encoder_attention_mask': image_atts}
        input_ids = paddle.empty(shape=[image.shape[0], 1], dtype='int64').fill_(value=self.tokenizer.bos_token_id)
        query_tokens = self.query_tokens.expand(shape=[image_embeds.shape[0
            ], -1, -1])
        outputs = self.Qformer.generate(input_ids=input_ids, query_embeds=
            query_tokens, max_length=max_length, min_length=min_length,
            num_beams=num_beams, do_sample=use_nucleus_sampling, top_p=
            top_p, eos_token_id=self.tokenizer.sep_token_id, pad_token_id=
            self.tokenizer.pad_token_id, **model_kwargs)
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens
            =True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = paddle.ones(shape=image_embeds.shape[:-1], dtype='int64')
        query_tokens = self.query_tokens.expand(shape=[image_embeds.shape[0
            ], -1, -1])
        query_output = self.Qformer.bert(query_embeds=query_tokens,
            encoder_hidden_states=image_embeds, encoder_attention_mask=
            image_atts, return_dict=True)
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask, return_dict=True)
        return text_output.last_hidden_state[:, (0), :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = paddle.ones(shape=image_inputs.shape[:-1], dtype='int64'
            )
        query_tokens = self.query_tokens.expand(shape=[image_inputs.shape[0
            ], -1, -1])
        query_atts = paddle.ones(shape=query_tokens.shape[:-1], dtype='int64'
            )
        attention_mask = paddle.concat(x=[query_atts, text_atts], axis=1)
        output_itm = self.Qformer.bert(text_ids, query_embeds=query_tokens,
            attention_mask=attention_mask, encoder_hidden_states=
            image_inputs, encoder_attention_mask=image_atts, return_dict=True)
        vl_embeddings = output_itm.last_hidden_state[:, :query_tokens.shape
            [1], :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, (1)].mean(axis=1)
        return itm_logit

    @paddle.no_grad()
    def extract_features(self, samples, mode='multimodal'):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get('image')
        caption = samples.get('text_input')
        assert mode in ['image', 'text', 'multimodal'
            ], "mode must be one of 'image', 'text', 'multimodal'"
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None
        if mode == 'image':
            assert image is not None, "Image is not provided for mode 'image' or 'multimodal'"
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image)
                    )
            image_embeds_frozen = image_embeds_frozen.astype(dtype='float32')
            image_atts = paddle.ones(shape=image_embeds_frozen.shape[:-1],
                dtype='int64')
            query_tokens = self.query_tokens.expand(shape=[
                image_embeds_frozen.shape[0], -1, -1])
            query_output = self.Qformer.bert(query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts, return_dict=True)
            image_embeds = query_output.last_hidden_state
            image_features = paddle.nn.functional.normalize(x=self.
                vision_proj(image_embeds), axis=-1)
        elif mode == 'text':
            assert caption is not None, "text input is None for mode 'text' or 'multimodal'"
            text = self.tokenizer(caption, return_tensors='pt', padding=True
                )
            text_output = self.Qformer.bert(text.input_ids, attention_mask=
                text.attention_mask, return_dict=True)
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = paddle.nn.functional.normalize(x=text_features,
                axis=-1)
        elif mode == 'multimodal':
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image)
                    )
            image_embeds_frozen = image_embeds_frozen.astype(dtype='float32')
            image_atts = paddle.ones(shape=image_embeds_frozen.shape[:-1],
                dtype='int64')
            query_tokens = self.query_tokens.expand(shape=[
                image_embeds_frozen.shape[0], -1, -1])
            query_atts = paddle.ones(shape=query_tokens.shape[:-1], dtype=
                'int64')
            text = self.tokenizer(caption, return_tensors='pt', padding=True
                )
            attention_mask = paddle.concat(x=[query_atts, text.
                attention_mask], axis=1)
            output = self.Qformer.bert(text.input_ids, query_embeds=
                query_tokens, attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts, return_dict=True)
            multimodal_embeds = output.last_hidden_state[:, :query_tokens.
                shape[1], :]
        return dict(image_embeds=image_embeds,
            image_embeds_proj=image_features, text_embeds=text_embeds,
            text_embeds_proj=text_features, multimodal_embeds=multimodal_embeds
            )

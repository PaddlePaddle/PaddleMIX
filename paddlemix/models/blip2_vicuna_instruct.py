import os
import sys
import paddle

from paddlemix.models.blip2 import (
    Blip2Config,
    Blip2ForConditionalGeneration,    
    Blip2QFormerConfig,
    Blip2VisionConfig,
)
from paddlemix.models.blip2.utils import create_tokenizer, load_model
from paddlemix.processors.blip_processing import (
    Blip2Processor,
    BlipImageProcessor,
    BlipTextProcessor,
)
from paddlemix.utils.log import logger
from paddlemix.models.blip2.Qformer import BertLMHeadModel

from paddlenlp.transformers import LlamaTokenizer     # transformers对应到paddlemix里面是什么
from paddlemix.models.blip2.configuration import Blip2Config
from paddlemix.models.blip2.eva_vit import VisionTransformer
from paddlenlp.transformers import T5Config


def __init__(
    self,
    config: Blip2Config,
    vit_model="eva_clip_g",
    img_size=224,
    drop_path_rate=0,
    use_grad_checkpoint=False,
    vit_precision="fp16",
    freeze_vit=True,
    num_query_token=32,
    llm_model="",
    prompt="",
    max_txt_len=128,
    max_output_txt_len=256,
    apply_lemmatizer=False,
    qformer_text_input=True
):
    self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
    self.Qformer = BertLMHeadModel(
    config=config.qformer_config,
    encoder_width=self.visual_encoder.num_features,
    train_in_satge1=False,
    text_hidden_size=self.language_model.hidden_size,
    model_config=config,  # in order to pass some parameters that are not available in config.qformer_config
)
    self.visual_encoder = VisionTransformer(config=config.vision_config)
    
    self.qformer_text_input = qformer_text_input
    tokenizer_name = config.qformer_config.tokenizer_name or "bert-base-uncased"
    self.tokenizer = self.init_tokenizer(tokenizer_name)

    
    


def generate(
    self,
    samples,
    use_nucleus_sampling=False,
    num_beams=5,
    max_length=256,
    min_length=1,
    top_p=0.9,
    repetition_penalty=1.5,
    length_penalty=1,
    num_captions=1,
    temperature=1,
):
        self.llm_tokenizer.padding_side = "left"

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        image = samples["image"]

        bs = image.size(0)
        
        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."
            
         # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]
            
        image_embeds = self.Qformer.ln_vision(self.visual_encoder(image))
        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, axis=0)
        else:
            num_beams = 1
            
        image_atts = paddle.ones(shape=image_embeds.shape[:-1], dtype="int64")
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
        
        input_ids = paddle.empty(shape=[image.shape[0], 1], dtype="int64").fill_(value=self.tokenizer.bos_token_id)
        query_tokens = self.Qformer.query_tokens.expand(shape=[image_embeds.shape[0], -1, -1])
        
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt,
                padding='max_length',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pd",
            )
            
            query_atts = paddle.ones(shape=query_tokens.shape[:-1], dtype="int64")
            Qformer_atts = paddle.concat(x=[query_atts, text_Qformer.attention_mask], axis=1)
            
        # for video data
        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:,:,j,:,:]
                with self.maybe_autocast():
                    frame_embeds = self.Qformer.ln_vision(self.visual_encoder(this_frame))
                frame_atts = paddle.ones(shape = frame_embeds.shape[:-1], dtype="int64")
                
                
                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                frame_inputs_llm = self.Qformer.language_projection(frame_query_output)
                frame_atts_llm = paddle.ones(shape = frame_inputs_llm.shape[:-1], dtype="int64")
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
        else:
            with self.maybe_autocast():
                image_embeds = self.Qformer.ln_vision(self.visual_encoder(image))
            image_atts = paddle.ones(shape = image_embeds.shape[:-1], dtype="int64")
            
            
            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            
            inputs_llm = self.Qformer.language_projection(query_output)
            atts_llm = paddle.ones(shape=inputs_llm.shape[:-1], dtype="int64")
            

        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pd"
        )

        
            
            



            
            
      
        
        

    


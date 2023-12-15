import string
import paddle
import paddle.nn as nn
import logging
import string

from paddlemix.models.blip2.modeling_utils import (
    all_gather_with_grad,
    concat_all_gather,
    disabled_train,
    masked_fill,
)

from paddlenlp.transformers import AutoTokenizer, OPTForCausalLM
from paddlemix.models.blip2.modeling import Blip2ForConditionalGeneration


class Blip2OptInstruct(Blip2ForConditionalGeneration):
    def init(
        self,
        llm_model="facebook/opt-2.7b",
        prompt="",
        freeze_vit=True,
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        qformer_text_input=True,
        
    ):
        super().__init__()
        
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
            
        
        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
        self.llm_model = OPTForCausalLM.from_pretrained(
            llm_model, dtype=paddle.float16
        )
        
        self.llm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.llm_tokenizer.add_special_tokens({"bos_token": "</s>"})
        self.llm_tokenizer.add_special_tokens({"eos_token": "</s>"})
        self.llm_tokenizer.add_special_tokens({"unk_token": "</s>"})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        
        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )
        
        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pd")
        
        if getattr(prompt_tokens, "attention_mask", None) is not None:
            self.prompt_length = prompt_tokens.attention_mask.sum(1)
        else:
            self.prompt_length = 0

        
        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input
        
    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                paddle.concat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                paddle.concat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = paddle.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = paddle.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len
    
    def forward(self, samples):
        image = samples["image"]
        
        with paddle.amp.auto_cast():
            image_embeds = self.Qformer.ln_vision(self.visual_encoder(image))
        image_atts = paddle.ones(image_embeds.shape[:-1], dtype=paddle.int64)
        
        bs = image.shape[0]
        query_tokens = self.Qformer.query_tokens.expand(image_embeds.shape[0], -1, -1)
        
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                samples["text_input"],
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pd",
            )
            query_atts = paddle.ones(query_tokens.shape[:-1], dtype=paddle.int64)
            Qformer_atts = paddle.concat([query_atts, text_Qformer.attention_mask], axis=1)

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

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:, : query_tokens.size(1), :])
        atts_llm = paddle.ones(inputs_llm.shape[:-1], dtype=paddle.int64)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "left"
        
        text_input_tokens = self.llm_tokenizer(
            samples["text_input"],
            return_tensors="pd",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        )
        
        self.llm_tokenizer.truncation_side = "right"
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples["text_output"]],
            return_tensors="pd",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        )
        
        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )
        
        # do not apply loss to the padding
        targets = llm_tokens["input_ids"].masked_fill(llm_tokens["input_ids"] == self.llm_tokenizer.pad_token_id, -100)

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = paddle.ones(atts_llm.shape, dtype=paddle.int64).fill_(-100)
        targets = paddle.concat([empty_targets, targets], axis=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens["input_ids"])
        inputs_embeds = paddle.concat([inputs_llm, inputs_embeds], axis=1)
        attention_mask = paddle.concat([atts_llm, llm_tokens["attention_mask"]], axis=1)
        
        with paddle.amp.auto_cast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss

        return {"loss": loss}
    
    @paddle.no_grad()
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

        bs = image.shape[0]

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(", ".join(samples["ocr_tokens"][i][:30])) for i, p in enumerate(prompt)]

        query_tokens = self.Qformer.query_tokens.expand([bs, -1, -1])
        if self.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]

            text_Qformer = self.tokenizer(
                prompt,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pd",
            )
            query_atts = paddle.ones(query_tokens.shape[:-1], dtype=paddle.int64)
            Qformer_atts = paddle.concat([query_atts, text_Qformer.attention_mask], axis=1)

        # For video data
        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:, :, j, :, :]
                with paddle.amp.auto_cast():
                    frame_embeds = self.Qformer.ln_vision(self.visual_encoder(this_frame))
                frame_atts = paddle.ones(frame_embeds.shape[:-1], dtype=paddle.int64)

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
                frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:, : query_tokens.size(1), :])
                frame_atts_llm = paddle.ones(frame_inputs_llm.shape[:-1], dtype=paddle.int64)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = paddle.concat(inputs_llm, axis=1)
            atts_llm = paddle.concat(atts_llm, axis=1)
        else:
            with paddle.amp.auto_cast():
                image_embeds = self.Qformer.ln_vision(self.visual_encoder(image))
            image_atts = paddle.ones(image_embeds.shape[:-1], dtype=paddle.int64)

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

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:, : query_tokens.size(1), :])
            atts_llm = paddle.ones(inputs_llm.shape[:-1], dtype=paddle.int64)

        llm_tokens = self.llm_tokenizer(prompt, padding="longest", return_tensors="pd")

        with paddle.amp.auto_cast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = paddle.concat([inputs_llm, inputs_embeds], axis=1)
            attention_mask = paddle.concat([atts_llm, llm_tokens.attention_mask], axis=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        outputs[outputs == 0] = 2  # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text
    
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if "ocr_tokens" in samples:
                    text_input = [
                        prompt.format(", ".join(samples["ocr_tokens"][i][:30]), samples["text_input"][i])
                        for i in range(len(samples["text_input"]))
                    ]
                elif "choices" in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [
                            f"({string.ascii_lowercase[j]}) {ch}" for j, ch in enumerate(samples["choices"][i])
                        ]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["text_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples, num_beams=num_beams, max_length=max_len, min_length=min_len, length_penalty=length_penalty
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)

        return output_text
    
    def predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        # If candidates is a list of lists, each sample has its candidates, then we need to iterate one by one
        if type(candidates[0]) == list:
            results = []

            for i in range(samples["image"].shape[0]):
                this_sample = {
                    "image": samples["image"][i].unsqueeze(0),
                    "prompt": samples["prompt"],
                }

                if "text_input" in samples.keys():
                    this_sample["text_input"] = [samples["text_input"][i]]

                if "context" in samples.keys():
                    this_sample["context"] = [samples["context"][i]]

                if "history" in samples.keys():
                    this_sample["history"] = [samples["history"][i]]

                if "caption" in samples.keys():
                    this_sample["caption"] = [samples["caption"][i]]

                this_result = self._predict_class(this_sample, candidates[i], n_segments)
                results.append(this_result)

            try:
                results = paddle.concat(results, axis=0)
            except:
                results = [res.tolist()[0] for res in results]

            return results

        return self._predict_class(samples, candidates, n_segments)

    def _predict_class(
        self,
        samples,
        candidates,
        n_segments=1,
    ):
        image = samples["image"]
        prompt = samples["prompt"]

        bs = image.shape[0]

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        if "text_input" in samples.keys():
            if type(samples["text_input"][0]) == list:
                prompt = [prompt[i].format(*samples["text_input"][i]) for i in range(len(prompt))]
            else:
                prompt = [prompt[i].format(samples["text_input"][i]) for i in range(len(prompt))]

        # scienceqa
        if "context" in samples.keys() and samples["context"] != "":
            prompt = [f'context: {samples["context"][i]}. {prompt[i]}' for i in range(len(prompt))]

        # visual dialog
        if "history" in samples.keys() and samples["history"][0] != "":
            prompt = [f'dialog history: {samples["history"][i]}\n{prompt[i]}' for i in range(len(prompt))]

        if "caption" in samples.keys() and samples["caption"][0] != "":
            prompt = [f'This image has the caption "{samples["caption"][i]}". {prompt[i]}' for i in range(len(prompt))]

        query_tokens = self.Qformer.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                prompt, padding="longest", truncation=True, max_length=self.max_txt_len, return_tensors="pd"
            )
            query_atts = paddle.ones(query_tokens.shape[:-1], dtype=paddle.int64)
            Qformer_atts = paddle.concat([query_atts, text_Qformer.attention_mask], axis=1)

        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:, :, j, :, :]
                with self.maybe_autocast():
                    frame_embeds = self.Qformer.ln_vision(self.visual_encoder(this_frame))
                    frame_atts = paddle.ones(frame_embeds.shape[:-1], dtype=paddle.int64)

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

                frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:, : query_tokens.size(1), :])
                frame_atts_llm = paddle.ones(frame_inputs_llm.shape[:-1], dtype=paddle.int64)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = paddle.concat(inputs_llm, axis=1)
            atts_llm = paddle.concat(atts_llm, axis=1)
        else:
            with paddle.amp.auto_cast():
                image_embeds = self.Qformer.ln_vision(self.visual_encoder(image))
            image_atts = paddle.ones(image_embeds.shape[:-1], dtype=paddle.int64)

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

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:, : query_tokens.size(1), :])
            atts_llm = paddle.ones(inputs_llm.shape[:-1], dtype=paddle.int64)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "left"
        text_input_tokens = self.llm_tokenizer(
            prompt,
            return_tensors="pd",
            padding="longest",
            # truncation=True,
            # max_length=self.max_txt_len,
        )

        empty_targets = paddle.ones(atts_llm.shape, dtype=paddle.int64).fill_(-100)

        # self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "right"
        n_cands = len(candidates)
        with self.maybe_autocast(dtype=paddle.bfloat16):
            all_losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len

                this_output_tokens = self.llm_tokenizer(
                    candidates[start_i:end_i],
                    return_tensors="pd",
                    padding="longest",
                    # truncation=True,
                    # max_length=self.max_output_txt_len,
                )

                this_input_tokens_ids = text_input_tokens.input_ids.repeat_interleave(seg_len, axis=0)
                this_input_tokens_atts = text_input_tokens.attention_mask.repeat_interleave(seg_len, axis=0)

                this_output_tokens_ids = this_output_tokens.input_ids.repeat(bs, 1)
                this_output_tokens_atts = this_output_tokens.attention_mask.repeat(bs, 1)

                this_llm_tokens, this_input_targets_len = self.concat_text_input_output(
                    this_input_tokens_ids, this_input_tokens_atts, this_output_tokens_ids, this_output_tokens_atts
                )

                this_llm_input_ids = this_llm_tokens["input_ids"]
                this_llm_atts = this_llm_tokens["attention_mask"]
                # this_llm_input_ids = paddle.concat([this_input_tokens_ids, this_output_tokens_ids], axis=1)
                # this_llm_atts = paddle.concat([this_input_tokens_atts, this_output_tokens_atts], axis=1)

                inputs_embeds = self.llm_model.get_input_embeddings()(this_llm_input_ids)
                inputs_embeds = paddle.concat([inputs_llm.repeat_interleave(seg_len, axis=0), inputs_embeds], axis=1)
                attention_mask = paddle.concat([atts_llm.repeat_interleave(seg_len, axis=0), this_llm_atts], axis=1)

                this_targets = this_llm_input_ids.masked_fill(
                    this_llm_input_ids == self.llm_tokenizer.pad_token_id, -100
                )
                # this_targets[:, :this_input_tokens_ids.size(1)] = -100
                for i, l in enumerate(this_input_targets_len):
                    this_targets[i][:l] = -100

                this_targets = paddle.concat([empty_targets.repeat_interleave(seg_len, axis=0), this_targets], axis=1)

                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )

                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                # output_class_ranks = paddle.argsort(loss, axis=-1)
                all_losses.append(loss)

            all_losses = paddle.concat(all_losses, axis=-1)
            output_class_ranks = paddle.argsort(all_losses, axis=-1)

        return output_class_ranks
    
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
    
    @classmethod
    def from_config(cls, cfg):
        cfg = dict()
        cfg["image_size"] = 224
        cfg["drop_path_rate"] = 0
        cfg["use_grad_checkpoint"] = False
        cfg["vit_precision"] = "fp32"
        cfg["freeze_vit"] = True
        cfg["num_query_token"] = 32
        cfg["llm_model"] = "facebook/opt-2.7b"

        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llm_model = cfg.get("llm_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_text_input = cfg.get("qformer_text_input", True)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            llm_model=llm_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            qformer_text_input=qformer_text_input,
        )

        return model










        

        


        
        
        
        
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

import importlib

import paddle
import paddle.nn as nn
from paddlenlp.transformers import GPTModel


class Sequence2AudioMAE(nn.Layer):
    def __init__(
        self,
        base_learning_rate,
        sequence_gen_length,
        sequence_input_key,
        sequence_input_embed_dim,
        cond_stage_config,
        optimizer_type="AdamW",
        use_warmup=True,
        use_ar_gen_loss=False,
        use_audiomae_linear=False,
        target_tokens_mask_ratio=0.0,
        random_mask_ratio=False,
        **kwargs
    ):
        super().__init__()
        assert use_audiomae_linear is False
        self.random_mask_ratio = random_mask_ratio
        self.learning_rate = base_learning_rate
        self.cond_stage_config = cond_stage_config
        self.use_audiomae_linear = use_audiomae_linear
        self.optimizer_type = optimizer_type
        self.use_warmup = use_warmup
        self.use_ar_gen_loss = use_ar_gen_loss
        # Even though the LDM can be conditioned on multiple pooling rate
        # Our model always predict the highest pooling rate

        self.mae_token_num = sequence_gen_length
        self.sequence_input_key = sequence_input_key
        self.sequence_input_embed_dim = sequence_input_embed_dim
        self.target_tokens_mask_ratio = target_tokens_mask_ratio

        self.start_of_sequence_tokens = nn.Embedding(32, 768)
        self.end_of_sequence_tokens = nn.Embedding(32, 768)

        self.input_sequence_embed_linear = nn.LayerList([])
        self.initial_learning_rate = None

        for dim in self.sequence_input_embed_dim:
            self.input_sequence_embed_linear.append(nn.Linear(dim, 768))

        self.cond_stage_models = nn.LayerList([])
        self.instantiate_cond_stage(cond_stage_config)
        self.initialize_param_check_toolkit()

        self.model = GPTModel.from_pretrained("gpt2")

        self.loss_fn = nn.L1Loss()

        self.logger_save_dir = None
        self.logger_exp_name = None
        self.logger_exp_group_name = None
        self.logger_version = None

    def set_log_dir(self, save_dir, exp_group_name, exp_name):
        self.logger_save_dir = save_dir
        self.logger_exp_group_name = exp_group_name
        self.logger_exp_name = exp_name

    def cfg_uncond(self, batch_size):
        unconditional_conditioning = {}
        for key in self.cond_stage_model_metadata:
            model_idx = self.cond_stage_model_metadata[key]["model_idx"]
            unconditional_conditioning[key] = self.cond_stage_models[model_idx].get_unconditional_condition(batch_size)
        assert (
            "crossattn_audiomae_pooled" in unconditional_conditioning.keys()
        ), "The module is not initialized with AudioMAE"
        unconditional_conditioning["crossattn_clap_to_audiomae_feature"] = unconditional_conditioning[
            "crossattn_audiomae_pooled"
        ]
        return unconditional_conditioning

    def add_sos_eos_tokens(self, _id, sequence, attn_mask):
        batchsize = sequence.shape[0]

        new_attn_mask_step = paddle.ones((batchsize, 1))
        key_id = paddle.to_tensor([_id])

        # Add two more steps to attn mask
        new_attn_mask = paddle.concat([new_attn_mask_step, attn_mask, new_attn_mask_step], axis=1)

        # Add two more tokens in the sequence
        sos_token = self.start_of_sequence_tokens(key_id).expand([batchsize, 1, -1])
        eos_token = self.end_of_sequence_tokens(key_id).expand([batchsize, 1, -1])
        new_sequence = paddle.concat([sos_token, sequence, eos_token], axis=1)
        return new_sequence, new_attn_mask

    def truncate_sequence_and_mask(self, sequence, mask, max_len=512):
        if sequence.shape[1] > max_len:
            print(
                "The input sequence length to GPT-2 model is too long:",
                sequence.shape[1],
            )
            return sequence[:, :max_len], mask[:, :max_len]
        else:
            return sequence, mask

    def get_input_sequence_and_mask(self, cond_dict):
        input_embeds = None
        input_embeds_attn_mask = None
        for _id, sequence_key in enumerate(self.sequence_input_key):
            assert sequence_key in cond_dict.keys(), "Invalid sequence key %s" % sequence_key
            cond_embed = cond_dict[sequence_key]
            if isinstance(cond_embed, list):
                assert (
                    len(cond_embed) == 2
                ), "The crossattn returned list should have length 2, including embed and attn_mask"
                item_input_embeds, item_attn_mask = cond_embed

                item_input_embeds = self.input_sequence_embed_linear[_id](item_input_embeds)

                item_input_embeds, item_attn_mask = self.add_sos_eos_tokens(_id, item_input_embeds, item_attn_mask)

                if input_embeds is None and input_embeds_attn_mask is None:
                    input_embeds, input_embeds_attn_mask = (
                        item_input_embeds,
                        item_attn_mask,
                    )
                else:
                    input_embeds = paddle.concat(
                        [input_embeds, item_input_embeds], axis=1
                    )  # The 1-st dimension is time steps
                    input_embeds_attn_mask = paddle.concat(
                        [input_embeds_attn_mask, item_attn_mask], axis=1
                    )  # The 1-st dimension is time steps
            else:
                assert isinstance(cond_embed, paddle.Tensor)
                cond_embed = self.input_sequence_embed_linear[_id](cond_embed)
                attn_mask = paddle.ones((cond_embed.shape[0], cond_embed.shape[1]))

                item_input_embeds, item_attn_mask = self.add_sos_eos_tokens(_id, cond_embed, attn_mask)

                if input_embeds is None and input_embeds_attn_mask is None:
                    input_embeds, input_embeds_attn_mask = (
                        item_input_embeds,
                        item_attn_mask,
                    )
                else:
                    input_embeds, input_embeds_attn_mask = paddle.concat(
                        [input_embeds, item_input_embeds], axis=1
                    ), paddle.concat([input_embeds_attn_mask, item_attn_mask], axis=1)

        assert input_embeds is not None and input_embeds_attn_mask is not None

        input_embeds, input_embeds_attn_mask = self.truncate_sequence_and_mask(
            input_embeds, input_embeds_attn_mask, int(1024 - self.mae_token_num)
        )
        cond_sequence_end_time_idx = input_embeds.shape[1]  # The index that we start to collect the output embeds

        return input_embeds, input_embeds_attn_mask, cond_sequence_end_time_idx

    def mask_target_sequence(self, target_embeds, target_embeds_attn_mask):
        time_seq_mask = None
        if self.target_tokens_mask_ratio > 1e-4:
            batchsize, time_seq_len, embed_dim = target_embeds.shape
            _, time_seq_len = target_embeds_attn_mask.shape
            # Generate random mask
            if self.random_mask_ratio:
                mask_ratio = paddle.rand((1,)).item() * self.target_tokens_mask_ratio
            else:
                mask_ratio = self.target_tokens_mask_ratio

            time_seq_mask = paddle.rand((batchsize, time_seq_len)) > mask_ratio

            # Mask the target embedding
            target_embeds = target_embeds * time_seq_mask.unsqueeze(-1)
            target_embeds_attn_mask = target_embeds_attn_mask * time_seq_mask
        return target_embeds, target_embeds_attn_mask, time_seq_mask

    def generate_partial(self, batch, cond_dict=None, no_grad=False):
        if cond_dict is None:
            cond_dict = self.get_input(batch)

        print("Generate partially prompted audio with in-context learning")

        target_embeds, target_embeds_attn_mask = (
            cond_dict["crossattn_audiomae_pooled"][0],
            cond_dict["crossattn_audiomae_pooled"][1],
        )

        target_time_steps = target_embeds.shape[1]

        (
            input_embeds,
            input_embeds_attn_mask,
            cond_sequence_end_time_idx,
        ) = self.get_input_sequence_and_mask(cond_dict)

        model_input = paddle.concat([input_embeds, target_embeds[:, : target_time_steps // 4, :]], axis=1)
        model_input_mask = paddle.concat(
            [
                input_embeds_attn_mask,
                target_embeds_attn_mask[:, : target_time_steps // 4],
            ],
            axis=1,
        )

        steps = self.mae_token_num

        for _ in range(3 * steps // 4):
            output = self.model(inputs_embeds=model_input, attention_mask=model_input_mask, return_dict=True)[
                "last_hidden_state"
            ]
            # Update the model input
            model_input = paddle.concat([model_input, output[:, -1:, :]], axis=1)
            # Update the attention mask
            attention_mask_new_step = paddle.ones((model_input_mask.shape[0], 1))
            model_input_mask = paddle.concat([model_input_mask, attention_mask_new_step], axis=1)

        output = model_input[:, cond_sequence_end_time_idx:]

        return output, cond_dict

    def generate(self, batch, cond_dict=None, no_grad=False):
        if cond_dict is None:
            cond_dict = self.get_input(batch)

        (
            input_embeds,
            input_embeds_attn_mask,
            cond_sequence_end_time_idx,
        ) = self.get_input_sequence_and_mask(cond_dict)
        model_input = input_embeds
        model_input_mask = input_embeds_attn_mask

        steps = self.mae_token_num

        for _ in range(steps):
            output = self.model(inputs_embeds=model_input, attention_mask=model_input_mask, return_dict=True)[
                "last_hidden_state"
            ]
            # Update the model input
            model_input = paddle.concat([model_input, output[:, -1:, :]], axis=1)
            # Update the attention mask
            attention_mask_new_step = paddle.ones((model_input_mask.shape[0], 1))
            model_input_mask = paddle.concat([model_input_mask, attention_mask_new_step], axis=1)

        return model_input[:, cond_sequence_end_time_idx:], cond_dict

    def get_input_item(self, batch, k):
        fname, text, waveform, stft, fbank = (
            batch["fname"],
            batch["text"],
            batch["waveform"],
            batch["stft"],
            batch["log_mel_spec"],
        )
        ret = {}

        ret["fbank"] = paddle.cast(fbank.unsqueeze(1), dtype="float32")
        ret["stft"] = paddle.cast(stft, dtype="float32")
        ret["waveform"] = paddle.cast(waveform, dtype="float32")
        ret["text"] = list(text)
        ret["fname"] = fname

        for key in batch.keys():
            if key not in ret.keys():
                ret[key] = batch[key]

        return ret[k]

    def get_input(self, batch):
        cond_dict = {}
        if len(self.cond_stage_model_metadata.keys()) > 0:
            unconditional_cfg = False

            for cond_model_key in self.cond_stage_model_metadata.keys():
                cond_stage_key = self.cond_stage_model_metadata[cond_model_key]["cond_stage_key"]

                # The original data for conditioning
                xc = self.get_input_item(batch, cond_stage_key)
                if type(xc) == paddle.Tensor:
                    xc = xc

                c = self.get_learned_conditioning(xc, key=cond_model_key, unconditional_cfg=unconditional_cfg)
                cond_dict[cond_model_key] = c

        return cond_dict

    def instantiate_cond_stage(self, config):
        self.cond_stage_model_metadata = {}

        for i, cond_model_key in enumerate(config.keys()):
            model = instantiate_from_config(config[cond_model_key])
            self.cond_stage_models.append(model)
            self.cond_stage_model_metadata[cond_model_key] = {
                "model_idx": i,
                "cond_stage_key": config[cond_model_key]["cond_stage_key"],
                "conditioning_key": config[cond_model_key]["conditioning_key"],
            }

    def get_learned_conditioning(self, c, key, unconditional_cfg):
        assert key in self.cond_stage_model_metadata.keys()

        # Classifier-free guidance
        if not unconditional_cfg:
            c = self.cond_stage_models[self.cond_stage_model_metadata[key]["model_idx"]](c)
        else:
            if isinstance(c, paddle.Tensor):
                batchsize = c.shape[0]
            elif isinstance(c, list):
                batchsize = len(c)
            else:
                raise NotImplementedError()
            c = self.cond_stage_models[self.cond_stage_model_metadata[key]["model_idx"]].get_unconditional_condition(
                batchsize
            )

        return c

    def initialize_param_check_toolkit(self):
        self.tracked_steps = 0
        self.param_dict = {}

    def statistic_require_grad_tensor_number(self, module, name=None):
        requires_grad_num = 0
        total_num = 0
        require_grad_tensor = None
        for p in module.parameters():
            if not p.stop_gradient:
                requires_grad_num += 1
                if require_grad_tensor is None:
                    require_grad_tensor = p
            total_num += 1
        print(
            "Module: [%s] have %s trainable parameters out of %s total parameters (%.2f)"
            % (name, requires_grad_num, total_num, requires_grad_num / total_num)
        )
        return require_grad_tensor


class SequenceGenAudioMAECond(Sequence2AudioMAE):
    def __init__(
        self,
        cond_stage_config,
        base_learning_rate,
        sequence_gen_length,
        sequence_input_key,
        sequence_input_embed_dim,
        batchsize,
        always_output_audiomae_gt=False,
        pretrained_path=None,
        force_reload_pretrain_avoid_overwrite=False,
        learnable=True,
        use_warmup=True,
        use_gt_mae_output=True,  # False: does not use AudioMAE GT, True: Use AudioMAE GT
        use_gt_mae_prob=0.0,
    ):  # The prob of using AudioMAE GT
        if use_warmup:
            use_warmup = False

        super().__init__(
            base_learning_rate=base_learning_rate,
            cond_stage_config=cond_stage_config,
            sequence_gen_length=sequence_gen_length,
            sequence_input_key=sequence_input_key,
            use_warmup=use_warmup,
            sequence_input_embed_dim=sequence_input_embed_dim,
            batchsize=batchsize,
        )

        assert use_gt_mae_output is not None and use_gt_mae_prob is not None
        self.always_output_audiomae_gt = always_output_audiomae_gt
        self.force_reload_pretrain_avoid_overwrite = force_reload_pretrain_avoid_overwrite
        self.pretrained_path = pretrained_path
        if self.force_reload_pretrain_avoid_overwrite:
            self.is_reload = False
        else:
            self.is_reload = True

        self.load_pretrain_model()

        self.use_gt_mae_output = use_gt_mae_output
        self.use_gt_mae_prob = use_gt_mae_prob
        self.learnable = learnable

        if not learnable:
            # Only optimize the GPT2 model
            for p in self.model.parameters():
                p.stop_gradient = True
            self.eval()

    def load_pretrain_model(self):
        if self.pretrained_path is not None:
            print("Reload SequenceGenAudioMAECond from %s" % self.pretrained_path)
            state_dict = paddle.load(self.pretrained_path)["state_dict"]
            self.load_dict(state_dict)

    # Required
    def get_unconditional_condition(self, batchsize):
        return_dict = self.cfg_uncond(batchsize)
        return_dict["crossattn_audiomae_generated"] = [
            return_dict["crossattn_audiomae_pooled"][0],
            paddle.ones_like(return_dict["crossattn_audiomae_pooled"][1], dtype="float32"),
        ]
        return return_dict

    def forward(self, batch):
        # The conditional module can return both tensor or dictionaries
        # The returned tensor will be corresponding to the cond_stage_key
        # The returned dict will have keys that correspond to the cond_stage_key
        ret_dict = {}

        if self.force_reload_pretrain_avoid_overwrite and not self.is_reload:
            self.load_pretrain_model()
            self.is_reload = True

        input_embeds, cond_dict = self.generate(batch)
        input_embeds_mask = paddle.ones((input_embeds.shape[0], input_embeds.shape[1]), dtype="float32")
        ret_dict["crossattn_audiomae_generated"] = [
            input_embeds,
            input_embeds_mask,
        ]  # Input sequence and mask

        # If the following two keys are not in cond_stage_key, then they will not be used as condition
        for key in cond_dict.keys():
            ret_dict[key] = cond_dict[key]

        return ret_dict


def instantiate_from_config(config):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package="paddlemix.models.audioldm2"), cls)

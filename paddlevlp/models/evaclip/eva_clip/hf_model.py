import paddle
""" huggingface model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.
"""
import re
try:
    import transformers
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, AutoConfig, PretrainedConfig
    from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions
except ImportError as e:
    transformers = None

    class BaseModelOutput:
        pass

    class PretrainedConfig:
        pass


from .hf_configs import arch_dict


def _camel2snake(s):
    return re.sub('(?<!^)(?=[A-Z])', '_', s).lower()


_POOLERS = {}


def register_pooler(cls):
    """Decorator registering pooler class"""
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(paddle.nn.Layer):
    """Mean pooling"""

    def forward(self, x, attention_mask):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(axis=-1)
        return masked_output.sum(axis=1) / attention_mask.sum(axis=-1,
                                                              keepdim=True)


@register_pooler
class MaxPooler(paddle.nn.Layer):
    """Max pooling"""

    def forward(self, x, attention_mask):
        masked_output = x.last_hidden_state.masked_fill(
            attention_mask.unsqueeze(axis=-1), -torch.inf)
        return masked_output.max(axis=1).values


@register_pooler
class ClsPooler(paddle.nn.Layer):
    """CLS token pooling"""

    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x, attention_mask):
        if self.use_pooler_output and isinstance(
                x, (BaseModelOutputWithPooling,
                    BaseModelOutputWithPoolingAndCrossAttentions
                    )) and x.pooler_output is not None:
            return x.pooler_output
        return x.last_hidden_state[:, (self.cls_token_position), :]


class HFTextEncoder(paddle.nn.Layer):
    """HuggingFace model adapter"""

    def __init__(self,
                 model_name_or_path: str,
                 output_dim: int,
                 tokenizer_name: str=None,
                 config: PretrainedConfig=None,
                 pooler_type: str=None,
                 proj: str=None,
                 pretrained: bool=True,
                 masked_language_modeling: bool=False):
        super().__init__()
        self.output_dim = output_dim
        uses_transformer_pooler = pooler_type == 'cls_pooler'
        if transformers is None:
            raise RuntimeError(
                'Please `pip install transformers` to use pre-trained HuggingFace models'
            )
        if config is None:
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            if masked_language_modeling:
                create_func, model_args = (
                    AutoModelForMaskedLM.from_pretrained,
                    model_name_or_path) if pretrained else (
                        AutoModelForMaskedLM.from_config, self.config)
            else:
                create_func, model_args = (
                    AutoModel.from_pretrained,
                    model_name_or_path) if pretrained else (
                        AutoModel.from_config, self.config)
            if hasattr(
                    self.config,
                    'is_encoder_decoder') and self.config.is_encoder_decoder:
                self.transformer = create_func(model_args)
                self.transformer = self.transformer.encoder
            else:
                self.transformer = create_func(
                    model_args, add_pooling_layer=uses_transformer_pooler)
        else:
            self.config = config
            if masked_language_modeling:
                self.transformer = AutoModelForMaskedLM.from_config(config)
            else:
                self.transformer = AutoModel.from_config(config)
        if pooler_type is None:
            self.pooler = _POOLERS[arch_dict[self.config.model_type][
                'pooler']]()
        else:
            self.pooler = _POOLERS[pooler_type]()
        d_model = getattr(
            self.config,
            arch_dict[self.config.model_type]['config_names']['width'])
        if d_model == output_dim and proj is None:
            self.proj = paddle.nn.Identity()
        elif proj == 'linear':
            self.proj = paddle.nn.Linear(
                in_features=d_model, out_features=output_dim, bias_attr=False)
        elif proj == 'mlp':
            hidden_size = (d_model + output_dim) // 2
            self.proj = paddle.nn.Sequential(
                paddle.nn.Linear(
                    in_features=d_model,
                    out_features=hidden_size,
                    bias_attr=False),
                paddle.nn.GELU(),
                paddle.nn.Linear(
                    in_features=hidden_size,
                    out_features=output_dim,
                    bias_attr=False))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def mask(self,
             input_ids,
             vocab_size,
             device,
             targets=None,
             masked_indices=None,
             probability_matrix=None):
        if masked_indices is None:
            masked_indices = paddle.bernoulli(x=probability_matrix).astype(
                dtype='bool')
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        if targets is not None:
            targets[~masked_indices] = -100
        indices_replaced = paddle.bernoulli(x=paddle.full(
            shape=input_ids.shape,
            fill_value=0.8)).astype(dtype='bool') & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id
        indices_random = paddle.bernoulli(x=paddle.full(
            shape=input_ids.shape, fill_value=0.5)).astype(
                dtype='bool') & masked_indices & ~indices_replaced
        if isinstance(device, paddle.dtype):
            dtype = device
        elif isinstance(device,
                        str) and device not in ['cpu', 'cuda', 'ipu', 'xpu']:
            dtype = device
        elif isinstance(device, paddle.Tensor):
            dtype = device.dtype
        else:
            dtype = paddle.randint(
                low=vocab_size, high=input_ids.shape).astype('int64').dtype
        random_words = paddle.randint(
            low=vocab_size, high=input_ids.shape).astype('int64').cast(dtype)
        input_ids[indices_random] = random_words[indices_random]
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def forward_mlm(self, input_ids, image_embeds, mlm_probability=0.25):
        labels = input_ids.clone()
        attn_mask = (input_ids != self.config.pad_token_id).astype(
            dtype='int64')
        if isinstance(input_ids.place, paddle.dtype):
            dtype = input_ids.place
        elif isinstance(
                input_ids.place,
                str) and input_ids.place not in ['cpu', 'cuda', 'ipu', 'xpu']:
            dtype = input_ids.place
        elif isinstance(input_ids.place, paddle.Tensor):
            dtype = input_ids.place.dtype
        else:
            dtype = paddle.ones(
                shape=image_embeds.shape[:-1], dtype='int64').dtype
        image_atts = paddle.ones(
            shape=image_embeds.shape[:-1], dtype='int64').cast(dtype)
        vocab_size = getattr(
            self.config,
            arch_dict[self.config.model_type]['config_names']['vocab_size'])
        probability_matrix = paddle.full(
            shape=labels.shape, fill_value=mlm_probability)
        input_ids, labels = self.mask(
            input_ids,
            vocab_size,
            input_ids.place,
            targets=labels,
            probability_matrix=probability_matrix)
        mlm_output = self.transformer(
            input_ids,
            attention_mask=attn_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            labels=labels)
        return mlm_output.loss

    def forward(self, x):
        attn_mask = (x != self.config.pad_token_id).astype(dtype='int64')
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)
        return self.proj(pooled_out)

    def lock(self, unlocked_layers: int=0, freeze_layer_norm: bool=True):
        if not unlocked_layers:
            for n, p in self.transformer.named_parameters():
                """Class Method: *.split, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                p.stop_gradient = not (not freeze_layer_norm if
                                       'LayerNorm' in n.split('.') else False)
            return
        encoder = self.transformer.encoder if hasattr(
            self.transformer, 'encoder') else self.transformer
        layer_list = getattr(
            encoder,
            arch_dict[self.config.model_type]['config_names']['layer_attr'])
        print(
            f'Unlocking {unlocked_layers}/{len(layer_list) + 1} layers of hf model'
        )
        embeddings = getattr(self.transformer, arch_dict[
            self.config.model_type]['config_names']['token_embeddings_attr'])
        modules = [embeddings, *layer_list][:-unlocked_layers]
        for module in modules:
            for n, p in module.named_parameters():
                """Class Method: *.split, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
                p.stop_gradient = not (not freeze_layer_norm if
                                       'LayerNorm' in n.split('.') else False)

    def set_grad_checkpointing(self, enable=True):
        self.transformer.gradient_checkpointing_enable()

    def get_num_layers(self):
        encoder = self.transformer.encoder if hasattr(
            self.transformer, 'encoder') else self.transformer
        layer_list = getattr(
            encoder,
            arch_dict[self.config.model_type]['config_names']['layer_attr'])
        return len(layer_list)

    def init_parameters(self):
        pass

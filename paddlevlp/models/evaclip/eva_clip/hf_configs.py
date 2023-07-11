arch_dict = {
    'roberta': {
        'config_names': {
            'context_length': 'max_position_embeddings',
            'vocab_size': 'vocab_size',
            'width': 'hidden_size',
            'heads': 'num_attention_heads',
            'layers': 'num_hidden_layers',
            'layer_attr': 'layer',
            'token_embeddings_attr': 'embeddings'
        },
        'pooler': 'mean_pooler'
    },
    'xlm-roberta': {
        'config_names': {
            'context_length': 'max_position_embeddings',
            'vocab_size': 'vocab_size',
            'width': 'hidden_size',
            'heads': 'num_attention_heads',
            'layers': 'num_hidden_layers',
            'layer_attr': 'layer',
            'token_embeddings_attr': 'embeddings'
        },
        'pooler': 'mean_pooler'
    },
    'mt5': {
        'config_names': {
            'context_length': '',
            'vocab_size': 'vocab_size',
            'width': 'd_model',
            'heads': 'num_heads',
            'layers': 'num_layers',
            'layer_attr': 'block',
            'token_embeddings_attr': 'embed_tokens'
        },
        'pooler': 'mean_pooler'
    },
    'bert': {
        'config_names': {
            'context_length': 'max_position_embeddings',
            'vocab_size': 'vocab_size',
            'width': 'hidden_size',
            'heads': 'num_attention_heads',
            'layers': 'num_hidden_layers',
            'layer_attr': 'layer',
            'token_embeddings_attr': 'embeddings'
        },
        'pooler': 'mean_pooler'
    }
}

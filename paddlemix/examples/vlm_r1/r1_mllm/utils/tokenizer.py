
from paddlenlp.utils.import_utils import import_module

from .constant import SUPPORTED_MODELS,IMAGE_PROCESSOR_MAPPING,TOKENIZER_MAPPING,VL_PROCESSOR_MAPPING


def get_processor(model_name,model_path,**kwargs):
    if model_name in SUPPORTED_MODELS.keys():
        img_processor_cls = import_module(f"paddlemix.processors.{IMAGE_PROCESSOR_MAPPING[model_name]}")
        image_processor = img_processor_cls()
        tokenizer = import_module(f"paddlemix.models.{TOKENIZER_MAPPING[model_name]}").from_pretrained(model_path,padding_side='left')
        vl_processor_cls = import_module(f"paddlemix.processors.{VL_PROCESSOR_MAPPING[model_name]}")
        processor = vl_processor_cls(image_processor, tokenizer)

        pad_token_id = processor.tokenizer.pad_token_id
        processor.pad_token_id = pad_token_id
        processor.eos_token_id = processor.tokenizer.eos_token_id
        if kwargs.get('max_pixels',None):
            processor.image_processor.max_pixels = kwargs['max_pixels']
        if kwargs.get('min_pixels',None):
            processor.image_processor.min_pixels = kwargs['min_pixels']
    else:
        raise ValueError(f"Invalid model: {model_name}")
    return processor,tokenizer

# TODO
def get_tokenizer():
    pass
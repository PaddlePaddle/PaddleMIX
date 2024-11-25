import paddle
"""
Processor class for MiniCPMV.
"""
from typing import List, Optional, Union, Dict, Any
import re
from paddlenlp.transformers.processing_utils import ProcessorMixin
from .image_processing_minicpmv import MiniCPMVBatchFeature
from paddlenlp.transformers.tokenizer_utils_base import (
    PreTokenizedInput,
    TensorType,
    TextInput,
)
from paddlenlp.transformers.image_utils import ImageInput
# from .image_processing_minicpmv import MiniCPMVImageProcessor

__all__ = [
    "MiniCPMVProcessor",
]

class MiniCPMVProcessor(ProcessorMixin):
    """
    Constructs a MiniCPMV processor which wraps a MiniCPMV image processor and a MiniCPMV tokenizer into a single processor.

    [`MiniCPMVProcessor`] offers all the functionalities of [`MiniCPMVImageProcessor`] and [`LlamaTokenizerWrapper`]. See the
    [`~MiniCPMVProcessor.__call__`] and [`~MiniCPMVProcessor.decode`] for more information.

    Args:
        image_processor ([`MiniCPMVImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerWrapper`], *optional*):
            The tokenizer is a required input.
    """
    attributes = ['image_processor', 'tokenizer']
    image_processor_class = 'MiniCPMVImageProcessor'
    tokenizer_class = 'LlamaTokenizer'

    def __init__(self, image_processor=None, tokenizer=None):
        super().__init__(image_processor, tokenizer)
        self.version = image_processor.version

    def __call__(self, text: Union[TextInput, PreTokenizedInput,
        List[TextInput], List[PreTokenizedInput]], images:ImageInput=None, max_length: Optional[int]
        =None, do_pad: Optional[bool]=True, max_slice_nums: int=None,
        use_image_id: bool=None, return_tensors: Optional[Union[str,TensorType]]=TensorType.PADDLE, **kwargs) ->MiniCPMVBatchFeature:
        if images is not None:
            image_inputs = self.image_processor(images, do_pad=do_pad,
                max_slice_nums=max_slice_nums, return_tensors=return_tensors)
        return self._convert_images_texts_to_inputs(image_inputs, text,
            max_slice_nums=max_slice_nums, use_image_id=use_image_id,
            max_length=max_length, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        output_ids = args[0]
        result_text = []
        for result in output_ids:
            result = result[result != 0]
            if result[0] == self.tokenizer.bos_id:
                result = result[1:]
            if result[-1] == self.tokenizer.eos_id:
                result = result[:-1]
            result_text.append(self.tokenizer.decode(result, *args[1:], **
                kwargs).strip())
        return result_text

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        result = args[0]
        result = result[result != 0]
        if result[0] == self.tokenizer.bos_id:
            result = result[1:]
        if result[-1] == self.tokenizer.eos_id or hasattr(self.tokenizer,
            'eot_id') and result[-1] == self.tokenizer.eot_id:
            result = result[:-1]
        return self.tokenizer.decode(result, *args[1:], **kwargs).strip()

    def _convert(self, input_str, max_inp_length: Optional[int]=None):
        if self.version > 2.5 or not getattr(self.tokenizer, 'add_bos_token', False):
            tokenizer_output = self.tokenizer(input_str, return_token_type_ids=False, return_attention_mask=False)
            input_ids = tokenizer_output['input_ids']
        else:
            tokenizer_output = self.tokenizer(input_str, return_token_type_ids=False, return_attention_mask=False)
            input_ids = [self.tokenizer.bos_id] + tokenizer_output['input_ids']
        
        if max_inp_length is not None:
            input_ids = input_ids[0:max_inp_length] if isinstance(input_ids, list) else input_ids[:max_inp_length]
        
        input_ids = paddle.to_tensor(data=input_ids, dtype='int32')
        start_cond = (input_ids == self.tokenizer.im_start_id) | (input_ids == self.tokenizer.slice_start_id)
        end_cond = (input_ids == self.tokenizer.im_end_id) | (input_ids == self.tokenizer.slice_end_id)
        
        image_start_tokens = paddle.nonzero(start_cond)[:,0]
        image_start_tokens = image_start_tokens + 1
        image_end_tokens = paddle.nonzero(end_cond)[:,0]
        valid_image_nums = max(len(image_start_tokens), len(image_end_tokens))
        image_bounds = paddle.hstack(x=[
            image_start_tokens[:valid_image_nums].unsqueeze(axis=-1), 
            image_end_tokens[:valid_image_nums].unsqueeze(axis=-1)
        ])
        
        return input_ids, image_bounds

    def _convert_images_texts_to_inputs(
            self, 
            images, 
            texts: Union[str, List[str]], 
            truncation=None, 
            max_length=None,
            max_slice_nums=None,
            use_image_id=None, 
            return_tensors=None,
            **kwargs
        ):
        if images is None or not len(images):
            model_inputs = self.tokenizer(texts, return_tensors=return_tensors, truncation=truncation, max_length=max_length, **kwargs)
            return MiniCPMVBatchFeature(data={**model_inputs})
        
        pattern = "(<image>./</image>)"
        images, image_sizes, tgt_sizes = images["pixel_values"], images["image_sizes"], images["tgt_sizes"]
        
        if isinstance(texts, str):
            texts = [texts]
        input_ids_list = []
        image_bounds_list = []
        for index, text in enumerate(texts):
            image_tags = re.findall(pattern, text)
            assert len(image_tags) == len(image_sizes[index])
            text_chunks = text.split(pattern)
            final_text = ""
            for i in range(len(image_tags)):
                final_text = final_text + text_chunks[i] + \
                    self.image_processor.get_slice_image_placeholder(
                        image_sizes[index][i], 
                        i,
                        max_slice_nums,
                        use_image_id
                    )
            final_text += text_chunks[-1]
            input_ids, image_bounds = self._convert(final_text, max_length)
            input_ids_list.append(input_ids)
            image_bounds_list.append(image_bounds)
        
        padded_input_ids, padding_lengths = self.pad(
            input_ids_list,
            padding_side="left"
        )
        padding_lengths = [padded_input_ids.shape[1] - len(ids) for ids in input_ids_list]
        
        for i, length in enumerate(padding_lengths):
            image_bounds_list[i] = image_bounds_list[i] + length
        
        attention_mask = paddle.not_equal(padded_input_ids, paddle.zeros_like(padded_input_ids))

        return MiniCPMVBatchFeature(data={
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "pixel_values": images,
            "image_sizes": image_sizes,
            "image_bound": image_bounds_list,
            "tgt_sizes": tgt_sizes
        })

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names +
            image_processor_input_names))

    def pad(self, inputs, max_length=None, padding_value=0, padding_side='left'
        ):
        items = []
        if isinstance(inputs[0], list):
            assert isinstance(inputs[0][0], paddle.Tensor)
            for it in inputs:
                for tr in it:
                    items.append(tr)
        else:
            assert isinstance(inputs[0], paddle.Tensor)
            items = inputs
        batch_size = len(items)
        shape = tuple(items[0].shape)
        dim = len(shape)
        assert dim <= 2
        if max_length is None:
            max_length = 0
        max_length = max(max_length, max(tuple(item.shape)[-1] for item in
            items))
        min_length = min(tuple(item.shape)[-1] for item in items)
        dtype = items[0].dtype
        if dim == 0:
            return paddle.stack(x=[item for item in items], axis=0), [0]
        elif dim == 1:
            if max_length == min_length:
                return paddle.stack(x=[item for item in items], axis=0), [0
                    ] * batch_size
            tensor = paddle.zeros(shape=(batch_size, max_length), dtype=dtype
                ) + padding_value
        else:
            tensor = paddle.zeros(shape=(batch_size, max_length, shape[-1]),
                dtype=dtype) + padding_value
        padding_length = []
        for i, item in enumerate(items):
            if dim == 1:
                if padding_side == 'left':
                    tensor[i, -len(item):] = item.clone()
                else:
                    tensor[i, :len(item)] = item.clone()
            elif dim == 2:
                if padding_side == 'left':
                    tensor[i, -len(item):, :] = item.clone()
                else:
                    tensor[i, :len(item), :] = item.clone()
            padding_length.append(tuple(tensor.shape)[-1] - len(item))
        return tensor, padding_length




import dataclasses
from dataclasses import dataclass
from typing import Dict, List,Union,Tuple,Any
from PIL import Image
from enum import IntEnum, auto
from functools import partial,reduce
import copy

import paddle
from paddlenlp.transformers import LlamaTokenizerFast
import numpy as np
from paddlenlp.transformers.image_transforms import (
    normalize,
    rescale,
)
from paddlenlp.transformers.image_utils import to_numpy_array

from .base_processing import ProcessorMixin
from .processing_utils import BaseImageProcessor
from .image_processing_utils import BatchFeature

__all__ = ["JanusImageProcessor", "JanusVLChatProcessor","JanusFlowVLChatProcessor"]

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class JanusImageProcessor(BaseImageProcessor):
    model_input_names = ['pixel_values']

    def __init__(self, image_size: int, min_size: int=14, image_mean: Union
        [Tuple[float, float, float], List[float]]=(0.48145466, 0.4578275, 
        0.40821073), image_std: Union[Tuple[float, float, float], List[
        float]]=(0.26862954, 0.26130258, 0.27577711), rescale_factor: float
        =1.0 / 255.0, do_normalize: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean
        self.image_std = image_std
        self.min_size = min_size
        self.do_normalize = do_normalize
        if image_mean is None:
            self.background_color = 127, 127, 127
        else:
            self.background_color = tuple([int(x * 255) for x in image_mean])
        self.transform = [
            partial(rescale, scale=self.rescale_factor, data_format="channels_first"),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format="channels_first"),
        ] 

    def resize(self, pil_img: Image) ->np.ndarray:
        """

        Args:
            pil_img (PIL.Image): [H, W, 3] in PIL.Image in RGB

        Returns:
            x (np.ndarray): [3, self.image_size, self.image_size]
        """
        width, height = pil_img.size
        max_size = max(width, height)
        size = [max(int(height / max_size * self.image_size), self.min_size
            ), max(int(width / max_size * self.image_size), self.min_size)]
        if width <= 0 or height <= 0 or size[0] <= 0 or size[1] <= 0:
            print(f'orig size = {pil_img.size}, new size = {size}')
            raise ValueError('Invalid size!')
        pil_img = paddle.vision.transforms.resize(pil_img, size,
            interpolation='bicubic')
        pil_img = expand2square(pil_img, self.background_color)
        x = to_numpy_array(pil_img)
        x = np.transpose(x, (2, 0, 1))
        return x

    def preprocess(self, images, return_tensors: str='pt', **kwargs
        ) -> BatchFeature:
        images: List[np.ndarray] = [self.resize(image) for image in images]
        images = reduce(lambda x, f: [*map(f, x)], self.transform, images)
        data = {'pixel_values': images}
        return BatchFeature(data=data,
            tensor_type=return_tensors)

    @property
    def default_shape(self):
        return [3, self.image_size, self.image_size]
    
    def to_dict(self, saving_file=False) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this processor instance.
        """
        output = copy.deepcopy(self.__dict__)
        output.pop("transform")
        output["processor_type"] = self.__class__.__name__

        return output
    
    
class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    DeepSeek = auto()
    PLAIN = auto()
    ALIGNMENT = auto()

@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: List[str] = (("USER", "ASSISTANT"),)
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)

        if self.sep_style == SeparatorStyle.DeepSeek:
            seps = [self.sep, self.sep2]
            if system_prompt == "" or system_prompt is None:
                ret = ""
            else:
                ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = "[INST] "
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if type(message) is tuple:  # multimodal message
                        message, _ = message
                    if i == 0:
                        ret += message + " "
                    else:
                        ret += tag + " " + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i % 2 == 0:
                        ret += message + seps[i % 2]
                    else:
                        ret += message + seps[i % 2]
                else:
                    ret += ""
            return ret
        elif self.sep_style == SeparatorStyle.ALIGNMENT:
            seps = [self.sep, self.sep2]
            ret = ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i % 2 == 0:
                        ret += "<image>\n" + seps[i % 2]
                    else:
                        ret += message + seps[i % 2]
                else:
                    ret += ""
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def get_prompt_for_current_round(self, content=None):
        """Get current round formatted question prompt during sft training"""
        if self.sep_style == SeparatorStyle.PLAIN:
            formatted_question = "<image>\n"
        elif self.sep_style == SeparatorStyle.DeepSeek:
            formatted_question = (
                f"{self.roles[0]}: " + content.strip() + self.sep + f"{self.roles[1]}:"
            )
        else:
            raise ValueError(f"Unsupported sep_style: {self.sep_style}")
        return formatted_question

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def reset_message(self):
        """Reset a new message."""
        self.messages = []

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        ret = [{"role": "system", "content": system_prompt}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


class DictOutput(object):

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

@dataclass
class JanusVLChatProcessorOutput(DictOutput):
    sft_format: str
    input_ids: paddle.Tensor
    pixel_values: paddle.Tensor
    num_image_tokens: paddle.int32

    def __len__(self):
        return len(self.input_ids)


@dataclass
class JanusBatchedVLChatProcessorOutput(DictOutput):
    sft_format: List[str]
    input_ids: paddle.Tensor
    pixel_values: paddle.Tensor
    attention_mask: paddle.Tensor
    images_seq_mask: paddle.bool
    images_emb_mask: paddle.bool

    def to(self, device, dtype='bfloat16'):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.images_seq_mask = self.images_seq_mask.to(device)
        self.images_emb_mask = self.images_emb_mask.to(device)
        self.pixel_values = self.pixel_values.to(device=device, dtype=dtype)
        return self


class JanusVLChatProcessor(ProcessorMixin):
    image_processor_class = 'AutoImageProcessor'
    tokenizer_class = 'LlamaTokenizer', 'LlamaTokenizerFast'
    attributes = ['image_processor', 'tokenizer']
    system_prompt = (
        'You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.'
    )

    def __init__(self, image_processor: JanusImageProcessor, tokenizer:
        LlamaTokenizerFast, image_tag: str=
        '<image_placeholder>', image_start_tag: str='<begin_of_image>',
        image_end_tag: str='<end_of_image>', num_image_tokens: int=576,
        add_special_token: bool=False, sft_format: str='deepseek',
        mask_prompt: bool=True, ignore_id: int=-100, **kwargs):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        image_id = self.tokenizer.vocab.get(image_tag)
        if image_id is None:
            special_tokens = [image_tag]
            special_tokens_dict = {'additional_special_tokens': special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            print(f'Add image tag = {image_tag} to the tokenizer')
        self.image_tag = image_tag
        self.image_start_tag = image_start_tag
        self.image_end_tag = image_end_tag
        self.num_image_tokens = num_image_tokens
        self.add_special_token = add_special_token
        self.sft_format = sft_format
        self.mask_prompt = mask_prompt
        self.ignore_id = ignore_id
        self.conv_templates: Dict[str, Conversation] = {}
        
        # llava_llama2 template
        self.register_conv_template(
            Conversation(
                name="llava_llama2",
                system_message="You are a helpful language and vision assistant. "
                "You are able to understand the visual content that the user provides, "
                "and assist the user with a variety of tasks using natural language.",
                system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
                roles=("[INST]", "[/INST]"),
                messages=(),
                offset=0,
                sep_style=SeparatorStyle.LLAMA2,
                sep=" ",
                sep2=" </s><s>",
                stop_token_ids=[2],
            )
        )
        # llama2 template
        # reference: https://github.com/facebookresearch/llama/blob/cfc3fc8c1968d390eb830e65c63865e980873a06/llama/generation.py#L212
        self.register_conv_template(
            Conversation(
                name="llama-2",
                system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
                roles=("[INST]", "[/INST]"),
                messages=(),
                offset=0,
                sep_style=SeparatorStyle.LLAMA2,
                sep=" ",
                sep2=" </s><s>",
                stop_token_ids=[2],
            )
        )        
        # deepseek template
        self.register_conv_template(
            Conversation(
                name="deepseek",
                system_template="{system_message}",
                # system_message="You are a helpful assistant. Please answer truthfully and write out your "
                # "thinking step by step to be sure you get the right answer.",
                system_message="",
                roles=("User", "Assistant"),
                messages=(),
                offset=0,
                sep_style=SeparatorStyle.DeepSeek,
                sep="\n\n",
                sep2="<｜end▁of▁sentence｜>",
                stop_token_ids=[100001],
                stop_str=["User:", "<｜end▁of▁sentence｜>"],
            )
        )
        self.register_conv_template(
            Conversation(
                name="plain",
                system_template="",
                system_message="",
                roles=("", ""),
                messages=(),
                offset=0,
                sep_style=SeparatorStyle.PLAIN,
                sep="",
                sep2="",
                stop_token_ids=[2],
                stop_str=["</s>"],
            )
        )
        self.register_conv_template(
            Conversation(
                name="alignment",
                system_template="",
                system_message="",
                roles=("", ""),
                messages=(),
                offset=0,
                sep_style=SeparatorStyle.ALIGNMENT,
                sep="",
                sep2="",
                stop_token_ids=[2],
                stop_str=["</s>"],
            )
        )
        super().__init__(image_processor, tokenizer, image_tag,
            num_image_tokens, add_special_token, sft_format, mask_prompt,
            ignore_id, **kwargs)
        
    def register_conv_template(self,template: Conversation, override: bool = False):
        """Register a new conversation template."""
        if not override:
            assert (
                template.name not in self.conv_templates
            ), f"{template.name} has been registered."

        self.conv_templates[template.name] = template

    def get_conv_template(self,name: str) -> Conversation:
        """Get a conversation template."""
        return self.conv_templates[name].copy()
    
    def new_chat_template(self):
        conv = self.get_conv_template(self.sft_format)
        conv.set_system_message(self.system_prompt)
        return conv

    def apply_sft_template_for_multi_turn_prompts(self, conversations: List
        [Dict[str, str]], sft_format: str='deepseek', system_prompt: str=''):
        """
        Applies the SFT template to conversation.

        An example of conversation:
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder> is Figure 1.
                            <image_placeholder> is Figure 2.
                            Which image is brighter?",
                "images": [
                    "./multi-images/attribute_comparison_1.png",
                    "./multi-images/attribute_comparison_2.png"
                ]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        Args:
            conversations (List[Dict]): A conversation with a List of Dict[str, str] text.
            sft_format (str, optional): The format of the SFT template to use. Defaults to "deepseek".
            system_prompt (str, optional): The system prompt to use in the SFT template. Defaults to "".

        Returns:
            sft_prompt (str): The formatted text.
        """
        conv = self.get_conv_template(sft_format)
        conv.set_system_message(system_prompt)
        for message in conversations:
            conv.append_message(message['role'], message['content'].strip())
        sft_prompt = conv.get_prompt().strip()
        return sft_prompt

    @property
    def image_token(self):
        return self.image_tag

    @property
    def image_id(self):
        image_id = self.tokenizer.vocab.get(self.image_tag)
        return image_id

    @property
    def image_start_id(self):
        image_start_id = self.tokenizer.vocab.get(self.image_start_tag)
        return image_start_id

    @property
    def image_end_id(self):
        image_end_id = self.tokenizer.vocab.get(self.image_end_tag)
        return image_end_id

    @property
    def image_start_token(self):
        return self.image_start_tag

    @property
    def image_end_token(self):
        return self.image_end_tag

    @property
    def pad_id(self):
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        return pad_id

    def add_image_token(self, image_indices: List[int], input_ids: paddle.
        Tensor):
        """

        Args:
            image_indices (List[int]): [index_0, index_1, ..., index_j]
            input_ids (torch.LongTensor): [N]

        Returns:
            input_ids (torch.LongTensor): [N + image tokens]
            num_image_tokens (torch.IntTensor): [n_images]
        """
        input_slices = []
        start = 0
        for index in image_indices:
            if self.add_special_token:
                end = index + 1
            else:
                end = index
            input_slices.append(input_ids[start:end])
            input_slices.append(self.image_start_id * paddle.ones(shape=[1],
                dtype='int64'))
            input_slices.append(self.image_id * paddle.ones(shape=(self.
                num_image_tokens,), dtype='int64'))
            input_slices.append(self.image_end_id * paddle.ones(shape=[1],
                dtype='int64'))
            start = index + 1
        input_slices.append(input_ids[start:])
        input_ids = paddle.concat(x=input_slices, axis=0)
        num_image_tokens = paddle.to_tensor(data=[self.num_image_tokens] *
            len(image_indices), dtype='int32')
        return input_ids, num_image_tokens

    def process_one(self, prompt: str=None, conversations: List[Dict[str,
        str]]=None, images: List[Image.Image]=None, **kwargs):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - target_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """
        assert prompt is None or conversations is None, 'prompt and conversations cannot be used at the same time.'
        if prompt is None:
            sft_format = self.apply_sft_template_for_multi_turn_prompts(
                conversations=conversations, sft_format=self.sft_format,
                system_prompt=self.system_prompt)
        else:
            sft_format = prompt
        input_ids = self.tokenizer.encode(sft_format)
        input_ids['input_ids'] = paddle.cast(paddle.to_tensor(input_ids['input_ids']), dtype='int64')
        image_token_mask: paddle.bool = input_ids['input_ids'] == self.image_id
        image_indices = image_token_mask.nonzero()
        input_ids, num_image_tokens = self.add_image_token(image_indices=image_indices, input_ids=input_ids['input_ids'])
        images_outputs = self.image_processor(images, return_tensors='pd')
        prepare = JanusVLChatProcessorOutput(sft_format=sft_format, input_ids=
            input_ids, pixel_values=images_outputs.pixel_values,
            num_image_tokens=num_image_tokens)
        return prepare

    def __call__(self, *, prompt: str=None, conversations: List[Dict[str,
        str]]=None, images: List[Image.Image]=None, force_batchify: bool=True, **
        kwargs):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            force_batchify (bool): force batchify the inputs;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """
        prepare = self.process_one(prompt=prompt, conversations=
            conversations, images=images)
        if force_batchify:
            prepare = self.batchify([prepare])
        return prepare

    def batchify(self, prepare_list: List[JanusVLChatProcessorOutput]
        ) ->JanusBatchedVLChatProcessorOutput:
        """
        Preprocesses the inputs for multimodal inference.

        Args:
            prepare_list (List[VLChatProcessorOutput]): A list of VLChatProcessorOutput.

        Returns:
            BatchedVLChatProcessorOutput: A dictionary of the inputs to use for multimodal inference.
        """
        batch_size = len(prepare_list)
        sft_format = []
        n_images = []
        seq_lens = []
        for prepare in prepare_list:
            n_images.append(len(prepare.num_image_tokens))
            seq_lens.append(len(prepare))
        input_token_max_len = max(seq_lens)
        max_n_images = max(1, max(n_images))
        batched_input_ids = paddle.full(shape=(batch_size,
            input_token_max_len), fill_value=self.pad_id).astype(dtype='int64')
        batched_attention_mask = paddle.zeros(shape=(batch_size,
            input_token_max_len)).astype(dtype='int64')
        batched_pixel_values = paddle.zeros(shape=(batch_size, max_n_images,
            *self.image_processor.default_shape)).astype(dtype='float32')
        batched_images_seq_mask = paddle.zeros(shape=(batch_size,
            input_token_max_len)).astype(dtype='bool')
        batched_images_emb_mask = paddle.zeros(shape=(batch_size,
            max_n_images, self.num_image_tokens)).astype(dtype='bool')
        for i, prepare in enumerate(prepare_list):
            input_ids = prepare.input_ids
            seq_len = len(prepare)
            n_image = len(prepare.num_image_tokens)
            batched_attention_mask[i, -seq_len:] = 1
            batched_input_ids[i, -seq_len:] = paddle.to_tensor(data=
                input_ids, dtype='int64')
            batched_images_seq_mask[i, -seq_len:] = input_ids == self.image_id
            if n_image > 0:
                batched_pixel_values[i, :n_image] = prepare.pixel_values
                for j, n_image_tokens in enumerate(prepare.num_image_tokens):
                    batched_images_emb_mask[i, j, :n_image_tokens] = True
            sft_format.append(prepare.sft_format)
        batched_prepares = JanusBatchedVLChatProcessorOutput(input_ids=
            batched_input_ids, attention_mask=batched_attention_mask,
            pixel_values=batched_pixel_values, images_seq_mask=
            batched_images_seq_mask, images_emb_mask=
            batched_images_emb_mask, sft_format=sft_format)
        return batched_prepares


class JanusFlowVLChatProcessor(JanusVLChatProcessor):
    def __init__(
        self,
        image_processor: JanusImageProcessor,
        tokenizer: LlamaTokenizerFast,
        image_tag: str = "<image_placeholder>",
        image_start_tag: str = "<begin_of_image>",
        image_end_tag: str = "<end_of_image>",
        image_gen_tag: str = "<｜begin▁of▁generation｜>",
        num_image_tokens: int = 576,
        add_special_token: bool = False,
        sft_format: str = "deepseek",
        mask_prompt: bool = True,
        ignore_id: int = -100,
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            image_tag=image_tag,
            image_start_tag=image_start_tag,
            image_end_tag=image_end_tag,
            num_image_tokens=num_image_tokens,
            add_special_token=add_special_token,
            sft_format=sft_format,
            mask_prompt=mask_prompt,
            ignore_id=ignore_id,
            **kwargs,)
        image_gen_id = self.tokenizer.vocab.get(image_gen_tag)
        if image_gen_id is None:
            special_tokens = [image_gen_tag]
            special_tokens_dict = {"additional_special_tokens": special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Add generation tag = {image_gen_tag} to the tokenizer")

        assert image_start_tag is not None and image_end_tag is not None
        boi_id = self.tokenizer.vocab.get(image_start_tag)
        eoi_id = self.tokenizer.vocab.get(image_end_tag)
        if boi_id is None:
            special_tokens = [image_start_tag]
            special_tokens_dict = {"additional_special_tokens": special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Add boi tag = {image_start_tag} to the tokenizer")
        if eoi_id is None:
            special_tokens = [image_end_tag]
            special_tokens_dict = {"additional_special_tokens": special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Add eoi tag = {image_end_tag} to the tokenizer")
        self.image_gen_tag = image_gen_tag
        self.tokenizer.pad_token_id = self.tokenizer.vocab.get("<｜▁pad▁｜>")
        
    @property
    def image_gen_id(self):
        image_gen_id = self.tokenizer.vocab.get(self.image_gen_tag)
        return image_gen_id

    def add_image_token(
        self,
        image_indices: List[int],
        input_ids: paddle.Tensor,
    ):
        """

        Args:
            image_indices (List[int]): [index_0, index_1, ..., index_j]
            input_ids (torch.LongTensor): [N]

        Returns:
            input_ids (torch.LongTensor): [N + image tokens]
            num_image_tokens (torch.IntTensor): [n_images]
        """

        input_slices = []
        start = 0
        for index in image_indices:
            if self.add_special_token:
                end = index + 1
            else:
                end = index
            input_slices.append(input_ids[start:end])
            input_slices.append(self.image_start_id * paddle.ones(shape=[1],
                dtype='int64'))
            input_slices.append(self.image_id * paddle.ones(shape=(self.
                num_image_tokens,), dtype='int64'))
            input_slices.append(self.image_end_id * paddle.ones(shape=[1],
                dtype='int64'))
            start = index + 1
        input_slices.append(input_ids[start:])
        input_ids = paddle.concat(x=input_slices, axis=0)
        num_image_tokens = paddle.to_tensor(data=[self.num_image_tokens] *
            len(image_indices), dtype='int32')
        return input_ids, num_image_tokens
    
    def batchify(
        self, prepare_list: List[JanusVLChatProcessorOutput]
    ) -> JanusBatchedVLChatProcessorOutput:
        """
        Preprocesses the inputs for multimodal inference.

        Args:
            prepare_list (List[VLChatProcessorOutput]): A list of VLChatProcessorOutput.

        Returns:
            BatchedVLChatProcessorOutput: A dictionary of the inputs to use for multimodal inference.
        """

        batch_size = len(prepare_list)
        sft_format = []
        n_images = []
        seq_lens = []
        for prepare in prepare_list:
            n_images.append(len(prepare.num_image_tokens))
            seq_lens.append(len(prepare))
        input_token_max_len = max(seq_lens)
        max_n_images = max(1, max(n_images))
        batched_input_ids = paddle.full(shape=(batch_size,
            input_token_max_len), fill_value=self.pad_id).astype(dtype='int64')
        batched_attention_mask = paddle.zeros(shape=(batch_size,
            input_token_max_len)).astype(dtype='int64')
        batched_pixel_values = paddle.zeros(shape=(batch_size, max_n_images,
            *self.image_processor.default_shape)).astype(dtype='float32')
        batched_images_seq_mask = paddle.zeros(shape=(batch_size,
            input_token_max_len)).astype(dtype='bool')
        batched_images_emb_mask = paddle.zeros(shape=(batch_size,
            max_n_images, self.num_image_tokens)).astype(dtype='bool')
        for i, prepare in enumerate(prepare_list):
            input_ids = prepare.input_ids
            seq_len = len(prepare)
            n_image = len(prepare.num_image_tokens)
            batched_attention_mask[i, -seq_len:] = 1
            batched_input_ids[i, -seq_len:] = paddle.to_tensor(data=
                input_ids, dtype='int64')
            batched_images_seq_mask[i, -seq_len:] = input_ids == self.image_id
            if n_image > 0:
                batched_pixel_values[i, :n_image] = prepare.pixel_values
                for j, n_image_tokens in enumerate(prepare.num_image_tokens):
                    batched_images_emb_mask[i, j, :n_image_tokens] = True
            sft_format.append(prepare.sft_format)

        batched_prepares = JanusBatchedVLChatProcessorOutput(
            input_ids=batched_input_ids,
            attention_mask=batched_attention_mask,
            pixel_values=batched_pixel_values,
            images_seq_mask=batched_images_seq_mask,
            images_emb_mask=batched_images_emb_mask,
            sft_format=sft_format,
        )

        return batched_prepares
import paddle
""" CLIP tokenizer

Copied from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import gzip
import html
import os
from functools import lru_cache
from typing import Union, List
import ftfy
import regex as re
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


@lru_cache()
def default_bpe():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../eva_clip/bpe_simple_vocab_16e6.txt.gz')


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord('!'), ord('~') + 1)) + list(
        range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub('\\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self,
                 bpe_path: str=default_bpe(),
                 special_tokens=None,
                 context_length=77):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode('utf-8').split('\n')
        merges = merges[1:49152 - 256 - 2 + 1]
        """Class Method: *.split, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [(v + '</w>') for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        if not special_tokens:
            special_tokens = ['<start_of_text>', '<end_of_text>']
        else:
            special_tokens = ['<start_of_text>', '<end_of_text>'
                              ] + special_tokens
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t: t for t in special_tokens}
        special = '|'.join(special_tokens)
        self.pat = re.compile(
            special +
            "|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
            re.IGNORECASE)
        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]
        self.context_length = context_length

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>', )
        pairs = get_pairs(word)
        if not pairs:
            return token + '</w>'
        while True:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[
                        i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b]
                            for b in token.encode('utf-8'))
            """Class Method: *.split, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
            bpe_tokens.extend(self.encoder[bpe_token]
                              for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            'utf-8', errors='replace').replace('</w>', ' ')
        return text

    def tokenize(self, texts: Union[str, List[str]]):
        """
        Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        if isinstance(texts, str):
            texts = [texts]
        sot_token = self.encoder['<start_of_text>']
        eot_token = self.encoder['<end_of_text>']
        all_tokens = [([sot_token] + self.encode(text) + [eot_token])
                      for text in texts]
        result = []
        type_ids = []
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.context_length:
                tokens = tokens[:self.context_length]
                tokens[-1] = eot_token
            else:
                tokens = tokens[:self.context_length] + [eot_token] * max(
                    0, self.context_length - len(tokens))

            type_id = [int(i != sot_token and i != eot_token) for i in tokens]
            # print("len tokens:{}, type_id:{}".format(len(tokens), len(type_id)))
            type_ids.append(type_id)
            result.append(tokens)
        # print("len of tokens:{}, type_id:{}".format(len(result), len(type_ids)))
        return result, type_ids


class HFTokenizer:
    """HuggingFace tokenizer wrapper"""

    def __init__(self, tokenizer_name: str):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, texts: Union[str, List[str]],
                 context_length: int=77) -> paddle.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        texts = [whitespace_clean(basic_clean(text)) for text in texts]
        input_ids = self.tokenizer(
            texts,
            return_tensors='pt',
            max_length=context_length,
            padding='max_length',
            truncation=True).input_ids
        return input_ids

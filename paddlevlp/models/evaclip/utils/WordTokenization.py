import re
import six
import collections
import paddle


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    fin = open(vocab_file, 'rb')
    for num, line in enumerate(fin):
        items = convert_to_unicode(line.strip()).split("\t")
        if len(items) > 2:
            break
        token = items[0]
        index = items[1] if len(items) == 2 else num
        token = token.strip()
        vocab[token] = int(index)
    return vocab


class WordTokenizer(object):
    def __init__(self,
                 vocab_file,
                 context_length,
                 do_lower_case=True,
                 sp_vocab=False):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pat = re.compile(r'([a-zA-Z0-9]+\s|\S+\s|[a-zA-Z0-9]+$|\S+$)')
        self.do_lower_case = do_lower_case
        self.sp_vocab = sp_vocab
        self.pad = '[PAD]'
        self.unk = '[UNK]'
        self.cls = '[CLS]'
        self.pad_id = self.vocab.get(self.pad)
        self.cls_id = self.vocab.get(self.cls)
        self.max_length = context_length

    def wordpiece(self, token, vocab, unk_token, sp_vocab=False):
        """call with single word"""
        chars = list(token.strip())
        max_input_chars_per_word = 1024
        if len(chars) > max_input_chars_per_word:
            return [unk_token], [(0, len(chars))]

        is_bad = False
        start = 0
        sub_tokens = []
        sub_pos = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if start == 0 and sp_vocab:
                    substr = u'\u2581' + substr
                if start > 0 and not sp_vocab:
                    if re.match("^[A-Za-z0-9]+$", substr):
                        substr = "##" + substr
                    else:
                        substr = substr
                if substr in vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                is_bad = True
                break
            sub_tokens.append(cur_substr)
            sub_pos.append((start, end))
            start = end
        if is_bad:
            return [unk_token], [(0, len(chars))]
        else:
            return sub_tokens, sub_pos

    def word_token(self, text):
        if len(text) == 0:
            return []
        text = convert_to_unicode(text)
        if self.do_lower_case:
            text = text.lower()
        res = []
        for match in self.pat.finditer(text):
            words, _ = self.wordpiece(
                match.group(0),
                vocab=self.vocab,
                unk_token=self.unk,
                sp_vocab=self.sp_vocab)
            res.extend(words)
        #print(res)
        return res

    def convert_tokens_to_ids(self, tokens):
        #print(tokens)
        return [self.vocab.get(t, self.vocab[self.unk]) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.inv_vocab.get(i) for i in ids]

    def padding_to_max(self, one_list):
        max_length_m = self.max_length - 1
        return [self.cls_id] + one_list[:max_length_m] + [self.pad_id] * max(
            0, max_length_m - len(one_list))

    def decode(self, tokens):
        if paddle.is_tensor(tokens):
            tokens = paddle.tolist(tokens)
        tokens = [token for token in tokens if token not in (0, )]
        return ''.join(self.convert_ids_to_tokens(tokens))

    def encode(self, text):
        return paddle.to_tensor(
            self.convert_tokens_to_ids(self.word_token(text)))

    def tokenize(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        tensor_list = []
        type_ids = []
        for one_line in texts:
            ids = self.convert_tokens_to_ids(self.word_token(one_line))
            padding_res = self.padding_to_max(ids)
            tensor_list.append(padding_res)
            type_ids.append([
                int(i != self.pad_id and i != self.cls_id) for i in padding_res
            ])
        #input_ids = paddle.to_tensor(tensor_list)
        return tensor_list, type_ids


if __name__ == '__main__':
    word_tk = WordTokenizer(vocab_file="./vocab.txt", context_length=40)

    text_cn = "以及方便更好的追踪需求的实际进度指导后续效率优化的方向,今天是国庆节，群众们开始爱国了"
    text_en = "Subsequent global financial downturns doing little to encourage developers." * 10

    print(text_cn)
    print('word level: ', word_tk.word_token(text_cn))
    input_ids, _ = word_tk.tokenize(text_cn)
    print(input_ids)
    print('after decode')
    print(word_tk.decode(input_ids[0]), end='$$$')
    # print(text_en)
    # print(word_tk.tokenize(text_en))

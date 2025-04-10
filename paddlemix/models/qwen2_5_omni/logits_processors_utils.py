import paddle
from paddlenlp.generation import LogitsProcessor

class SuppressTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, suppress_tokens):
        self.suppress_tokens = paddle.to_tensor(list(suppress_tokens))

    def __call__(self, input_ids: paddle.Tensor, scores: paddle.Tensor) -> paddle.Tensor:
        vocab_tensor = paddle.arange(scores.shape[-1])
        suppress_token_mask = paddle.isin(vocab_tensor, self.suppress_tokens)
        scores_processed = paddle.where(suppress_token_mask, -float("inf"), scores)
        return scores_processed
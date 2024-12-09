import paddle


class BCELoss(paddle.nn.Layer):

    def forward(self, prediction, target):
        loss = paddle.nn.functional.binary_cross_entropy_with_logits(logit=
            prediction, label=target)
        return loss, {}


class BCELossWithQuant(paddle.nn.Layer):

    def __init__(self, codebook_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight

    def forward(self, qloss, target, prediction, split):
        bce_loss = paddle.nn.functional.binary_cross_entropy_with_logits(logit
            =prediction, label=target)
        loss = bce_loss + self.codebook_weight * qloss
        return loss, {'{}/total_loss'.format(split): loss.clone().detach().
            mean(), '{}/bce_loss'.format(split): bce_loss.detach().mean(),
            '{}/quant_loss'.format(split): qloss.detach().mean()}

import paddle
import paddlenlp
from paddlenlp.transformers import XLMRobertaConfig, XLMRobertaModel


class MCLIPConfig(XLMRobertaConfig):
    model_type = 'M-CLIP'

    def __init__(self, transformerDimSize=1024, imageDimSize=768, **kwargs):
        self.transformerDimensions = transformerDimSize
        self.numDims = imageDimSize
        super().__init__(**kwargs)


class MultilingualCLIP(paddlenlp.transformers.PreTrainedModel):
    config_class = MCLIPConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.transformer = XLMRobertaModel(config)
        self.LinearTransformation = paddle.nn.Linear(
            in_features=config.transformerDimensions,
            out_features=config.numDims)

    def forward(self, input_ids, attention_mask):
        embs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask)[0]
        embs2 = (embs * attention_mask.unsqueeze(axis=2)).sum(
            axis=1) / attention_mask.sum(axis=1)[:, (None)]
        return self.LinearTransformation(embs2), embs

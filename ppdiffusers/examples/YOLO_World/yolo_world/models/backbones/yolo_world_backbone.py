import itertools
from typing import List, Sequence, Tuple

import paddle.nn as nn
from paddle import Tensor
from ppdet.core.workspace import register
from ppdiffusers.transformers import (AutoTokenizer, CLIPTextConfig)
from ppdiffusers.transformers import CLIPTextModelWithProjection as CLIPTP

from ppdet.modeling.backbones.yolov8_csp_darknet import YOLOv8CSPDarkNet


@register
class HuggingCLIPLanguageBackbone(nn.Layer):

    def __init__(self,
                 model_name: str,
                 frozen_modules: Sequence[str] = (),
                 dropout: float = 0.0,
                 training_use_cache: bool = False) -> None:

        super().__init__()

        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        clip_config = CLIPTextConfig.from_pretrained(model_name, attention_dropout=dropout)
        self.model = CLIPTP.from_pretrained(model_name, config=clip_config)
        self._freeze_modules()

    def forward_tokenizer(self, texts):
        if not hasattr(self, 'text'):
            text = list(itertools.chain(*texts))
            text = self.tokenizer(text=text, return_tensors='pd', padding=True)
        return self.text

    def forward(self, text: List[List[str]]) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')
        text = list(itertools.chain(*text))
        text = self.tokenizer(text=text, return_tensors='pd', padding=True)
        print(self.model.device)
        txt_outputs = self.model(**text)
        txt_feats = txt_outputs.text_embeds
        txt_feats = txt_feats / txt_feats.norm(p=2, axis=-1, keepdim=True)
        txt_feats = txt_feats.reshape([-1, num_per_batch[0], txt_feats.shape[-1]])
        return txt_feats

    def _freeze_modules(self):

        if len(self.frozen_modules) == 0:
            # not freeze
            return
        if self.frozen_modules[0] == "all":
            self.model.eval()
            for _, module in self.model.named_sublayers():
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            return
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break


@register
class MultiModalYOLOBackbone(nn.Layer):
    __inject__ = ['image_model', 'text_model']

    def __init__(self,
                 image_model,
                 text_model,
                 frozen_stages: int = -1,
                 with_text_model: bool = True) -> None:
        super().__init__()
        self.with_text_model = with_text_model
        self.image_model = image_model
        if self.with_text_model:
            self.text_model = text_model
        else:
            self.text_model = None
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze the parameters of the specified stage so that they are no
        longer updated."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self.image_model, self.image_model.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


    def forward(self, data) -> Tuple[Tuple[Tensor], Tensor]:
        text = data['texts']
        img_feats = self.image_model(data)
        if self.with_text_model:
            txt_feats = self.text_model(text)
            return img_feats, txt_feats
        else:
            return img_feats, None

    def forward_text(self, text: List[List[str]]) -> Tensor:
        assert self.with_text_model, "forward_text() requires a text model"
        txt_feats = self.text_model(text)
        return txt_feats

    def forward_image(self, image: Tensor) -> Tuple[Tensor]:
        return self.image_model(image)


if __name__ == "__main__":
    text_model = CLIPLanguageBackbone("openai/clip-vit-base-patch32")
    text_model.eval()
    print(text_model.forward([["hi"]]))
    image_model = YOLOv8CSPDarkNet()

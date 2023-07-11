import paddle
from typing import Optional, Sequence, Tuple
from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


class ResizeMaxSize(paddle.nn.Layer):
    def __init__(self, max_size, interpolation="bicubic", fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f'Size should be int. Got {type(max_size)}')
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, paddle.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = paddle.vision.transforms.resize(img, new_size,
                                                  self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = paddle.vision.transforms.pad(img,
                                               padding=[
                                                   pad_w // 2, pad_h // 2,
                                                   pad_w - pad_w // 2,
                                                   pad_h - pad_h // 2
                                               ],
                                               fill=self.fill)
        return img


def _convert_to_rgb(image):
    return image.convert('RGB')


def image_transform(image_size: int,
                    is_train: bool,
                    mean: Optional[Tuple[float, ...]]=None,
                    std: Optional[Tuple[float, ...]]=None,
                    resize_longest_max: bool=False,
                    fill_color: int=0):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean, ) * 3
    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std, ) * 3
    if isinstance(image_size,
                  (list, tuple)) and image_size[0] == image_size[1]:
        image_size = image_size[0]
    normalize = paddle.vision.transforms.Normalize(mean=mean, std=std)
    if is_train:
        return paddle.vision.transforms.Compose([
            paddle.vision.transforms.RandomResizedCrop(
                image_size, scale=(0.9, 1.0), interpolation="bicubic"),
            _convert_to_rgb, paddle.vision.transforms.ToTensor(), normalize
        ])
    else:
        if resize_longest_max:
            transforms = [ResizeMaxSize(image_size, fill=fill_color)]
        else:
            transforms = [
                paddle.vision.transforms.Resize(
                    image_size, interpolation="bicubic"),
                paddle.vision.transforms.CenterCrop(image_size)
            ]
        transforms.extend(
            [_convert_to_rgb, paddle.vision.transforms.ToTensor(), normalize])
        return paddle.vision.transforms.Compose(transforms)

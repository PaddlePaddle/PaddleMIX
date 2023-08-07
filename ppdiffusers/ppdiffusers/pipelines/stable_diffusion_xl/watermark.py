import paddle
import numpy as np
from ...utils import is_invisible_watermark_available
if is_invisible_watermark_available():
    from imwatermark import WatermarkEncoder
WATERMARK_MESSAGE = 197828617679262
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]


class StableDiffusionXLWatermarker:
    def __init__(self):
        self.watermark = WATERMARK_BITS
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark('bits', self.watermark)

    def apply_watermark(self, images: paddle.Tensor):
        if images.shape[-1] < 256:
            return images
        images = (255 * (images / 2 + 0.5)).cpu().transpose(
            perm=[0, 2, 3, 1]).astype(dtype='float32').numpy()
        images = [self.encoder.encode(image, 'dwtDct') for image in images]
        images = paddle.to_tensor(data=np.array(images)).transpose(
            perm=[0, 3, 1, 2])
        images = paddle.clip(x=2 * (images / 255 - 0.5), min=-1.0, max=1.0)
        return images

#!/usr/bin/env python3
from numbers import Integral
from typing import Sequence
# from examples.YOLO_World.third_party.PaddleYOLO.ppdet.data.transform.operators import LetterResize, YOLOv5KeepRatioResize
import numpy as np
import cv2


class YOLOv5KeepRatioResize():
    # only used for yolov5 rect eval to get higher mAP
    # only apply to image

    def __init__(self,
                 target_size,
                 keep_ratio=True,
                 batch_shapes=True,
                 size_divisor=32,
                 extra_pad_ratio=0.5):
        super(YOLOv5KeepRatioResize, self).__init__()
        assert keep_ratio == True
        self.keep_ratio = keep_ratio
        self.batch_shapes = batch_shapes
        if not isinstance(target_size, (Integral, Sequence)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or Tuple, now is {}".
                format(type(target_size)))
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size

        self.size_divisor = size_divisor
        self.extra_pad_ratio = extra_pad_ratio

    def _get_rescale_ratio(self, old_size, scale):
        w, h = old_size
        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
            scale_factor = scale
        elif isinstance(scale, (tuple, list)):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            scale_factor = min(max_long_edge / max(h, w),
                               max_short_edge / min(h, w))
        else:
            raise TypeError('Scale must be a number or tuple of int, '
                            f'but got {type(scale)}')
        return scale_factor

    def apply_image(self, image):
        original_h, original_w = image.shape[:2]
        ratio = self._get_rescale_ratio((original_h, original_w),
                                        self.target_size)
        if ratio != 1:
            # resize image according to the shape
            # NOTE: We are currently testing on COCO that modifying
            # this code will not affect the results.
            # If you find that it has an effect on your results,
            # please feel free to contact us.
            image = cv2.resize(
                image, (int(original_w * ratio), int(original_h * ratio)),
                interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)

        resized_h, resized_w = image.shape[:2]
        scale_ratio_h = resized_h / original_h
        scale_ratio_w = resized_w / original_w
        return image, (resized_h, resized_w), (scale_ratio_h, scale_ratio_w)

    def apply(self, sample, context=None):
        """ Resize the image numpy.
        """
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))

        im, (resize_h, resize_w), (
            scale_ratio_h, scale_ratio_w) = self.apply_image(sample['image'])
        # (427, 640) (480, 640)
        sample['image'] = im.astype(np.float32)
        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        sample['scale_factor'] = np.asarray(
            [scale_ratio_h, scale_ratio_w], dtype=np.float32)

        shapes = [[1, 1]]
        aspect_ratio = resize_h / resize_w
        shapes = [aspect_ratio,
                  1] if aspect_ratio < 1 else [1, 1 / aspect_ratio]
        batch_shapes = np.ceil(
            np.array(shapes) * 640 / self.size_divisor +
            self.extra_pad_ratio).astype(np.int64) * self.size_divisor
        sample['batch_shape'] = batch_shapes
        return sample


class LetterResize():
    # only used for yolov5 rect eval to get higher mAP
    # only apply to image

    def __init__(self,
                 scale=[640, 640],
                 pad_val=144,
                 use_mini_pad=False,
                 stretch_only=False,
                 allow_scale_up=False,
                 half_pad_param=False):
        super(LetterResize, self).__init__()
        self.scale = scale
        self.pad_val = pad_val
        if isinstance(pad_val, (int, float)):
            pad_val = dict(img=pad_val, seg=255)
        assert isinstance(
            pad_val, dict), f'pad_val must be dict, but got {type(pad_val)}'

        self.use_mini_pad = use_mini_pad
        self.stretch_only = stretch_only
        self.allow_scale_up = allow_scale_up
        self.half_pad_param = half_pad_param

    def _resize_img(self, results):
        image = results['image']
        # Use batch_shape if a batch_shape policy is configured
        if 'batch_shape' in results:
            scale = tuple(results['batch_shape'])  # hw
        else:
            scale = self.scale[::-1]  # wh -> hw

        image_shape = image.shape[:2]  # height, width

        # Scale ratio (new / old)
        ratio = min(scale[0] / image_shape[0], scale[1] / image_shape[1])
        # (448, 672) / (427, 640) = 1.0491803278688525
        # (512, 672) / (480, 640) = 1.05

        # only scale down, do not scale up (for better test mAP)
        if not self.allow_scale_up:
            ratio = min(ratio, 1.0)

        ratio = [ratio, ratio]  # float -> (float, float) for (height, width)

        # compute the best size of the image
        no_pad_shape = (int(round(image_shape[0] * ratio[0])),
                        int(round(image_shape[1] * ratio[1])))
        # [427, 640]  [480, 640]
        # padding height & width
        padding_h, padding_w = [
            scale[0] - no_pad_shape[0], scale[1] - no_pad_shape[1]
        ]  # [21, 32] 32, 32
        if self.use_mini_pad:
            # minimum rectangle padding
            padding_w, padding_h = np.mod(padding_w, 32), np.mod(padding_h, 32)
        elif self.stretch_only:
            # stretch to the specified size directly
            padding_h, padding_w = 0.0, 0.0
            no_pad_shape = (scale[0], scale[1])
            ratio = [scale[0] / image_shape[0],
                     scale[1] / image_shape[1]]  # height, width ratios

        if image_shape != no_pad_shape:
            # compare with no resize and padding size
            image = cv2.resize(
                image, (no_pad_shape[1], no_pad_shape[0]),
                interpolation='bilinear')

        scale_factor = (no_pad_shape[1] / image_shape[1],
                        no_pad_shape[0] / image_shape[0])

        if 'scale_factor' in results:
            results['scale_factor_origin'] = results['scale_factor']
        results['scale_factor'] = scale_factor

        # padding
        top_padding, left_padding = int(round(padding_h // 2 - 0.1)), int(
            round(padding_w // 2 - 0.1))
        bottom_padding = padding_h - top_padding
        right_padding = padding_w - left_padding

        padding_list = [
            top_padding, bottom_padding, left_padding, right_padding
        ]  # [10, 11, 16, 16]  [16, 16, 16, 16]
        if top_padding != 0 or bottom_padding != 0 or \
                left_padding != 0 or right_padding != 0:
            if isinstance(self.pad_val, int) and image.ndim == 3:
                self.pad_val = tuple(
                    self.pad_val for _ in range(image.shape[2]))
            # image = cv2.impad(
            #     img=image,
            #     padding=(padding_list[2], padding_list[0], padding_list[3],
            #              padding_list[1]),
            #     pad_val=pad_val,
            #     padding_mode='constant')
            top, bottom, left, right = padding_list
            image = cv2.copyMakeBorder(
                image,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=self.pad_val)
            # (448, 672, 3)

        results['image'] = image.astype(np.float32)
        results['im0_shape'] = np.asarray(image_shape, dtype=np.float32)
        results['im_shape'] = np.asarray([image.shape[:2]], dtype=np.float32)

        if 'pad_param' in results:
            results['pad_param_origin'] = results['pad_param'] * \
                                          np.repeat(ratio, 2)

        if self.half_pad_param:
            results['pad_param'] = np.array(
                [padding_h / 2, padding_h / 2, padding_w / 2, padding_w / 2],
                dtype=np.float32)
        else:
            # We found in object detection, using padding list with
            # int type can get higher mAP.
            results['pad_param'] = np.array(padding_list, dtype=np.float32)

        return results

    def apply(self, sample, context=None):
        sample = self._resize_img(sample)
        # if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
        #     sample = self._resize_bboxes(sample)
        if 'scale_factor_origin' in sample:
            scale_factor_origin = sample.pop('scale_factor_origin')
            scale_ratio_h, scale_ratio_w = (
                sample['scale_factor'][0] * scale_factor_origin[0],
                sample['scale_factor'][1] * scale_factor_origin[1])
            sample['scale_factor'] = np.asarray(
                [scale_ratio_h, scale_ratio_w], dtype=np.float32)

        if 'pad_param_origin' in sample:
            pad_param_origin = sample.pop('pad_param_origin')
            sample['pad_param'] += pad_param_origin

        return sample

a=np.random.random((1421, 947, 3))
sample = {}
resize = YOLOv5KeepRatioResize(target_size=(640,640))
sample["image"] = a
sample = resize.apply(sample)
print(sample)
# resize = LetterResize(scale=[640, 640], pad_val=114)
# sample = resize.apply(sample)
# print(sample)

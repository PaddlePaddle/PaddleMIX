# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import cv2
import math
import numpy as np
import paddle
import yaml
import time
import codecs
import ast
import argparse
import decord
from decord import VideoReader, cpu
from paddle.inference import Config
from paddle.inference import create_predictor



KEYPOINT_SUPPORT_MODELS = {
    'HigherHRNet': 'keypoint_bottomup',
    'HRNet': 'keypoint_topdown',
    'VitPose_TopDown_WholeBody': 'keypoint_topdown_wholebody'
}


# Global dictionary
SUPPORT_MODELS = {
    'YOLO', 'PPYOLOE', 'RCNN', 'SSD', 'Face', 'FCOS', 'SOLOv2', 'TTFNet',
    'S2ANet', 'JDE', 'FairMOT', 'DeepSORT', 'GFL', 'PicoDet', 'CenterNet',
    'TOOD', 'RetinaNet', 'StrongBaseline', 'STGCN', 'YOLOX', 'YOLOF', 'PPHGNet',
    'PPLCNet', 'DETR', 'CenterTrack', 'CLRNet'
}




def create_inputs(imgs, im_info):
    """generate input for different model type
    Args:
        imgs (list(numpy)): list of images (np.ndarray)
        im_info (list(dict)): list of image info
    Returns:
        inputs (dict): input of model
    """
    inputs = {}

    im_shape = []
    scale_factor = []
    if len(imgs) == 1:
        inputs['image'] = np.array((imgs[0], )).astype('float32')
        inputs['im_shape'] = np.array(
            (im_info[0]['im_shape'], )).astype('float32')
        inputs['scale_factor'] = np.array(
            (im_info[0]['scale_factor'], )).astype('float32')
        return inputs

    for e in im_info:
        im_shape.append(np.array((e['im_shape'], )).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))

    inputs['im_shape'] = np.concatenate(im_shape, axis=0)
    inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = img
        padding_imgs.append(padding_im)
    inputs['image'] = np.stack(padding_imgs, axis=0)
    return inputs

def preprocess(im, preprocess_ops):
    # process image by preprocess_ops
    im_info = {
        'scale_factor': np.array(
            [1., 1.], dtype=np.float32),
        'im_shape': None,
    }
    im, im_info = decode_image(im, im_info)
    for operator in preprocess_ops:
        im, im_info = operator(im, im_info)
    return im, im_info

class Permute(object):
    """permute image
    Args:
        to_bgr (bool): whether convert RGB to BGR 
        channel_first (bool): whether convert HWC to CHW
    """

    def __init__(self, ):
        super(Permute, self).__init__()

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.transpose((2, 0, 1)).copy()
        return im, im_info


class NormalizeImage(object):
    """normalize image
    Args:
        mean (list): im - mean
        std (list): im / std
        is_scale (bool): whether need im / 255
        norm_type (str): type in ['mean_std', 'none']
    """

    def __init__(self, mean, std, is_scale=True, norm_type='mean_std'):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.norm_type = norm_type

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        print("preprocess NormalizeImage")
        im = im.astype(np.float32, copy=False)
        if self.is_scale:
            scale = 1.0 / 255.0
            im *= scale

        if self.norm_type == 'mean_std':
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
            im -= mean
            im /= std
        return im, im_info



class Resize(object):
    """resize image by target_size and max_size
    Args:
        target_size (int): the target size of image
        keep_ratio (bool): whether keep_ratio or not, default true
        interp (int): method of resize
    """

    def __init__(self, target_size, keep_ratio=True, interp=cv2.INTER_LINEAR):
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.interp = interp

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        assert len(self.target_size) == 2
        assert self.target_size[0] > 0 and self.target_size[1] > 0
        im_channel = im.shape[2]
        im_scale_y, im_scale_x = self.generate_scale(im)
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)
        im_info['im_shape'] = np.array(im.shape[:2]).astype('float32')
        im_info['scale_factor'] = np.array(
            [im_scale_y, im_scale_x]).astype('float32')
        return im, im_info

    def generate_scale(self, im):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
        Returns:
            im_scale_x: the resize ratio of X
            im_scale_y: the resize ratio of Y
        """
        origin_shape = im.shape[:2]
        im_c = im.shape[2]
        if self.keep_ratio:
            im_size_min = np.min(origin_shape)
            im_size_max = np.max(origin_shape)
            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)
            im_scale = float(target_size_min) / float(im_size_min)
            if np.round(im_scale * im_size_max) > target_size_max:
                im_scale = float(target_size_max) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / float(origin_shape[0])
            im_scale_x = resize_w / float(origin_shape[1])
        return im_scale_y, im_scale_x


class Times(object):
    def __init__(self):
        self.time = 0.
        # start time
        self.st = 0.
        # end time
        self.et = 0.

    def start(self):
        self.st = time.time()

    def end(self, repeats=1, accumulative=True):
        self.et = time.time()
        if accumulative:
            self.time += (self.et - self.st) / repeats
        else:
            self.time = (self.et - self.st) / repeats

    def reset(self):
        self.time = 0.
        self.st = 0.
        self.et = 0.

    def value(self):
        return round(self.time, 4)


class Timer(Times):
    def __init__(self, with_tracker=False):
        super(Timer, self).__init__()
        self.with_tracker = with_tracker
        self.preprocess_time_s = Times()
        self.inference_time_s = Times()
        self.postprocess_time_s = Times()
        self.tracking_time_s = Times()
        self.img_num = 0

    def info(self, average=False):
        pre_time = self.preprocess_time_s.value()
        infer_time = self.inference_time_s.value()
        post_time = self.postprocess_time_s.value()
        track_time = self.tracking_time_s.value()

        total_time = pre_time + infer_time + post_time
        if self.with_tracker:
            total_time = total_time + track_time
        total_time = round(total_time, 4)
        print("------------------ Inference Time Info ----------------------")
        print("total_time(ms): {}, img_num: {}".format(total_time * 1000,
                                                       self.img_num))
        preprocess_time = round(pre_time / max(1, self.img_num),
                                4) if average else pre_time
        postprocess_time = round(post_time / max(1, self.img_num),
                                 4) if average else post_time
        inference_time = round(infer_time / max(1, self.img_num),
                               4) if average else infer_time
        tracking_time = round(track_time / max(1, self.img_num),
                              4) if average else track_time

        average_latency = total_time / max(1, self.img_num)
        qps = 0
        if total_time > 0:
            qps = 1 / average_latency
        print("average latency time(ms): {:.2f}, QPS: {:2f}".format(
            average_latency * 1000, qps))
        if self.with_tracker:
            print(
                "preprocess_time(ms): {:.2f}, inference_time(ms): {:.2f}, postprocess_time(ms): {:.2f}, tracking_time(ms): {:.2f}".
                format(preprocess_time * 1000, inference_time * 1000,
                       postprocess_time * 1000, tracking_time * 1000))
        else:
            print(
                "preprocess_time(ms): {:.2f}, inference_time(ms): {:.2f}, postprocess_time(ms): {:.2f}".
                format(preprocess_time * 1000, inference_time * 1000,
                       postprocess_time * 1000))

    def report(self, average=False):
        dic = {}
        pre_time = self.preprocess_time_s.value()
        infer_time = self.inference_time_s.value()
        post_time = self.postprocess_time_s.value()
        track_time = self.tracking_time_s.value()

        dic['preprocess_time_s'] = round(pre_time / max(1, self.img_num),
                                         4) if average else pre_time
        dic['inference_time_s'] = round(infer_time / max(1, self.img_num),
                                        4) if average else infer_time
        dic['postprocess_time_s'] = round(post_time / max(1, self.img_num),
                                          4) if average else post_time
        dic['img_num'] = self.img_num
        total_time = pre_time + infer_time + post_time
        if self.with_tracker:
            dic['tracking_time_s'] = round(track_time / max(1, self.img_num),
                                           4) if average else track_time
            total_time = total_time + track_time
        dic['total_time_s'] = round(total_time, 4)
        return dic


def load_predictor(model_dir,
                   arch,
                   run_mode='paddle',
                   batch_size=1,
                   device='CPU',
                   min_subgraph_size=3,
                   use_dynamic_shape=False,
                   trt_min_shape=1,
                   trt_max_shape=1280,
                   trt_opt_shape=640,
                   trt_calib_mode=False,
                   cpu_threads=1,
                   enable_mkldnn=False,
                   enable_mkldnn_bfloat16=False,
                   delete_shuffle_pass=False):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16/trt_int8)
        use_dynamic_shape (bool): use dynamic shape or not
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        delete_shuffle_pass (bool): whether to remove shuffle_channel_detect_pass in TensorRT. 
                                    Used by action model.
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need device == 'GPU'.
    """
    if device != 'GPU' and run_mode != 'paddle':
        raise ValueError(
            "Predict by TensorRT mode: {}, expect device=='GPU', but device == {}"
            .format(run_mode, device))

    if paddle.__version__ >= '3.0.0' or paddle.__version__ == '0.0.0':
        model_path = model_dir
        model_prefix = 'model'
        infer_param = os.path.join(model_dir, 'model.pdiparams')
        if not os.path.exists(infer_param):
            model_prefix = 'inference'
            if paddle.framework.use_pir_api():
                infer_model = os.path.join(model_dir, 'inference.pdmodel')
            else:
                infer_model = os.path.join(model_dir, 'inference.json')
            if not os.path.exists(infer_model):
                raise ValueError(
                    "Cannot find any inference model in dir: {}.".format(model_dir))
        config = Config(model_path, model_prefix)
        
    else:
        infer_model = os.path.join(model_dir, 'model.pdmodel')
        infer_params = os.path.join(model_dir, 'model.pdiparams')
        if not os.path.exists(infer_model):
            infer_model = os.path.join(model_dir, 'inference.pdmodel')
            infer_params = os.path.join(model_dir, 'inference.pdiparams')
            if not os.path.exists(infer_model):
                raise ValueError(
                    "Cannot find any inference model in dir: {},".format(model_dir))
        config = Config(infer_model, infer_params)
 
    if device == 'GPU':
        # initial GPU memory(M), device ID
        config.enable_use_gpu(200, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    elif device == 'XPU':
        if config.lite_engine_enabled():
            config.enable_lite_engine()
        config.enable_xpu(10 * 1024 * 1024)
    elif device == 'NPU':
        config.enable_custom_device('npu')
    elif device == 'MLU':
        config.enable_custom_device('mlu')
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)
        if enable_mkldnn:
            try:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
                if enable_mkldnn_bfloat16:
                    config.enable_mkldnn_bfloat16()
            except Exception as e:
                print(
                    "The current environment does not support `mkldnn`, so disable mkldnn."
                )
                pass

    precision_map = {
        'trt_int8': Config.Precision.Int8,
        'trt_fp32': Config.Precision.Float32,
        'trt_fp16': Config.Precision.Half
    }
    if run_mode in precision_map.keys():
        config.enable_tensorrt_engine(
            workspace_size=(1 << 25) * batch_size,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[run_mode],
            use_static=False,
            use_calib_mode=trt_calib_mode)
        if FLAGS.collect_trt_shape_info:
            config.collect_shape_range_info(FLAGS.tuned_trt_shape_file)
        elif os.path.exists(FLAGS.tuned_trt_shape_file):
            print(f'Use dynamic shape file: '
                  f'{FLAGS.tuned_trt_shape_file} for TRT...')
            config.enable_tuned_tensorrt_dynamic_shape(
                FLAGS.tuned_trt_shape_file, True)

        if use_dynamic_shape:
            min_input_shape = {
                'image': [batch_size, 3, trt_min_shape, trt_min_shape],
                'scale_factor': [batch_size, 2]
            }
            max_input_shape = {
                'image': [batch_size, 3, trt_max_shape, trt_max_shape],
                'scale_factor': [batch_size, 2]
            }
            opt_input_shape = {
                'image': [batch_size, 3, trt_opt_shape, trt_opt_shape],
                'scale_factor': [batch_size, 2]
            }
            config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape,
                                              opt_input_shape)
            print('trt set dynamic shape done!')

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    if delete_shuffle_pass:
        config.delete_pass("shuffle_channel_detect_pass")
    predictor = create_predictor(config)
    return predictor, config




def _box2cs(image_size, box):
    """This encodes bbox(x,y,w,h) into (center, scale)

    Args:
        x, y, w, h

    Returns:
        tuple: A tuple containing center and scale.

        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = box[:4]
    input_size = image_size
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
    scale = scale * 1.25

    return center, scale


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt


def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt


def get_affine_transform(center,
                         input_size,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ]): Size of the destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(output_size) == 2
    assert len(shift) == 2
    if not isinstance(input_size, (np.ndarray, list)):
        input_size = np.array([input_size, input_size], dtype=np.float32)
    scale_tmp = input_size

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

class TopDownAffineImage(object):
    """apply affine transform to image and coords

    Args:
        trainsize (list): [w, h], the standard size used to train
        use_udp (bool): whether to use Unbiased Data Processing.
        records(dict): the dict contained the image and coords

    Returns:
        records (dict): contain the image and coords after transformed

    """

    def __init__(self, trainsize, use_udp=False, use_box2cs=True):
        self.trainsize = trainsize
        self.use_udp = use_udp
        self.use_box2cs = use_box2cs

    def __call__(self, records, im_info):
        if self.use_box2cs:
            center, scale = _box2cs(self.trainsize, [0,0,im_info['im_shape'][1],im_info['im_shape'][0]]) 
        else:
            imshape = im_info['im_shape'][::-1]
            center = im_info['center'] if 'center' in im_info else imshape / 2.
            scale = im_info['scale'] if 'scale' in im_info else imshape

        image = records
        rot = records['rotate'] if "rotate" in records else 0
        if self.use_udp:
            trans = get_warp_matrix(
                rot, center * 2.0,
                [self.trainsize[0] - 1.0, self.trainsize[1] - 1.0],
                scale * 200.0)
            image = cv2.warpAffine(
                image,
                trans, (int(self.trainsize[0]), int(self.trainsize[1])),
                flags=cv2.INTER_LINEAR)
            joints[:, 0:2] = warp_affine_joints(joints[:, 0:2].copy(), trans)
        else:
            trans = get_affine_transform(center, scale *
                                         200, rot, self.trainsize)
            image = cv2.warpAffine(
                image,
                trans, (int(self.trainsize[0]), int(self.trainsize[1])),
                flags=cv2.INTER_LINEAR)
        return image, im_info



class Detector(object):
    """
    Args:
        pred_config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of pre batch in inference
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
        enable_mkldnn_bfloat16 (bool): whether to turn on mkldnn bfloat16
        output_dir (str): The path of output
        threshold (float): The threshold of score for visualization
        delete_shuffle_pass (bool): whether to remove shuffle_channel_detect_pass in TensorRT. 
                                    Used by action model.
    """

    def __init__(self,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 enable_mkldnn_bfloat16=False,
                 output_dir='output',
                 threshold=0.5,
                 delete_shuffle_pass=False,
                 use_fd_format=False):
        self.pred_config = self.set_config(
            model_dir, use_fd_format=use_fd_format)
        self.predictor, self.config = load_predictor(
            model_dir,
            self.pred_config.arch,
            run_mode=run_mode,
            batch_size=batch_size,
            min_subgraph_size=self.pred_config.min_subgraph_size,
            device=device,
            use_dynamic_shape=self.pred_config.use_dynamic_shape,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            enable_mkldnn_bfloat16=enable_mkldnn_bfloat16,
            delete_shuffle_pass=delete_shuffle_pass)
        self.det_times = Timer()
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.threshold = threshold
        self.device = device

    def set_config(self, model_dir, use_fd_format):
        return PredictConfig(model_dir, use_fd_format=use_fd_format)

    def preprocess(self, image_list):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))

        input_im_lst = []
        input_im_info_lst = []
        for im_path in image_list:
            im, im_info = preprocess(im_path, preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            if input_names[i] == 'x':
                input_tensor.copy_from_cpu(inputs['image'])
            else:
                input_tensor.copy_from_cpu(inputs[input_names[i]])

        return inputs

    def postprocess(self, inputs, result):
        # postprocess output of predictor
        np_boxes_num = result['boxes_num']
        assert isinstance(np_boxes_num, np.ndarray), \
            '`np_boxes_num` should be a `numpy.ndarray`'

        result = {k: v for k, v in result.items() if v is not None}
        return result

    def filter_box(self, result, threshold):
        np_boxes_num = result['boxes_num']
        boxes = result['boxes']
        start_idx = 0
        filter_boxes = []
        filter_num = []
        for i in range(len(np_boxes_num)):
            boxes_num = np_boxes_num[i]
            boxes_i = boxes[start_idx:start_idx + boxes_num, :]
            idx = boxes_i[:, 1] > threshold
            filter_boxes_i = boxes_i[idx, :]
            filter_boxes.append(filter_boxes_i)
            filter_num.append(filter_boxes_i.shape[0])
            start_idx += boxes_num
        boxes = np.concatenate(filter_boxes)
        filter_num = np.array(filter_num)
        filter_res = {'boxes': boxes, 'boxes_num': filter_num}
        return filter_res

    def predict(self, repeats=1, run_benchmark=False):
        '''
        Args:
            repeats (int): repeats number for prediction
        Returns:
            result (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matrix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's result include 'masks': np.ndarray:
                            shape: [N, im_h, im_w]
        '''
        # model prediction
        np_boxes_num, np_boxes, np_masks = np.array([0]), None, None

        if run_benchmark:
            for i in range(repeats):
                self.predictor.run()
                if self.device == 'GPU':
                    paddle.device.cuda.synchronize()
                else:
                    paddle.device.synchronize(device=self.device.lower())
                
            result = dict(
                boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
            return result

        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            if len(output_names) == 1:
                # some exported model can not get tensor 'bbox_num' 
                np_boxes_num = np.array([len(np_boxes)])
            else:
                boxes_num = self.predictor.get_output_handle(output_names[1])
                np_boxes_num = boxes_num.copy_to_cpu()
            if self.pred_config.mask:
                masks_tensor = self.predictor.get_output_handle(output_names[2])
                np_masks = masks_tensor.copy_to_cpu()
        result = dict(boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
        return result

    def merge_batch_result(self, batch_result):
        if len(batch_result) == 1:
            return batch_result[0]
        res_key = batch_result[0].keys()
        results = {k: [] for k in res_key}
        for res in batch_result:
            for k, v in res.items():
                results[k].append(v)
        for k, v in results.items():
            if k not in ['masks', 'segm']:
                results[k] = np.concatenate(v)
        return results

    def get_timer(self):
        return self.det_times

    def predict_image_slice(self,
                            img_list,
                            slice_size=[640, 640],
                            overlap_ratio=[0.25, 0.25],
                            combine_method='nms',
                            match_threshold=0.6,
                            match_metric='ios',
                            run_benchmark=False,
                            repeats=1,
                            visual=True,
                            save_results=False):
        # slice infer only support bs=1
        results = []
        try:
            import sahi
            from sahi.slicing import slice_image
        except Exception as e:
            print(
                'sahi not found, plaese install sahi. '
                'for example: `pip install sahi`, see https://github.com/obss/sahi.'
            )
            raise e
        num_classes = len(self.pred_config.labels)
        for i in range(len(img_list)):
            ori_image = img_list[i]
            slice_image_result = sahi.slicing.slice_image(
                image=ori_image,
                slice_height=slice_size[0],
                slice_width=slice_size[1],
                overlap_height_ratio=overlap_ratio[0],
                overlap_width_ratio=overlap_ratio[1])
            sub_img_num = len(slice_image_result)
            merged_bboxs = []
            print('slice to {} sub_samples.', sub_img_num)

            batch_image_list = [
                slice_image_result.images[_ind] for _ind in range(sub_img_num)
            ]
            if run_benchmark:
                # preprocess
                inputs = self.preprocess(batch_image_list)  # warmup
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                result = self.predict(repeats=50, run_benchmark=True)  # warmup
                self.det_times.inference_time_s.start()
                result = self.predict(repeats=repeats, run_benchmark=True)
                self.det_times.inference_time_s.end(repeats=repeats)

                # postprocess
                result_warmup = self.postprocess(inputs, result)  # warmup
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += 1

                cm, gm, gu = get_current_memory_mb()
                self.cpu_mem += cm
                self.gpu_mem += gm
                self.gpu_util += gu
            else:
                # preprocess
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                self.det_times.inference_time_s.start()
                result = self.predict()
                self.det_times.inference_time_s.end()

                # postprocess
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += 1

            st, ed = 0, result['boxes_num'][0]  # start_index, end_index
            for _ind in range(sub_img_num):
                boxes_num = result['boxes_num'][_ind]
                ed = st + boxes_num
                shift_amount = slice_image_result.starting_pixels[_ind]
                result['boxes'][st:ed][:, 2:4] = result['boxes'][
                    st:ed][:, 2:4] + shift_amount
                result['boxes'][st:ed][:, 4:6] = result['boxes'][
                    st:ed][:, 4:6] + shift_amount
                merged_bboxs.append(result['boxes'][st:ed])
                st = ed

            merged_results = {'boxes': []}
            if combine_method == 'nms':
                final_boxes = multiclass_nms(
                    np.concatenate(merged_bboxs), num_classes, match_threshold,
                    match_metric)
                merged_results['boxes'] = np.concatenate(final_boxes)
            elif combine_method == 'concat':
                merged_results['boxes'] = np.concatenate(merged_bboxs)
            else:
                raise ValueError(
                    "Now only support 'nms' or 'concat' to fuse detection results."
                )
            merged_results['boxes_num'] = np.array(
                [len(merged_results['boxes'])], dtype=np.int32)

            if visual:
                visualize(
                    [ori_image],  # should be list
                    merged_results,
                    self.pred_config.labels,
                    output_dir=self.output_dir,
                    threshold=self.threshold)

            results.append(merged_results)
            print('Test iter {}'.format(i))

        results = self.merge_batch_result(results)
        if save_results:
            Path(self.output_dir).mkdir(exist_ok=True)
            self.save_coco_results(
                img_list,
                results,
                use_coco_category=FLAGS.use_coco_category,
                task_type=FLAGS.task_type)
        return results

    def predict_image(self,
                      image_list,
                      run_benchmark=False,
                      repeats=1,
                      visual=True,
                      save_results=False):
        batch_loop_cnt = math.ceil(float(len(image_list)) / self.batch_size)
        results = []
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(image_list))
            batch_image_list = image_list[start_index:end_index]
            if run_benchmark:
                # preprocess
                inputs = self.preprocess(batch_image_list)  # warmup
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                result = self.predict(repeats=50, run_benchmark=True)  # warmup
                self.det_times.inference_time_s.start()
                result = self.predict(repeats=repeats, run_benchmark=True)
                self.det_times.inference_time_s.end(repeats=repeats)

                # postprocess
                result_warmup = self.postprocess(inputs, result)  # warmup
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += len(batch_image_list)

                cm, gm, gu = get_current_memory_mb()
                self.cpu_mem += cm
                self.gpu_mem += gm
                self.gpu_util += gu
            else:
                # preprocess
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                self.det_times.inference_time_s.start()
                result = self.predict()
                self.det_times.inference_time_s.end()

                # postprocess
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += len(batch_image_list)

                if visual:
                    visualize(
                        batch_image_list,
                        result,
                        self.pred_config.labels,
                        output_dir=self.output_dir,
                        threshold=self.threshold)
            results.append(result)
            print('Test iter {}'.format(i))
        results = self.merge_batch_result(results)
        if save_results:
            Path(self.output_dir).mkdir(exist_ok=True)
            self.save_coco_results(
                image_list,
                results,
                use_coco_category=FLAGS.use_coco_category,
                task_type=FLAGS.task_type)
        return results

    def predict_video(self, video_file, camera_id):
        video_out_name = 'output.mp4'
        if camera_id != -1:
            capture = cv2.VideoCapture(camera_id)
        else:
            capture = cv2.VideoCapture(video_file)
            video_out_name = os.path.split(video_file)[-1]
        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("fps: %d, frame_count: %d" % (fps, frame_count))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        out_path = os.path.join(self.output_dir, video_out_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        index = 1
        while (1):
            ret, frame = capture.read()
            if not ret:
                break
            print('detect frame: %d' % (index))
            index += 1
            results = self.predict_image([frame[:, :, ::-1]], visual=False)

            im = visualize_box_mask(
                frame,
                results,
                self.pred_config.labels,
                threshold=self.threshold)
            im = np.array(im)
            writer.write(im)
            if camera_id != -1:
                cv2.imshow('Mask Detection', im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        writer.release()

    def save_coco_results(self,
                          image_list,
                          results,
                          use_coco_category=False,
                          task_type='Detection'):
        bbox_results = []
        mask_results = []
        idx = 0
        print("Start saving coco json files...")
        for i, box_num in enumerate(results['boxes_num']):
            file_name = os.path.split(image_list[i])[-1]
            if use_coco_category:
                img_id = int(os.path.splitext(file_name)[0])
            else:
                img_id = i

            if 'boxes' in results:
                boxes = results['boxes'][idx:idx + box_num].tolist()
                if task_type == 'Rotate':
                    bbox = [
                        box[2], box[3], box[4], box[5], box[6], box[7], box[8],
                        box[9]
                    ]  # x1, y1, x2, y2, x3, y3, x4, y4
                else:  # default is 'Detection'
                    bbox: [box[2], box[3], box[4] - box[2],
                           box[5] - box[3]]  # xyxy -> xywh
                bbox_results.extend([{
                    'image_id': img_id,
                    'category_id': coco_clsid2catid[int(box[0])] \
                        if use_coco_category else int(box[0]),
                    'file_name': file_name,
                    'bbox': bbox,
                    'score': box[1]} for box in boxes])

            if 'masks' in results:
                import pycocotools.mask as mask_util

                boxes = results['boxes'][idx:idx + box_num].tolist()
                masks = results['masks'][i][:box_num].astype(np.uint8)
                seg_res = []
                for box, mask in zip(boxes, masks):
                    rle = mask_util.encode(
                        np.array(
                            mask[:, :, None], dtype=np.uint8, order="F"))[0]
                    if 'counts' in rle:
                        rle['counts'] = rle['counts'].decode("utf8")
                    seg_res.append({
                        'image_id': img_id,
                        'category_id': coco_clsid2catid[int(box[0])] \
                        if use_coco_category else int(box[0]),
                        'file_name': file_name,
                        'segmentation': rle,
                        'score': box[1]})
                mask_results.extend(seg_res)

            idx += box_num

        if bbox_results:
            bbox_file = os.path.join(self.output_dir, "bbox.json")
            with open(bbox_file, 'w') as f:
                json.dump(bbox_results, f)
            print(f"The bbox result is saved to {bbox_file}")
        if mask_results:
            mask_file = os.path.join(self.output_dir, "mask.json")
            with open(mask_file, 'w') as f:
                json.dump(mask_results, f)
            print(f"The mask result is saved to {mask_file}")


class PredictConfig():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir, use_fd_format=False):
        # parsing Yaml config for Preprocess
        fd_deploy_file = os.path.join(model_dir, 'inference.yml')
        ppdet_deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        if use_fd_format:
            if not os.path.exists(fd_deploy_file) and os.path.exists(
                    ppdet_deploy_file):
                raise RuntimeError(
                    "Non-FD format model detected. Please set `use_fd_format` to False."
                )
            deploy_file = fd_deploy_file
        else:
            if not os.path.exists(ppdet_deploy_file) and os.path.exists(
                    fd_deploy_file):
                raise RuntimeError(
                    "FD format model detected. Please set `use_fd_format` to False."
                )
            deploy_file = ppdet_deploy_file
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask = False
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        if 'mask' in yml_conf:
            self.mask = yml_conf['mask']
        self.tracker = None
        if 'tracker' in yml_conf:
            self.tracker = yml_conf['tracker']
        if 'NMS' in yml_conf:
            self.nms = yml_conf['NMS']
        if 'fpn_stride' in yml_conf:
            self.fpn_stride = yml_conf['fpn_stride']
        if self.arch == 'RCNN' and yml_conf.get('export_onnx', False):
            print(
                'The RCNN export model is used for ONNX and it only supports batch_size = 1'
            )
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type 
        """
        for support_model in SUPPORT_MODELS:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf[
            'arch'], SUPPORT_MODELS))

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')



def decode_image(im_file, im_info):
    """read rgb image
    Args:
        im_file (str|np.ndarray): input can be image path or np.ndarray
        im_info (dict): info of image
    Returns:
        im (np.ndarray):  processed image (np.ndarray)
        im_info (dict): info of processed image
    """
    if isinstance(im_file, str):
        with open(im_file, 'rb') as f:
            im_read = f.read()
        data = np.frombuffer(im_read, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        im = im_file
    im_info['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
    im_info['scale_factor'] = np.array([1., 1.], dtype=np.float32)
    return im, im_info

def visualize_pose(imgfile,
                   results,
                   visual_thresh=0.6,
                   save_name='pose.jpg',
                   save_dir='output',
                   returnimg=False,
                   ids=None,
                   draw_box=False):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        plt.switch_backend('agg')
    except Exception as e:
        print('Matplotlib not found, please install matplotlib.'
              'for example: `pip install matplotlib`.')
        raise e
    skeletons, scores = results['keypoint']
    skeletons = np.array(skeletons)
    kpt_nums = np.shape(skeletons)[1]
    if len(skeletons) > 0:
        kpt_nums = skeletons.shape[1]
    if kpt_nums == 17:  #plot coco keypoint
        EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8),
                 (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14),
                 (13, 15), (14, 16), (11, 12)]
    elif kpt_nums == 133:
        EDGES = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22), (91, 92), (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98), (98, 99), (91, 100), (100, 101), (101, 102), (102, 103), (91, 104), (104, 105), (105, 106), (106, 107), (91, 108), (108, 109), (109, 110), (110, 111), (112, 113), (113, 114), (114, 115), (115, 116), (112, 117), (117, 118), (118, 119), (119, 120), (112, 121), (121, 122), (122, 123), (123, 124), (112, 125), (125, 126), (126, 127), (127, 128), (112, 129), (129, 130), (130, 131), (131, 132)]
    
    else:  #plot mpii keypoint
        EDGES = [(0, 1), (1, 2), (3, 4), (4, 5), (2, 6), (3, 6), (6, 7), (7, 8),
                 (8, 9), (10, 11), (11, 12), (13, 14), (14, 15), (8, 12),
                 (8, 13)]
    NUM_EDGES = len(EDGES)
    if kpt_nums == 133:
        colors = [(51, 153, 255), (51, 153, 255), (51, 153, 255), (51, 153, 255), (51, 153, 255), (0, 255, 0), (255, 128, 0), (0, 255, 0), (255, 128, 0), (0, 255, 0), (255, 128, 0), (0, 255, 0), (255, 128, 0), (0, 255, 0), (255, 128, 0), (0, 255, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 153, 255), (255, 153, 255), (255, 153, 255), (255, 153, 255), (102, 178, 255), (102, 178, 255), (102, 178, 255), (102, 178, 255), (255, 51, 51), (255, 51, 51), (255, 51, 51), (255, 51, 51), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (255, 255, 255), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 153, 255), (255, 153, 255), (255, 153, 255), (255, 153, 255), (102, 178, 255), (102, 178, 255), (102, 178, 255), (102, 178, 255), (255, 51, 51), (255, 51, 51), (255, 51, 51), (255, 51, 51), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0)]
    else:
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    skeleton_link_colors = [(0, 255, 0), (0, 255, 0), (255, 128, 0), (255, 128, 0), (51, 153, 255), (51, 153, 255), (51, 153, 255), (51, 153, 255), (0, 255, 0), (255, 128, 0), (0, 255, 0), (255, 128, 0), (51, 153, 255), (51, 153, 255), (51, 153, 255), (51, 153, 255), (51, 153, 255), (51, 153, 255), (51, 153, 255), (0, 255, 0), (0, 255, 0), (0, 255, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 153, 255), (255, 153, 255), (255, 153, 255), (255, 153, 255), (102, 178, 255), (102, 178, 255), (102, 178, 255), (102, 178, 255), (255, 51, 51), (255, 51, 51), (255, 51, 51), (255, 51, 51), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 153, 255), (255, 153, 255), (255, 153, 255), (255, 153, 255), (102, 178, 255), (102, 178, 255), (102, 178, 255), (102, 178, 255), (255, 51, 51), (255, 51, 51), (255, 51, 51), (255, 51, 51), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0)]
    cmap = matplotlib.cm.get_cmap('hsv')
    plt.figure()

    img = cv2.imread(imgfile) if type(imgfile) == str else imgfile

    color_set = results['colors'] if 'colors' in results else None

    if 'bbox' in results and ids is None and draw_box:
        bboxs = results['bbox']
        for j, rect in enumerate(bboxs):
            xmin, ymin, xmax, ymax = rect
            color = colors[0] if color_set is None else colors[color_set[j] %
                                                               len(colors)]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)

    canvas = img.copy()
    for i in range(kpt_nums):
        for j in range(len(skeletons)):
            if skeletons[j][i, 2] < visual_thresh:
                continue
            if ids is None:
                color = colors[i] if color_set is None else colors[color_set[j]
                                                                   %
                                                                   len(colors)]
            else:
                color = get_color(ids[j])

            cv2.circle(
                canvas,
                tuple(skeletons[j][i, 0:2].astype('int32')),
                2,
                color,
                thickness=-1)

    to_plot = cv2.addWeighted(img, 0.3, canvas, 0.7, 0)
    fig = matplotlib.pyplot.gcf()

    stickwidth = 2

    for i in range(NUM_EDGES):
        for j in range(len(skeletons)):
            edge = EDGES[i]
            if skeletons[j][edge[0], 2] < visual_thresh or skeletons[j][edge[
                    1], 2] < visual_thresh:
                continue

            cur_canvas = canvas.copy()
            X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
            Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1])**2 + (Y[0] - Y[1])**2)**0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)),
                                       (int(length / 2), stickwidth),
                                       int(angle), 0, 360, 1)
            if ids is None:
                color = skeleton_link_colors[i] if color_set is None else colors[color_set[j]
                                                                   %
                                                                   len(colors)]
            else:
                color = get_color(ids[j])
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    if returnimg:
        return canvas
    save_name = os.path.join(
        save_dir, os.path.splitext(os.path.basename(imgfile))[0] + '_vis.jpg')
    plt.imsave(save_name, canvas[:, :, ::-1])
    print("keypoint visualize image saved to: " + save_name)
    plt.close()


def visualize_attr(im, results, boxes=None, is_mtmct=False):
    if isinstance(im, str):
        im = Image.open(im)
        im = np.ascontiguousarray(np.copy(im))
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    else:
        im = np.ascontiguousarray(np.copy(im))

    im_h, im_w = im.shape[:2]
    text_scale = max(0.5, im.shape[0] / 3000.)
    text_thickness = 1

    line_inter = im.shape[0] / 40.
    for i, res in enumerate(results):
        if boxes is None:
            text_w = 3
            text_h = 1
        elif is_mtmct:
            box = boxes[i]  # multi camera, bbox shape is x,y, w,h
            text_w = int(box[0]) + 3
            text_h = int(box[1])
        else:
            box = boxes[i]  # single camera, bbox shape is 0, 0, x,y, w,h
            text_w = int(box[2]) + 3
            text_h = int(box[3])
        for text in res:
            text_h += int(line_inter)
            text_loc = (text_w, text_h)
            cv2.putText(
                im,
                text,
                text_loc,
                cv2.FONT_ITALIC,
                text_scale, (0, 255, 255),
                thickness=text_thickness)
    return im

def translate_to_ori_images(keypoint_result, batch_records):
    kpts = keypoint_result['keypoint']
    scores = keypoint_result['score']
    kpts[..., 0] += batch_records[:, 0:1]
    kpts[..., 1] += batch_records[:, 1:2]
    return kpts, scores

def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--det_model_dir",
        type=str,
        default=None,
        help=("Directory include:'model.pdiparams', 'model.pdmodel', "
              "'infer_cfg.yml', created by tools/export_model.py."),
        required=True)
    parser.add_argument(
        "--keypoint_model_dir",
        type=str,
        default=None,
        help=("Directory include:'model.pdiparams', 'model.pdmodel', "
              "'infer_cfg.yml', created by tools/export_model.py."),
        required=True)
    parser.add_argument(
        "--image_file", type=str, default=None, help="Path of image file.")
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Dir of image file, `image_file` has a higher priority.")
    parser.add_argument(
        "--keypoint_batch_size",
        type=int,
        default=8,
        help=("batch_size for keypoint inference. In detection-keypoint unit"
              "inference, the batch size in detection is 1. Then collate det "
              "result in batch for keypoint inference."))
    parser.add_argument(
        "--video_file",
        type=str,
        default=None,
        help="Path of video file, `video_file` or `camera_id` has a highest priority."
    )
    parser.add_argument(
        "--camera_id",
        type=int,
        default=-1,
        help="device id of camera to predict.")
    parser.add_argument(
        "--det_threshold", type=float, default=0.5, help="Threshold of score.")
    parser.add_argument(
        "--keypoint_threshold",
        type=float,
        default=0.5,
        help="Threshold of score.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory of output visualization files.")
    parser.add_argument(
        "--run_mode",
        type=str,
        default='paddle',
        help="mode of running(paddle/trt_fp32/trt_fp16/trt_int8)")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU."
    )
    parser.add_argument(
        "--run_benchmark",
        type=ast.literal_eval,
        default=False,
        help="Whether to predict a image_file repeatedly for benchmark")
    parser.add_argument(
        "--enable_mkldnn",
        type=ast.literal_eval,
        default=False,
        help="Whether use mkldnn with CPU.")
    parser.add_argument("--reference_image_path", type=str, default="reference_image.jpg")
    parser.add_argument(
        "--cpu_threads", type=int, default=1, help="Num of threads with CPU.")
    parser.add_argument(
        "--trt_min_shape", type=int, default=1, help="min_shape for TensorRT.")
    parser.add_argument(
        "--trt_max_shape",
        type=int,
        default=1280,
        help="max_shape for TensorRT.")
    parser.add_argument(
        "--trt_opt_shape",
        type=int,
        default=640,
        help="opt_shape for TensorRT.")
    parser.add_argument(
        "--trt_calib_mode",
        type=bool,
        default=False,
        help="If the model is produced by TRT offline quantitative "
        "calibration, trt_calib_mode need to set True.")
    parser.add_argument(
        '--use_dark',
        type=ast.literal_eval,
        default=True,
        help='whether to use darkpose to get better keypoint position predict ')
    parser.add_argument(
        '--save_res',
        type=bool,
        default=False,
        help=(
            "whether to save predict results to json file"
            "1) store_res: a list of image_data"
            "2) image_data: [imageid, rects, [keypoints, scores]]"
            "3) rects: list of rect [xmin, ymin, xmax, ymax]"
            "4) keypoints: 17(joint numbers)*[x, y, conf], total 51 data in list"
            "5) scores: mean of all joint conf"))
    parser.add_argument(
        '--smooth',
        type=ast.literal_eval,
        default=False,
        help='smoothing keypoints for each frame, new incoming keypoints will be more stable.'
    )
    parser.add_argument(
        '--filter_type',
        type=str,
        default='OneEuro',
        help='when set --smooth True, choose filter type you want to use, it can be [OneEuro] or [EMA].'
    )
    return parser


class PredictConfig_KeyPoint():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir, use_fd_format=False):
        # parsing Yaml config for Preprocess
        fd_deploy_file = os.path.join(model_dir, 'inference.yml')
        ppdet_deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        if use_fd_format:
            if not os.path.exists(fd_deploy_file) and os.path.exists(
                    ppdet_deploy_file):
                raise RuntimeError(
                    "Non-FD format model detected. Please set `use_fd_format` to False."
                )
            deploy_file = fd_deploy_file
        else:
            if not os.path.exists(ppdet_deploy_file) and os.path.exists(
                    fd_deploy_file):
                raise RuntimeError(
                    "FD format model detected. Please set `use_fd_format` to False."
                )
            deploy_file = ppdet_deploy_file
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        self.archcls = KEYPOINT_SUPPORT_MODELS[yml_conf['arch']]
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.tagmap = False
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        if 'keypoint_bottomup' == self.archcls:
            self.tagmap = True
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type 
        """
        for support_model in KEYPOINT_SUPPORT_MODELS:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf[
            'arch'], KEYPOINT_SUPPORT_MODELS))

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


def expand_crop(images, rect, expand_ratio=0.3):
    imgh, imgw, c = images.shape
    label, conf, xmin, ymin, xmax, ymax = [int(x) for x in rect.tolist()]
    if label != 0:
        return None, None, None
    org_rect = [xmin, ymin, xmax, ymax]
    h_half = (ymax - ymin) * (1 + expand_ratio) / 2.
    w_half = (xmax - xmin) * (1 + expand_ratio) / 2.
    if h_half > w_half * 4 / 3:
        w_half = h_half * 0.75
    center = [(ymin + ymax) / 2., (xmin + xmax) / 2.]
    ymin = max(0, int(center[0] - h_half))
    ymax = min(imgh - 1, int(center[0] + h_half))
    xmin = max(0, int(center[1] - w_half))
    xmax = min(imgw - 1, int(center[1] + w_half))
    return images[ymin:ymax, xmin:xmax, :], [xmin, ymin, xmax, ymax], org_rect


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale * 200, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

class HRNetPostProcess(object):
    def __init__(self, use_dark=True):
        self.use_dark = use_dark

    def flip_back(self, output_flipped, matched_parts):
        assert output_flipped.ndim == 4,\
                'output_flipped should be [batch_size, num_joints, height, width]'

        output_flipped = output_flipped[:, :, :, ::-1]

        for pair in matched_parts:
            tmp = output_flipped[:, pair[0], :, :].copy()
            output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
            output_flipped[:, pair[1], :, :] = tmp

        return output_flipped

    def get_max_preds(self, heatmaps):
        """get predictions from score maps

        Args:
            heatmaps: numpy.ndarray([batch_size, num_joints, height, width])

        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 2]), keypoints coords
            maxvals: numpy.ndarray([batch_size, num_joints, 2]), the maximum confidence of the keypoints
        """
        assert isinstance(heatmaps,
                          np.ndarray), 'heatmaps should be numpy.ndarray'
        assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[1]
        width = heatmaps.shape[3]
        heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask

        return preds, maxvals

    def gaussian_blur(self, heatmap, kernel):
        border = (kernel - 1) // 2
        batch_size = heatmap.shape[0]
        num_joints = heatmap.shape[1]
        height = heatmap.shape[2]
        width = heatmap.shape[3]
        for i in range(batch_size):
            for j in range(num_joints):
                origin_max = np.max(heatmap[i, j])
                dr = np.zeros((height + 2 * border, width + 2 * border))
                dr[border:-border, border:-border] = heatmap[i, j].copy()
                dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
                heatmap[i, j] = dr[border:-border, border:-border].copy()
                heatmap[i, j] *= origin_max / np.max(heatmap[i, j])
        return heatmap

    def dark_parse(self, hm, coord):
        heatmap_height = hm.shape[0]
        heatmap_width = hm.shape[1]
        px = int(coord[0])
        py = int(coord[1])
        if 1 < px < heatmap_width - 2 and 1 < py < heatmap_height - 2:
            dx = 0.5 * (hm[py][px + 1] - hm[py][px - 1])
            dy = 0.5 * (hm[py + 1][px] - hm[py - 1][px])
            dxx = 0.25 * (hm[py][px + 2] - 2 * hm[py][px] + hm[py][px - 2])
            dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] \
                + hm[py-1][px-1])
            dyy = 0.25 * (
                hm[py + 2 * 1][px] - 2 * hm[py][px] + hm[py - 2 * 1][px])
            derivative = np.matrix([[dx], [dy]])
            hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
            if dxx * dyy - dxy**2 != 0:
                hessianinv = hessian.I
                offset = -hessianinv * derivative
                offset = np.squeeze(np.array(offset.T), axis=0)
                coord += offset
        return coord

    def dark_postprocess(self, hm, coords, kernelsize):
        """
        refer to https://github.com/ilovepose/DarkPose/lib/core/inference.py

        """
        hm = self.gaussian_blur(hm, kernelsize)
        hm = np.maximum(hm, 1e-10)
        hm = np.log(hm)
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                coords[n, p] = self.dark_parse(hm[n][p], coords[n][p])
        return coords

    def get_final_preds(self, heatmaps, center, scale, kernelsize=3):
        """the highest heatvalue location with a quarter offset in the
        direction from the highest response to the second highest response.

        Args:
            heatmaps (numpy.ndarray): The predicted heatmaps
            center (numpy.ndarray): The boxes center
            scale (numpy.ndarray): The scale factor

        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 2]), keypoints coords
            maxvals: numpy.ndarray([batch_size, num_joints, 1]), the maximum confidence of the keypoints
        """

        coords, maxvals = self.get_max_preds(heatmaps)

        heatmap_height = heatmaps.shape[2]
        heatmap_width = heatmaps.shape[3]
        if self.use_dark:
            coords = self.dark_postprocess(heatmaps, coords, kernelsize)
        else:
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    hm = heatmaps[n][p]
                    px = int(math.floor(coords[n][p][0] + 0.5))
                    py = int(math.floor(coords[n][p][1] + 0.5))
                    if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                        diff = np.array([
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ])
                        coords[n][p] += np.sign(diff) * .25
        preds = coords.copy()

        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = transform_preds(coords[i], center[i], scale[i],
                                       [heatmap_width, heatmap_height])

        return preds, maxvals

    def __call__(self, output, center, scale):
        preds, maxvals = self.get_final_preds(output, center, scale)
        return np.concatenate(
            (preds, maxvals), axis=-1), np.mean(
                maxvals, axis=1)




class KeyPointDetector(Detector):
    """
    Args:
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU/NPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of pre batch in inference
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
        use_dark(bool): whether to use postprocess in DarkPose
    """

    def __init__(self,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 output_dir='output',
                 threshold=0.5,
                 use_dark=True,
                 use_fd_format=False):
        super(KeyPointDetector, self).__init__(
            model_dir=model_dir,
            device=device,
            run_mode=run_mode,
            batch_size=batch_size,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            output_dir=output_dir,
            threshold=threshold,
            use_fd_format=use_fd_format)
        self.use_dark = use_dark

    def set_config(self, model_dir, use_fd_format):
        return PredictConfig_KeyPoint(model_dir, use_fd_format=use_fd_format)

    def get_person_from_rect(self, image, results):
        # crop the person result from image
        self.det_times.preprocess_time_s.start()
        valid_rects = results['boxes']
        rect_images = []
        new_rects = []
        org_rects = []
        for rect in valid_rects:
            rect_image, new_rect, org_rect = expand_crop(image, rect)
            if rect_image is None or rect_image.size == 0:
                continue
            rect_images.append(rect_image)
            new_rects.append(new_rect)
            org_rects.append(org_rect)
        self.det_times.preprocess_time_s.end()
        return rect_images, new_rects, org_rects

    def postprocess(self, inputs, result):
        np_heatmap = result['heatmap']
        np_masks = result['masks']
        # postprocess output of predictor
        if KEYPOINT_SUPPORT_MODELS[
                self.pred_config.arch] == 'keypoint_bottomup':
            results = {}
            h, w = inputs['im_shape'][0]
            preds = [np_heatmap]
            if np_masks is not None:
                preds += np_masks
            preds += [h, w]
            keypoint_postprocess = HrHRNetPostProcess()
            kpts, scores = keypoint_postprocess(*preds)
            results['keypoint'] = kpts
            results['score'] = scores
            return results
        elif KEYPOINT_SUPPORT_MODELS[
                self.pred_config.arch] == 'keypoint_topdown':
            results = {}
            imshape = inputs['im_shape'][:, ::-1]
            center = np.round(imshape / 2.)
            scale = imshape / 200.
            keypoint_postprocess = HRNetPostProcess(use_dark=self.use_dark)
            kpts, scores = keypoint_postprocess(np_heatmap, center, scale)
            results['keypoint'] = kpts
            results['score'] = scores
            return results
        elif KEYPOINT_SUPPORT_MODELS[
                self.pred_config.arch] == 'keypoint_topdown_wholebody':
            results = {}
            imshape = inputs['im_shape'][:, ::-1]
            center = []
            scale = []
            for i in range(len(inputs['im_shape'])):
                transize = np.shape(inputs["image"])
                tmp_center, tmp_scale = _box2cs([np.shape(inputs["image"])[-1],np.shape(inputs["image"])[-2]], [0,0,inputs['im_shape'][i][1],inputs['im_shape'][i][0]] )
                center.append(tmp_center)
                scale.append(tmp_scale)

            keypoint_postprocess = HRNetPostProcess(use_dark=self.use_dark)
            kpts, scores = keypoint_postprocess(np_heatmap, center, scale)
            results['keypoint'] = kpts
            results['score'] = scores
            return results
        else:
            raise ValueError("Unsupported arch: {}, expect {}".format(
                self.pred_config.arch, KEYPOINT_SUPPORT_MODELS))

    def predict(self, repeats=1):
        '''
        Args:
            repeats (int): repeat number for prediction
        Returns:
            results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matrix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's results include 'masks': np.ndarray:
                            shape: [N, im_h, im_w]
        '''
        # model prediction
        np_heatmap, np_masks = None, None
        for i in range(repeats):
            self.predictor.run()
            # self.predictor.run()
            output_names = self.predictor.get_output_names()
            heatmap_tensor = self.predictor.get_output_handle(output_names[0])
            np_heatmap = heatmap_tensor.copy_to_cpu()
            if self.pred_config.tagmap:
                masks_tensor = self.predictor.get_output_handle(output_names[1])
                heat_k = self.predictor.get_output_handle(output_names[2])
                inds_k = self.predictor.get_output_handle(output_names[3])
                np_masks = [
                    masks_tensor.copy_to_cpu(), heat_k.copy_to_cpu(),
                    inds_k.copy_to_cpu()
                ]
        result = dict(heatmap=np_heatmap, masks=np_masks)
        return result

    def predict_image(self,
                      image_list,
                      run_benchmark=False,
                      repeats=1,
                      visual=True):
        results = []
        batch_loop_cnt = math.ceil(float(len(image_list)) / self.batch_size)
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(image_list))
            batch_image_list = image_list[start_index:end_index]
            if run_benchmark:
                # preprocess
                inputs = self.preprocess(batch_image_list)  # warmup
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                result_warmup = self.predict(repeats=repeats)  # warmup
                self.det_times.inference_time_s.start()
                result = self.predict(repeats=repeats)
                self.det_times.inference_time_s.end(repeats=repeats)

                # postprocess
                result_warmup = self.postprocess(inputs, result)  # warmup
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += len(batch_image_list)

                cm, gm, gu = get_current_memory_mb()
                self.cpu_mem += cm
                self.gpu_mem += gm
                self.gpu_util += gu

            else:
                # preprocess
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                self.det_times.inference_time_s.start()
                result = self.predict()
                self.det_times.inference_time_s.end()

                # postprocess
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += len(batch_image_list)

                if visual:
                    if not os.path.exists(self.output_dir):
                        os.makedirs(self.output_dir)
                    visualize(
                        batch_image_list,
                        result,
                        visual_thresh=self.threshold,
                        save_dir=self.output_dir)

            results.append(result)
            if visual:
                print('Test iter {}'.format(i))
        results = self.merge_batch_result(results)
        return results

    def predict_video(self, video_file, camera_id):
        video_name = 'output.mp4'
        if camera_id != -1:
            capture = cv2.VideoCapture(camera_id)
        else:
            capture = cv2.VideoCapture(video_file)
            video_name = os.path.split(video_file)[-1]
        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("fps: %d, frame_count: %d" % (fps, frame_count))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        out_path = os.path.join(self.output_dir, video_name)
        fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        index = 1
        while (1):
            ret, frame = capture.read()
            if not ret:
                break
            print('detect frame: %d' % (index))
            index += 1
            results = self.predict_image([frame[:, :, ::-1]], visual=False)
            im_results = {}
            im_results['keypoint'] = [results['keypoint'], results['score']]
            im = visualize_pose(
                frame, im_results, visual_thresh=self.threshold, returnimg=True)
            writer.write(im)
            if camera_id != -1:
                cv2.imshow('Mask Detection', im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        writer.release()


def predict_with_given_det(image, det_res, keypoint_detector,
                           keypoint_batch_size, run_benchmark):
    keypoint_res = {}

    rec_images, records, det_rects = keypoint_detector.get_person_from_rect(
        image, det_res)

    if len(det_rects) == 0:
        keypoint_res['keypoint'] = [[], []]
        return keypoint_res

    keypoint_vector = []
    score_vector = []

    rect_vector = det_rects
    keypoint_results = keypoint_detector.predict_image(
        rec_images, run_benchmark, repeats=10, visual=False)
    keypoint_vector, score_vector = translate_to_ori_images(keypoint_results,
                                                            np.array(records))
    keypoint_res['keypoint'] = [
        keypoint_vector.tolist(), score_vector.tolist()
    ] if len(keypoint_vector) > 0 else [[], []]
    keypoint_res['bbox'] = rect_vector
    return keypoint_res


def topdown_unite_predict(detector,
                          topdown_keypoint_detector,
                          image_list,
                          keypoint_batch_size=1,
                          save_res=False):
    det_timer = detector.get_timer()
    store_res = []
    for i, img_file in enumerate(image_list):
        # Decode image in advance in det + pose prediction
        det_timer.preprocess_time_s.start()
        image, _ = decode_image(img_file, {})
        det_timer.preprocess_time_s.end()

        if FLAGS.run_benchmark:
            results = detector.predict_image(
                [image], run_benchmark=True, repeats=10)

            cm, gm, gu = get_current_memory_mb()
            detector.cpu_mem += cm
            detector.gpu_mem += gm
            detector.gpu_util += gu
        else:
            results = detector.predict_image([image], visual=False)
        results = detector.filter_box(results, FLAGS.det_threshold)
        if results['boxes_num'] > 0:
            keypoint_res = predict_with_given_det(
                image, results, topdown_keypoint_detector, keypoint_batch_size,
                FLAGS.run_benchmark)

            if save_res:
                save_name = img_file if isinstance(img_file, str) else i
                store_res.append([
                    save_name, keypoint_res['bbox'],
                    [keypoint_res['keypoint'][0], keypoint_res['keypoint'][1]]
                ])
        else:
            results["keypoint"] = [[], []]
            keypoint_res = results
        if FLAGS.run_benchmark:
            cm, gm, gu = get_current_memory_mb()
            topdown_keypoint_detector.cpu_mem += cm
            topdown_keypoint_detector.gpu_mem += gm
            topdown_keypoint_detector.gpu_util += gu
        else:
            if not os.path.exists(FLAGS.output_dir):
                os.makedirs(FLAGS.output_dir)
            visualize_pose(
                img_file,
                keypoint_res,
                visual_thresh=FLAGS.keypoint_threshold,
                save_dir=FLAGS.output_dir)
    if save_res:
        """
        1) store_res: a list of image_data
        2) image_data: [imageid, rects, [keypoints, scores]]
        3) rects: list of rect [xmin, ymin, xmax, ymax]
        4) keypoints: 17(joint numbers)*[x, y, conf], total 51 data in list
        5) scores: mean of all joint conf
        """
        with open("det_keypoint_unite_image_results.json", 'w') as wf:
            json.dump(store_res, wf, indent=4)

from paddlex import create_model
def topdown_unite_predict_video(topdown_keypoint_detector,
                                camera_id,
                                keypoint_batch_size=1,
                                save_res=False):
    video_name = 'output.mp4'
    
    capture = cv2.VideoCapture(FLAGS.video_file)
    vr = VideoReader(FLAGS.video_file, ctx=cpu(0))

   
    first_frame = vr[0].asnumpy()
    first_frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(FLAGS.reference_image_path, first_frame_bgr)
    
    # Get Video info : resolution, fps, frame count
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("fps: %d, frame_count: %d" % (fps, frame_count))

    out_path =FLAGS.output_dir
    fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    index = 0
    store_res = []
    keypoint_smoothing = KeypointSmoothing(
        width, height, filter_type=FLAGS.filter_type, beta=0.05)
    model = create_model(FLAGS.det_model_dir)
    pose_map=[]
    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        index += 1
        print('detect frame: %d' % (index))

        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(frame2)
        bboxes = {'boxes':[],'boxes_num':[]}
        for res in results:
            for i in res["boxes"]:
                if i['score'] > FLAGS.det_threshold:
                    temp = [0,i['score'],i['coordinate'][0],i['coordinate'][1],i['coordinate'][2],i['coordinate'][3]]
                    bboxes['boxes'].append(temp)
        bboxes['boxes_num'] = [len(bboxes['boxes'])]
        bboxes['boxes_num'] = np.array(bboxes['boxes_num'])
        bboxes['boxes'] = np.array(bboxes['boxes'])
        if bboxes['boxes_num'] == 0:
            writer.write(frame)
            continue
        keypoint_res = predict_with_given_det(
            frame2, bboxes, topdown_keypoint_detector, keypoint_batch_size,
            FLAGS.run_benchmark)

        if FLAGS.smooth and len(keypoint_res['keypoint'][0]) == 1:
            current_keypoints = np.array(keypoint_res['keypoint'][0][0])
            smooth_keypoints = keypoint_smoothing.smooth_process(
                current_keypoints)

            keypoint_res['keypoint'][0][0] = smooth_keypoints.tolist()
            
        zero = np.zeros((height, width, 3), dtype=np.uint8)
        im = visualize_pose(
            zero,
            keypoint_res,
            visual_thresh=FLAGS.keypoint_threshold,
            returnimg=True)
        pose_map.append(im)

        if save_res:
            store_res.append([
                index, keypoint_res['bbox'],
                [keypoint_res['keypoint'][0], keypoint_res['keypoint'][1]]
            ])

        writer.write(im)
        if camera_id != -1:
            cv2.imshow('Mask Detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    writer.release()
    cv2.imwrite(FLAGS.reference_image_path.replace("reference","control"), pose_map[0])
    print('output_video saved to: {}'.format(out_path))
    if save_res:
        """
        1) store_res: a list of frame_data
        2) frame_data: [frameid, rects, [keypoints, scores]]
        3) rects: list of rect [xmin, ymin, xmax, ymax]
        4) keypoints: 17(joint numbers)*[x, y, conf], total 51 data in list
        5) scores: mean of all joint conf
        """
        with open("det_keypoint_unite_video_results.json", 'w') as wf:
            json.dump(store_res, wf, indent=4)


class KeypointSmoothing(object):
    # The following code are modified from:
    # https://github.com/jaantollander/OneEuroFilter

    def __init__(self,
                 width,
                 height,
                 filter_type,
                 alpha=0.5,
                 fc_d=0.1,
                 fc_min=0.1,
                 beta=0.1,
                 thres_mult=0.3):
        super(KeypointSmoothing, self).__init__()
        self.image_width = width
        self.image_height = height
        self.threshold = np.array([
            0.005, 0.005, 0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
        ]) * thres_mult
        self.filter_type = filter_type
        self.alpha = alpha
        self.dx_prev_hat = None
        self.x_prev_hat = None
        self.fc_d = fc_d
        self.fc_min = fc_min
        self.beta = beta

        if self.filter_type == 'OneEuro':
            self.smooth_func = self.one_euro_filter
        elif self.filter_type == 'EMA':
            self.smooth_func = self.ema_filter
        else:
            raise ValueError('filter type must be one_euro or ema')

    def smooth_process(self, current_keypoints):
        if self.x_prev_hat is None:
            self.x_prev_hat = current_keypoints[:, :2]
            self.dx_prev_hat = np.zeros(current_keypoints[:, :2].shape)
            return current_keypoints
        else:
            result = current_keypoints
            num_keypoints = len(current_keypoints)
            for i in range(num_keypoints):
                result[i, :2] = self.smooth(current_keypoints[i, :2],
                                            self.threshold[i], i)
            return result

    def smooth(self, current_keypoint, threshold, index):
        distance = np.sqrt(
            np.square((current_keypoint[0] - self.x_prev_hat[index][0]) /
                      self.image_width) + np.square((current_keypoint[
                          1] - self.x_prev_hat[index][1]) / self.image_height))
        if distance < threshold:
            result = self.x_prev_hat[index]
        else:
            result = self.smooth_func(current_keypoint, self.x_prev_hat[index],
                                      index)

        return result

    def one_euro_filter(self, x_cur, x_pre, index):
        te = 1
        self.alpha = self.smoothing_factor(te, self.fc_d)
        dx_cur = (x_cur - x_pre) / te
        dx_cur_hat = self.exponential_smoothing(dx_cur, self.dx_prev_hat[index])

        fc = self.fc_min + self.beta * np.abs(dx_cur_hat)
        self.alpha = self.smoothing_factor(te, fc)
        x_cur_hat = self.exponential_smoothing(x_cur, x_pre)
        self.dx_prev_hat[index] = dx_cur_hat
        self.x_prev_hat[index] = x_cur_hat
        return x_cur_hat

    def ema_filter(self, x_cur, x_pre, index):
        x_cur_hat = self.exponential_smoothing(x_cur, x_pre)
        self.x_prev_hat[index] = x_cur_hat
        return x_cur_hat

    def smoothing_factor(self, te, fc):
        r = 2 * math.pi * fc * te
        return r / (r + 1)

    def exponential_smoothing(self, x_cur, x_pre, index=0):
        return self.alpha * x_cur + (1 - self.alpha) * x_pre


def main():
    deploy_file = os.path.join(FLAGS.det_model_dir, 'infer_cfg.yml')
    with open(deploy_file) as f:
        yml_conf = yaml.safe_load(f)
    arch = yml_conf['arch']
    detector_func = 'Detector'
    if arch == 'PicoDet':
        detector_func = 'DetectorPicoDet'

    topdown_keypoint_detector = KeyPointDetector(
        FLAGS.keypoint_model_dir,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        batch_size=FLAGS.keypoint_batch_size,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn,
        use_dark=FLAGS.use_dark)
    keypoint_arch = topdown_keypoint_detector.pred_config.arch
    assert KEYPOINT_SUPPORT_MODELS[keypoint_arch] == 'keypoint_topdown' or KEYPOINT_SUPPORT_MODELS[keypoint_arch] == 'keypoint_topdown_wholebody', 'Detection-Keypoint unite inference only supports topdown models.'

    # predict from video file or camera video stream
    if FLAGS.video_file is not None or FLAGS.camera_id != -1:
        topdown_unite_predict_video(topdown_keypoint_detector,
                                    FLAGS.camera_id, FLAGS.keypoint_batch_size,
                                    FLAGS.save_res)
    else:
        # predict from image
        img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
        topdown_unite_predict(topdown_keypoint_detector, img_list,
                              FLAGS.keypoint_batch_size, FLAGS.save_res)
        if not FLAGS.run_benchmark:
            detector.det_times.info(average=True)
            topdown_keypoint_detector.det_times.info(average=True)
        else:
            mode = FLAGS.run_mode
            det_model_dir = FLAGS.det_model_dir
            det_model_info = {
                'model_name': det_model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(detector, img_list, det_model_info, name='Det')
            keypoint_model_dir = FLAGS.keypoint_model_dir
            keypoint_model_info = {
                'model_name': keypoint_model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(topdown_keypoint_detector, img_list, keypoint_model_info,
                      FLAGS.keypoint_batch_size, 'KeyPoint')


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"

    main()

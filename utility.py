import time
import numpy as np
import cv2
import torch
import numpy as np
import numpy.ma as ma
import math
import operator

from pyk4a import ImageFormat
from typing import Optional, Tuple

import torch.nn as nn
import onnxruntime

class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count


class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider


def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def transform_2d_to_3d(depth, x, y, camera_param):
    image_h, image_w = depth.shape[:2]

    depth_roi = depth[max(0,y-20):min(image_h-1, y+20), max(0,x-20):min(image_w-1, x+20)]

    try:
        min_depth = np.min(ma.masked_where(depth_roi == 0, depth_roi))
        if min_depth >= 1:
            pass
        else:
            min_depth = 100000 #0

    except ValueError:
        min_depth = 0

    # min_depth = np.min(ma.masked_where(depth_roi == 0, depth_roi))
    # if min_depth >= 1:
    #     pass
    # else:onnx
    #     min_depth = 0

    #min_depth = np.median(depth_roi)
    #min_depth = np.min(depth_roi)

    fx = camera_param['fx']
    fy = camera_param['fy']
    cx = camera_param['cx']
    cy = camera_param['cy']

    pt_3d = (min_depth / fx * (x - cx),
             min_depth / fy * (y - cy),
             float(min_depth))

    return pt_3d

def cal_dist(pt1, pt2):

    dist_3d = math.sqrt((pt1[0]-pt2[0]) * (pt1[0]-pt2[0]) + (pt1[1]-pt2[1]) * (pt1[1]-pt2[1]) + (pt1[2]-pt2[2]) * (pt1[2]-pt2[2]))

    return dist_3d


def convert_to_bgra_if_required(color_format: ImageFormat, color_image):
    # examples for all possible pyk4a.ColorFormats
    if color_format == ImageFormat.COLOR_MJPG:
        color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
    elif color_format == ImageFormat.COLOR_NV12:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
        # this also works and it explains how the COLOR_NV12 color color_format is stored in memory
        # h, w = color_image.shape[0:2]
        # h = h // 3 * 2
        # luminance = color_image[:h]
        # chroma = color_image[h:, :w//2]
        # color_image = cv2.cvtColorTwoPlane(luminance, chroma, cv2.COLOR_YUV2BGRA_NV12)
    elif color_format == ImageFormat.COLOR_YUY2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
    return color_image


def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img


## SimSiam loss
def loss_negative_cosine(p, z, label):
    # stop gradient
    z = z.detach()

    batch_size = p.shape[0]

    p_view = p.view(batch_size, -1)
    z_view = z.reshape(batch_size, -1)

    p_norm = torch.norm(p_view, dim=1)
    p_norm = torch.unsqueeze(p_norm, 1)
    z_norm = torch.norm(z_view, dim=1)
    z_norm = torch.unsqueeze(z_norm, 1)

    p_norm = p_view / p_norm
    z_norm = z_view / z_norm

    loss = -torch.mean(torch.sum(p_norm * z_norm, dim=1))

    return loss


## BYOL loss
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


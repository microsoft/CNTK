# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from utils.cython_modules.cpu_nms import cpu_nms
try:
    from utils.cython_modules.gpu_nms import gpu_nms
    gpu_nms_available = True
except ImportError:
    gpu_nms_available = False

try:
    from config import cfg
except ImportError:
    from utils.default_config import cfg

def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if gpu_nms_available and cfg.USE_GPU_NMS and not force_cpu:
        return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    else:
        return cpu_nms(dets, thresh)

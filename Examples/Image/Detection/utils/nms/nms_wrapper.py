# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
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

import pdb

def nms(dets, thresh, force_cpu=False):
    '''
    Dispatches the call to either CPU or GPU NMS implementations
    '''
    if dets.shape[0] == 0:
        return []
    if gpu_nms_available and cfg.USE_GPU_NMS and not force_cpu:
        return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    else:
        return cpu_nms(dets, thresh)

def apply_nms_to_single_image_results(coords, labels, scores, nms_threshold=0.5, conf_threshold=0.0):
    '''
    Applies nms to the results for a single image.

    Args:
        coords:             (x_min, y_min, x_max, y_max) coordinates for n rois. shape = (n, 4)
        labels:             the predicted label per roi. shape = (n, 1)
        scores:             the predicted score per roi. shape = (n, 1)
        nms_threshold:      the threshold for discarding overlapping ROIs in nms
        conf_threshold:     a minimum value for the score of an ROI. ROIs with lower score will be discarded

    Returns:
        nmsKeepIndices - the indices of the ROIs to keep after nms
    '''

    # generate input for nms
    allIndices = []
    nmsRects = [[[]] for _ in range(max(labels) + 1)]
    coordsWithScores = np.hstack((coords, np.array([scores]).T))
    for i in range(max(labels) + 1):
        indices = np.where(np.array(labels) == i)[0]
        nmsRects[i][0] = coordsWithScores[indices,:]
        allIndices.append(indices)

    # call nms
    _, nmsKeepIndicesList = apply_nms_to_test_set_results(nmsRects, nms_threshold, conf_threshold)

    # map back to original roi indices
    nmsKeepIndices = []
    for i in range(max(labels) + 1):
        for keepIndex in nmsKeepIndicesList[i][0]:
            nmsKeepIndices.append(allIndices[i][keepIndex]) # for keepIndex in nmsKeepIndicesList[i][0]]
    assert (len(nmsKeepIndices) == len(set(nmsKeepIndices))) # check if no roi indices was added >1 times
    return nmsKeepIndices

def apply_nms_to_test_set_results(all_boxes, nms_threshold, conf_threshold):
    '''
    Applies nms to the results of multiple images.

    Args:
        all_boxes:      shape of all_boxes: e.g. 21 classes x 4952 images x 58 rois x 5 coords+score
        nms_threshold:  the threshold for discarding overlapping ROIs in nms
        conf_threshold: a minimum value for the score of an ROI. ROIs with lower score will be discarded

    Returns:
        nms_boxes - the reduced set of rois after nms
        nmsKeepIndices - the indices of the ROIs to keep after nms
    '''

    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    nms_keepIndices = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    for cls_ind in range(num_classes):
        for im_ind in range(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            keep = nms(dets.astype(np.float32), nms_threshold)

            # also filter out low confidences
            if conf_threshold > 0:
                #pdb.set_trace()
                keep_conf_idx = np.where(dets[:, -1] > conf_threshold)
                keep = list(set(keep_conf_idx[0]).intersection(keep))

            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
            nms_keepIndices[cls_ind][im_ind] = keep
    return nms_boxes, nms_keepIndices


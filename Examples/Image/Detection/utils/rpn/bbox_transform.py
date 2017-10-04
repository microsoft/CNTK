# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np

# compute example and gt width ctr, width and height
# and returns optimal target deltas
def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

# gets
# - boxes (n, 4) as [x_low, y_low, x_high, y_high]
# - deltas (n, 4) as [dx, dy, dw, dh]
# returns
# - pred_boxes (n, 4) as [x_low, y_low, x_high, y_high]
# where
# pred_ctr_x = dx * widths + ctr_x
# --> pred_x_low = pred_ctr_x - 0.5 * pred_w
# and
# pred_w = np.exp(dw) * widths
def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_info):
    '''
    Clip boxes to image boundaries.
    :param boxes: boxes
    :param im_info: (pad_width, pad_height, scaled_image_width, scaled_image_height, orig_img_width, orig_img_height)
                    e.g.(1000, 1000, 1000, 600, 500, 300) for an original image of 600x300 that is scaled and padded to 1000x1000
    '''

    im_info.shape = (6)
    padded_wh = im_info[0:2]
    scaled_wh = im_info[2:4]
    xy_offset = (padded_wh - scaled_wh) / 2
    xy_min = xy_offset
    xy_max = xy_offset + scaled_wh

    # x_min <= x1 <= x_max
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], xy_max[0] - 1), xy_min[0])
    # y_min <= y1 <= y_max
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], xy_max[1] - 1), xy_min[1])
    # x_min <= x2 <= x_max
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], xy_max[0] - 1), xy_min[0])
    # y_min <= y2 <= y_max
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], xy_max[1] - 1), xy_min[1])
    return boxes

def regress_rois(roi_proposals, roi_regression_factors, labels, dims_input):
    for i in range(len(labels)):
        label = labels[i]
        if label > 0:
            deltas = roi_regression_factors[i:i+1,label*4:(label+1)*4]
            roi_coords = roi_proposals[i:i+1,:]
            regressed_rois = bbox_transform_inv(roi_coords, deltas)
            roi_proposals[i,:] = regressed_rois

    if dims_input is not None:
        # dims_input -- (pad_width, pad_height, scaled_image_width, scaled_image_height, orig_img_width, orig_img_height)
        pad_width, pad_height, scaled_image_width, scaled_image_height, _, _ = dims_input
        left = (pad_width - scaled_image_width) / 2
        right = pad_width - left - 1
        top = (pad_height - scaled_image_height) / 2
        bottom = pad_height - top - 1

        roi_proposals[:,0] = roi_proposals[:,0].clip(left, right)
        roi_proposals[:,1] = roi_proposals[:,1].clip(top, bottom)
        roi_proposals[:,2] = roi_proposals[:,2].clip(left, right)
        roi_proposals[:,3] = roi_proposals[:,3].clip(top, bottom)

    return roi_proposals

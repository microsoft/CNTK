# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
from utils.rpn.bbox_transform import bbox_transform_inv

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


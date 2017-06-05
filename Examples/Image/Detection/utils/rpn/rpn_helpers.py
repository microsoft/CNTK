# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import cntk as C
from cntk import user_function, relu, softmax, reduce_sum, slice, splice, reshape, element_times, plus, alias
from cntk.initializer import glorot_uniform, normal
from cntk.layers import Convolution
from cntk.losses import cross_entropy_with_softmax
from utils.rpn.anchor_target_layer import AnchorTargetLayer
from utils.rpn.cntk_ignore_label import IgnoreLabel
from utils.rpn.proposal_layer import ProposalLayer
from utils.rpn.proposal_target_layer import ProposalTargetLayer
from utils.rpn.cntk_smoothL1_loss import SmoothL1Loss

def create_rpn(conv_out, scaled_gt_boxes, im_info, add_loss_functions=True,
               proposal_layer_param_string=None):
    '''
    Creates a region proposal network for object detection as proposed in the "Faster R-CNN" paper:
        Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun:
        "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"

    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").

    Args:
        conv_out:        The convolutional feature map, i.e. the output of the conv layers from the pretrained classification network
        scaled_gt_boxes: The ground truth boxes as (x1, y1, x2, y2, label). Coordinates are absolute pixels wrt. the input image.
        im_info:         (image_widht, image_height, image_scale) as CNTK variable or constant
        add_loss_functions: If set to True rpn_losses will be returned, otherwise None is returned for the losses
        proposal_layer_param_string: A yaml parameter string that is passed to the proposal layer.

    Returns:
        rpn_rois - the proposed ROIs
        rpn_losses - the losses (SmoothL1 loss for bbox regression plus cross entropy for objectness)
    '''

    # RPN network
    # init = 'normal', initValueScale = 0.01, initBias = 0.1
    rpn_conv_3x3 = Convolution((3, 3), 256, activation=relu, pad=True, strides=1,
                                init = normal(scale=0.01), init_bias=0.1)(conv_out)
    rpn_cls_score = Convolution((1, 1), 18, activation=None, name="rpn_cls_score",
                                init = normal(scale=0.01), init_bias=0.1)(rpn_conv_3x3)  # 2(bg/fg)  * 9(anchors)
    rpn_bbox_pred = Convolution((1, 1), 36, activation=None, name="rpn_bbox_pred",
                                init = normal(scale=0.01), init_bias=0.1)(rpn_conv_3x3)  # 4(coords) * 9(anchors)

    # apply softmax to get (bg, fg) probabilities and reshape predictions back to grid of (18, H, W)
    num_predictions = int(np.prod(rpn_cls_score.shape) / 2)
    rpn_cls_score_rshp = reshape(rpn_cls_score, (2, num_predictions))
    rpn_cls_prob = softmax(rpn_cls_score_rshp, axis=0, name="objness_softmax")
    rpn_cls_prob_reshape = reshape(rpn_cls_prob, rpn_cls_score.shape)

    # proposal layer
    rpn_rois_raw = user_function(ProposalLayer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, param_str=proposal_layer_param_string))
    rpn_rois = alias(rpn_rois_raw, name='rpn_rois')

    rpn_losses = None
    if(add_loss_functions):
        # RPN targets
        # Comment: rpn_cls_score is only passed   vvv   to get width and height of the conv feature map ...
        atl = user_function(AnchorTargetLayer(rpn_cls_score, scaled_gt_boxes, im_info, param_str=proposal_layer_param_string))
        rpn_labels = atl.outputs[0]
        rpn_bbox_targets = atl.outputs[1]
        rpn_bbox_inside_weights = atl.outputs[2]

        # For loss functions: ignore label predictions for the 'ignore label',
        # i.e. set target and prediction to 0 --> needs to be softmaxed before
        rpn_labels_rshp = reshape(rpn_labels, (1, num_predictions))
        ignore = user_function(IgnoreLabel(rpn_cls_prob, rpn_labels_rshp, ignore_label=-1))
        rpn_cls_prob_ignore = ignore.outputs[0]
        fg_targets = ignore.outputs[1]
        bg_targets = 1 - fg_targets
        rpn_labels_ignore = splice(bg_targets, fg_targets, axis=0)

        # RPN losses
        rpn_loss_cls = cross_entropy_with_softmax(rpn_cls_prob_ignore, rpn_labels_ignore, axis=0)
        rpn_loss_bbox = user_function(SmoothL1Loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights))
        rpn_losses = plus(reduce_sum(rpn_loss_cls), reduce_sum(rpn_loss_bbox), name="rpn_losses")

    return rpn_rois, rpn_losses

def create_proposal_target_layer(rpn_rois, scaled_gt_boxes, num_classes):
    '''
    Creates a proposal target layer that is used for training an object detection network as proposed in the "Faster R-CNN" paper:
        Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun:
        "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"

    Assigns object detection proposals to ground-truth targets.
    Produces proposal classification labels and bounding-box regression targets.
    It also adds gt_boxes to candidates and samples fg and bg rois for training.

    Args:
        rpn_rois:        The proposed ROIs, e.g. from a region proposal network
        scaled_gt_boxes: The ground truth boxes as (x1, y1, x2, y2, label). Coordinates are absolute pixels wrt. the input image.
        num_classes:     The number of classes in the data set

    Returns:
        rpn_target_rois - a set of rois containing the ground truth and a number of sampled fg and bg ROIs
        label_targets - the target labels for the rois
        bbox_targets - the regression coefficient targets for the rois
        bbox_inside_weights - the weights for the regression loss
    '''

    ptl_param_string = "'num_classes': {}".format(num_classes)
    ptl = user_function(ProposalTargetLayer(rpn_rois, scaled_gt_boxes, param_str=ptl_param_string))
    rois = alias(ptl.outputs[0], name='rpn_target_rois')
    label_targets = alias(ptl.outputs[1], name='label_targets')
    bbox_targets = alias(ptl.outputs[2], name='bbox_targets')
    bbox_inside_weights = alias(ptl.outputs[3], name='bbox_inside_w')

    return rois, label_targets, bbox_targets, bbox_inside_weights



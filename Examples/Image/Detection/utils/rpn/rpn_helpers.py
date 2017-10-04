# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import cntk
from cntk import reduce_sum, ops
from cntk import user_function, relu, softmax, slice, splice, reshape, element_times, plus, minus, alias, classification_error
from cntk.initializer import glorot_uniform, normal
from cntk.layers import Convolution
from cntk.losses import cross_entropy_with_softmax
from utils.rpn.anchor_target_layer import AnchorTargetLayer
from utils.rpn.proposal_layer import ProposalLayer
from utils.rpn.proposal_target_layer import ProposalTargetLayer
from utils.rpn.cntk_smoothL1_loss import SmoothL1Loss

def create_rpn(conv_out, scaled_gt_boxes, im_info, cfg, add_loss_functions=True):
    '''
    Creates a region proposal network for object detection as proposed in the "Faster R-CNN" paper:
        Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun:
        "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"

    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").

    Args:
        conv_out:        The convolutional feature map, i.e. the output of the conv layers from the pretrained classification network
        scaled_gt_boxes: The ground truth boxes as (x1, y1, x2, y2, label). Coordinates are absolute pixels wrt. the input image.
        im_info:         A CNTK variable or constant containing
                         (pad_width, pad_height, scaled_image_width, scaled_image_height, orig_img_width, orig_img_height)
                         e.g. (1000, 1000, 1000, 600, 500, 300) for an original image of 600x300 that is scaled and padded to 1000x1000
        cfg:             The configuration dictionary
        add_loss_functions: If set to True rpn_losses will be returned, otherwise None is returned for the losses

    Returns:
        rpn_rois - the proposed ROIs
        rpn_losses - the losses (SmoothL1 loss for bbox regression plus cross entropy for objectness)
    '''

    # RPN network
    # init = 'normal', initValueScale = 0.01, initBias = 0.1
    num_channels = cfg["MODEL"].RPN_NUM_CHANNELS
    rpn_conv_3x3 = Convolution((3, 3), num_channels, activation=relu, pad=True, strides=1,
                                init = normal(scale=0.01), init_bias=0.0)(conv_out)
    rpn_cls_score = Convolution((1, 1), 18, activation=None, name="rpn_cls_score",
                                init = normal(scale=0.01), init_bias=0.0)(rpn_conv_3x3)  # 2(bg/fg)  * 9(anchors)
    rpn_bbox_pred = Convolution((1, 1), 36, activation=None, name="rpn_bbox_pred",
                                init = normal(scale=0.01), init_bias=0.0)(rpn_conv_3x3)  # 4(coords) * 9(anchors)

    # apply softmax to get (bg, fg) probabilities and reshape predictions back to grid of (18, H, W)
    num_predictions = int(rpn_cls_score.shape[0] / 2)
    rpn_cls_score_rshp = reshape(rpn_cls_score, (2, num_predictions, rpn_cls_score.shape[1], rpn_cls_score.shape[2]), name="rpn_cls_score_rshp")
    p_rpn_cls_score_rshp = cntk.placeholder()
    rpn_cls_sm = softmax(p_rpn_cls_score_rshp, axis=0)
    rpn_cls_prob = cntk.as_block(rpn_cls_sm, [(p_rpn_cls_score_rshp, rpn_cls_score_rshp)], 'Softmax', 'rpn_cls_prob')
    rpn_cls_prob_reshape = reshape(rpn_cls_prob, rpn_cls_score.shape, name="rpn_cls_prob_reshape")

    # proposal layer
    rpn_rois = create_proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg)

    rpn_losses = None
    if(add_loss_functions):
        # RPN targets
        # Comment: rpn_cls_score is only passed   vvv   to get width and height of the conv feature map ...
        proposal_layer_params = "'feat_stride': {}\n'scales':\n - {}". \
            format(cfg["MODEL"].FEATURE_STRIDE, "\n - ".join([str(v) for v in cfg["DATA"].PROPOSAL_LAYER_SCALES]))
        atl = user_function(AnchorTargetLayer(rpn_cls_score, scaled_gt_boxes, im_info,
                                              rpn_batch_size=cfg["TRAIN"].RPN_BATCHSIZE,
                                              rpn_fg_fraction=cfg["TRAIN"].RPN_FG_FRACTION,
                                              clobber_positives=cfg["TRAIN"].RPN_CLOBBER_POSITIVES,
                                              positive_overlap=cfg["TRAIN"].RPN_POSITIVE_OVERLAP,
                                              negative_overlap=cfg["TRAIN"].RPN_NEGATIVE_OVERLAP,
                                              param_str=proposal_layer_params))
        rpn_labels = atl.outputs[0]
        rpn_bbox_targets = atl.outputs[1]
        rpn_bbox_inside_weights = atl.outputs[2]

        # classification loss
        p_rpn_labels = cntk.placeholder()
        p_rpn_cls_score_rshp = cntk.placeholder()

        keeps = cntk.greater_equal(p_rpn_labels, 0.0)
        fg_labels = element_times(p_rpn_labels, keeps, name="fg_targets")
        bg_labels = minus(1, fg_labels, name="bg_targets")
        rpn_labels_ignore = splice(bg_labels, fg_labels, axis=0)
        rpn_ce = cross_entropy_with_softmax(p_rpn_cls_score_rshp, rpn_labels_ignore, axis=0)
        rpn_loss_cls = element_times(rpn_ce, keeps)

        # The terms that are accounted for in the cls loss are those that have a label >= 0
        cls_num_terms = reduce_sum(keeps)
        cls_normalization_factor = 1.0 / cls_num_terms
        normalized_rpn_cls_loss = reduce_sum(rpn_loss_cls) * cls_normalization_factor

        reduced_rpn_loss_cls = cntk.as_block(normalized_rpn_cls_loss,
                                             [(p_rpn_labels, rpn_labels), (p_rpn_cls_score_rshp, rpn_cls_score_rshp)],
                                             'CE_with_ignore', 'norm_rpn_cls_loss')

        # regression loss
        p_rpn_bbox_pred = cntk.placeholder()
        p_rpn_bbox_targets = cntk.placeholder()
        p_rpn_bbox_inside_weights = cntk.placeholder()
        rpn_loss_bbox = SmoothL1Loss(cfg.SIGMA_RPN_L1, p_rpn_bbox_pred, p_rpn_bbox_targets, p_rpn_bbox_inside_weights, 1.0)
        # The bbox loss is normalized by the rpn batch size
        bbox_normalization_factor = 1.0 / cfg["TRAIN"].RPN_BATCHSIZE
        normalized_rpn_bbox_loss = reduce_sum(rpn_loss_bbox) * bbox_normalization_factor

        reduced_rpn_loss_bbox = cntk.as_block(normalized_rpn_bbox_loss,
                                              [(p_rpn_bbox_pred, rpn_bbox_pred), (p_rpn_bbox_targets, rpn_bbox_targets),
                                               (p_rpn_bbox_inside_weights, rpn_bbox_inside_weights)],
                                              'SmoothL1Loss', 'norm_rpn_bbox_loss')

        rpn_losses = plus(reduced_rpn_loss_cls, reduced_rpn_loss_bbox, name="rpn_losses")

    return rpn_rois, rpn_losses

def create_proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg, use_native_proposal_layer=False):
    layer_config = {}
    layer_config["feat_stride"] = cfg["MODEL"].FEATURE_STRIDE
    layer_config["scales"] = cfg["DATA"].PROPOSAL_LAYER_SCALES

    layer_config["train_pre_nms_topN"] = cfg["TRAIN"].RPN_PRE_NMS_TOP_N
    layer_config["train_post_nms_topN"] = cfg["TRAIN"].RPN_POST_NMS_TOP_N
    layer_config["train_nms_thresh"] = float(cfg["TRAIN"].RPN_NMS_THRESH)
    layer_config["train_min_size"] = float(cfg["TRAIN"].RPN_MIN_SIZE)

    layer_config["test_pre_nms_topN"] = cfg["TEST"].RPN_PRE_NMS_TOP_N
    layer_config["test_post_nms_topN"] = cfg["TEST"].RPN_POST_NMS_TOP_N
    layer_config["test_nms_thresh"] = float(cfg["TEST"].RPN_NMS_THRESH)
    layer_config["test_min_size"] = float(cfg["TEST"].RPN_MIN_SIZE)

    if use_native_proposal_layer:
        cntk.ops.register_native_user_function('ProposalLayerOp',
                                               'Cntk.ProposalLayerLib-' + cntk.__version__.rstrip('+'),
                                               'CreateProposalLayer')
        rpn_rois_raw = ops.native_user_function('ProposalLayerOp', [rpn_cls_prob_reshape, rpn_bbox_pred, im_info],
                                                layer_config, 'native_proposal_layer')
    else:
        rpn_rois_raw = user_function(ProposalLayer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, layer_config))

    return alias(rpn_rois_raw, name='rpn_rois')

def create_proposal_target_layer(rpn_rois, scaled_gt_boxes, cfg):
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

    ptl_param_string = "'num_classes': {}".format(cfg["DATA"].NUM_CLASSES)
    ptl = user_function(ProposalTargetLayer(rpn_rois, scaled_gt_boxes,
                                            batch_size=cfg.NUM_ROI_PROPOSALS,
                                            fg_fraction=cfg["TRAIN"].FG_FRACTION,
                                            normalize_targets=cfg.BBOX_NORMALIZE_TARGETS,
                                            normalize_means=cfg.BBOX_NORMALIZE_MEANS,
                                            normalize_stds=cfg.BBOX_NORMALIZE_STDS,
                                            fg_thresh=cfg["TRAIN"].FG_THRESH,
                                            bg_thresh_hi=cfg["TRAIN"].BG_THRESH_HI,
                                            bg_thresh_lo=cfg["TRAIN"].BG_THRESH_LO,
                                            param_str=ptl_param_string))

    # use an alias if you need to access the outputs, e.g., when cloning a trained network
    rois = alias(ptl.outputs[0], name='rpn_target_rois')
    label_targets = ptl.outputs[1]
    bbox_targets = ptl.outputs[2]
    bbox_inside_weights = ptl.outputs[3]

    return rois, label_targets, bbox_targets, bbox_inside_weights



# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os, sys
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path))
sys.path.append(os.path.join(abs_path, ".."))

import pytest
import numpy as np
import cntk
from cntk import user_function
from cntk.ops import input_variable
from rpn.proposal_layer import ProposalLayer as CntkProposalLayer
from rpn.proposal_target_layer import ProposalTargetLayer as CntkProposalTargetLayer
from rpn.anchor_target_layer import AnchorTargetLayer as CntkAnchorTargetLayer
from caffe_layers.proposal_layer import ProposalLayer as CaffeProposalLayer
from caffe_layers.proposal_target_layer import ProposalTargetLayer as CaffeProposalTargetLayer
from caffe_layers.anchor_target_layer import AnchorTargetLayer as CaffeAnchorTargetLayer

def test_proposal_layer():
    cls_prob_shape_cntk = (18,61,61)
    cls_prob_shape_caffe = (18,61,61)
    rpn_bbox_shape = (36, 61, 61)
    dims_info_shape = (6,)
    im_info = [1000, 1000, 1]

    # Create input tensors with values
    cls_prob =  np.random.random_sample(cls_prob_shape_cntk).astype(np.float32)
    rpn_bbox_pred = np.random.random_sample(rpn_bbox_shape).astype(np.float32)
    dims_input = np.array([1000, 1000, 1000, 1000, 1000, 1000]).astype(np.float32)

    # Create CNTK layer and call forward
    cls_prob_var = input_variable(cls_prob_shape_cntk)
    rpn_bbox_var = input_variable(rpn_bbox_shape)
    dims_info_var = input_variable(dims_info_shape)

    cntk_layer = user_function(CntkProposalLayer(cls_prob_var, rpn_bbox_var, dims_info_var))
    state, cntk_output = cntk_layer.forward({cls_prob_var: [cls_prob], rpn_bbox_var: [rpn_bbox_pred], dims_info_var: dims_input})
    cntk_proposals = cntk_output[next(iter(cntk_output))][0]

    # Create Caffe layer and call forward
    cls_prob_caffe = cls_prob.reshape(cls_prob_shape_caffe)
    bottom = [np.array([cls_prob_caffe]),np.array([rpn_bbox_pred]),np.array([im_info])]
    top = None # handled through return statement in caffe layer for unit testing

    param_str = "'feat_stride': 16"
    caffe_layer = CaffeProposalLayer()
    caffe_layer.set_param_str(param_str)
    caffe_layer.setup(bottom, top)
    caffe_output = caffe_layer.forward(bottom, top)
    caffe_proposals = caffe_output[:,1:]

    # assert that results are exactly the same
    assert cntk_proposals.shape == caffe_proposals.shape
    assert np.allclose(cntk_proposals, caffe_proposals, rtol=0.0, atol=0.0)
    print("Verified ProposalLayer")

def test_proposal_target_layer():
    num_rois = 400
    all_rois_shape_cntk = (num_rois,4)
    num_gt_boxes = 50
    gt_boxes_shape_cntk = (num_gt_boxes,5)

    # Create input tensors with values
    x1y1 = np.random.random_sample((num_rois, 2)) * 500
    wh = np.random.random_sample((num_rois, 2)) * 400
    x2y2 = x1y1 + wh + 50
    all_rois = np.hstack((x1y1, x2y2)).astype(np.float32)

    x1y1 = np.random.random_sample((num_gt_boxes, 2)) * 500
    wh = np.random.random_sample((num_gt_boxes, 2)) * 400
    x2y2 = x1y1 + wh + 50
    label = np.random.random_sample((num_gt_boxes, 1))
    label = (label * 17.0)
    gt_boxes = np.hstack((x1y1, x2y2, label)).astype(np.float32)

    # Create CNTK layer and call forward
    all_rois_var = input_variable(all_rois_shape_cntk)
    gt_boxes_var = input_variable(gt_boxes_shape_cntk)

    cntk_layer = user_function(CntkProposalTargetLayer(all_rois_var, gt_boxes_var, param_str="'num_classes': 17", deterministic=True))
    state, cntk_output = cntk_layer.forward({all_rois_var: [all_rois], gt_boxes_var: [gt_boxes]})

    roi_key = [k for k in cntk_output if 'rpn_target_rois_raw' in str(k)][0]
    labels_key = [k for k in cntk_output if 'label_targets_raw' in str(k)][0]
    bbox_key = [k for k in cntk_output if 'bbox_targets_raw' in str(k)][0]
    bbox_w_key = [k for k in cntk_output if 'bbox_inside_w_raw' in str(k)][0]

    cntk_rois = cntk_output[roi_key][0]
    cntk_labels_one_hot = cntk_output[labels_key][0]
    cntk_bbox_targets = cntk_output[bbox_key][0]
    cntk_bbox_inside_weights = cntk_output[bbox_w_key][0]

    cntk_labels = np.argmax(cntk_labels_one_hot, axis=1)

    # Create Caffe layer and call forward
    zeros = np.zeros((all_rois.shape[0], 1), dtype=gt_boxes.dtype)
    all_rois_caffe = np.hstack((zeros, all_rois))

    bottom = [np.array(all_rois_caffe),np.array(gt_boxes)]
    top = None # handled through return statement in caffe layer for unit testing

    param_str = "'num_classes': 17"
    caffe_layer = CaffeProposalTargetLayer()
    caffe_layer.set_param_str(param_str)
    caffe_layer.setup(bottom, top)
    caffe_layer.set_deterministic_mode()

    caffe_rois, caffe_labels, caffe_bbox_targets, caffe_bbox_inside_weights = caffe_layer.forward(bottom, top)
    caffe_rois = caffe_rois[:,1:]

    num_caffe_rois = caffe_rois.shape[0]
    cntk_rois = cntk_rois[:num_caffe_rois,:]
    cntk_labels = cntk_labels[:num_caffe_rois]
    cntk_bbox_targets = cntk_bbox_targets[:num_caffe_rois,:]
    cntk_bbox_inside_weights = cntk_bbox_inside_weights[:num_caffe_rois,:]

    # assert that results are exactly the same
    assert cntk_rois.shape == caffe_rois.shape
    assert cntk_labels.shape == caffe_labels.shape
    assert cntk_bbox_targets.shape == caffe_bbox_targets.shape
    assert cntk_bbox_inside_weights.shape == caffe_bbox_inside_weights.shape

    caffe_labels = [int(x) for x in caffe_labels]

    assert np.allclose(cntk_rois, caffe_rois, rtol=0.0, atol=0.0)
    assert np.allclose(cntk_labels, caffe_labels, rtol=0.0, atol=0.0)
    assert np.allclose(cntk_bbox_targets, caffe_bbox_targets, rtol=0.0, atol=0.0)
    assert np.allclose(cntk_bbox_inside_weights, caffe_bbox_inside_weights, rtol=0.0, atol=0.0)
    print("Verified ProposalTargetLayer")

def test_anchor_target_layer():
    rpn_cls_score_shape_cntk = (1, 18, 61, 61)
    num_gt_boxes = 50
    gt_boxes_shape_cntk = (num_gt_boxes,5)
    dims_info_shape = (6,)
    im_info = [1000, 1000, 1]

    # Create input tensors with values
    rpn_cls_score_dummy = np.random.random_sample(rpn_cls_score_shape_cntk).astype(np.float32)
    dims_input = np.array([1000, 1000, 1000, 1000, 1000, 1000]).astype(np.float32)

    x1y1 = np.random.random_sample((num_gt_boxes, 2)) * 500
    wh = np.random.random_sample((num_gt_boxes, 2)) * 400
    x2y2 = x1y1 + wh + 50
    label = np.random.random_sample((num_gt_boxes, 1))
    label = (label * 17.0)
    gt_boxes = np.hstack((x1y1, x2y2, label)).astype(np.float32)

    # Create CNTK layer and call forward
    rpn_cls_score_var = input_variable(rpn_cls_score_shape_cntk)
    gt_boxes_var = input_variable(gt_boxes_shape_cntk)
    dims_info_var = input_variable(dims_info_shape)

    cntk_layer = user_function(CntkAnchorTargetLayer(rpn_cls_score_var, gt_boxes_var, dims_info_var, deterministic=True))
    state, cntk_output = cntk_layer.forward({rpn_cls_score_var: [rpn_cls_score_dummy], gt_boxes_var: [gt_boxes], dims_info_var: dims_input})

    obj_key = [k for k in cntk_output if 'objectness_target' in str(k)][0]
    bbt_key = [k for k in cntk_output if 'rpn_bbox_target' in str(k)][0]
    bbw_key = [k for k in cntk_output if 'rpn_bbox_inside_w' in str(k)][0]

    cntk_objectness_target = cntk_output[obj_key][0]
    cntk_bbox_targets = cntk_output[bbt_key][0]
    cntk_bbox_inside_w = cntk_output[bbw_key][0]

    # Create Caffe layer and call forward
    bottom = [np.array(rpn_cls_score_dummy),np.array(gt_boxes), np.array(im_info)]
    top = None # handled through return statement in caffe layer for unit testing

    param_str = "'feat_stride': 16"
    caffe_layer = CaffeAnchorTargetLayer()
    caffe_layer.set_param_str(param_str)
    caffe_layer.setup(bottom, top)
    caffe_layer.set_deterministic_mode()

    caffe_objectness_target, caffe_bbox_targets, caffe_bbox_inside_w = caffe_layer.forward(bottom, top)

    # assert that results are exactly the same
    assert cntk_objectness_target.shape == caffe_objectness_target.shape
    assert cntk_bbox_targets.shape == caffe_bbox_targets.shape
    assert cntk_bbox_inside_w.shape == caffe_bbox_inside_w.shape

    assert np.allclose(cntk_objectness_target, caffe_objectness_target, rtol=0.0, atol=0.0)
    assert np.allclose(cntk_bbox_targets, caffe_bbox_targets, rtol=0.0, atol=0.0)
    assert np.allclose(cntk_bbox_inside_w, caffe_bbox_inside_w, rtol=0.0, atol=0.0)
    print("Verified AnchorTargetLayer")

if __name__ == '__main__':
    test_proposal_layer()
    test_proposal_target_layer()
    test_anchor_target_layer()

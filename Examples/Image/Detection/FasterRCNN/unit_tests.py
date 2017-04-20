import os, sys
from cntk import input as input_variable, user_function

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
sys.path.append(os.path.join(abs_path, "..", ".."))
sys.path.append(os.path.join(abs_path, "lib"))
sys.path.append(os.path.join(abs_path, "lib", "rpn"))
sys.path.append(os.path.join(abs_path, "lib", "nms"))
sys.path.append(os.path.join(abs_path, "lib", "nms", "gpu"))

import pytest
import numpy as np
from lib.rpn.proposal_layer import ProposalLayer as CntkProposalLayer
from lib.rpn.proposal_layer_caffe import ProposalLayer as CaffeProposalLayer

def test_proposal_layer():
    cls_prob_shape_cntk = (2,9,61,61)
    cls_prob_shape_caffe = (18,61,61)
    rpn_bbox_shape = (36, 61, 61)
    im_info = [1000, 1000, 1]
    test_specific_values = False

    # Create input tensors with values
    if test_specific_values:
        bg_probs = [0.2, 0.05, 0.05, 0.0, 0.0, 0.1, 0.1, 0.0, 0.5]
        fg_probs = np.ones(9) - bg_probs
        cls_prob = np.zeros((61, 61, 9, 2))
        cls_prob[:, :, :, 0] = bg_probs
        cls_prob[:, :, :, 1] = fg_probs
        cls_prob = np.ascontiguousarray(cls_prob.transpose(3, 2, 1, 0)).astype(np.float32)

        bbox_pred = [0.2, -0.1, 0.3, -0.4] * 9
        rpn_bbox_pred = np.zeros((61, 61, 36))
        rpn_bbox_pred[:, :, :] = bbox_pred
        rpn_bbox_pred = np.ascontiguousarray(rpn_bbox_pred.transpose(2, 1, 0)).astype(np.float32)
    else:
        cls_prob =  np.random.random_sample(cls_prob_shape_cntk).astype(np.float32)
        rpn_bbox_pred = np.random.random_sample(rpn_bbox_shape).astype(np.float32)

    # Create CNTK layer and call forward
    cls_prob_var = input_variable(cls_prob_shape_cntk)
    rpn_bbox_var = input_variable(rpn_bbox_shape)

    cntk_layer = user_function(CntkProposalLayer(cls_prob_var, rpn_bbox_var, im_info=im_info))
    state, cntk_output = cntk_layer.forward({cls_prob_var: [cls_prob], rpn_bbox_var: [rpn_bbox_pred]})
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

if __name__ == '__main__':
    test_proposal_layer()
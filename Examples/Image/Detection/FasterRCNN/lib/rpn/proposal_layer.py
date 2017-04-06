# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

from cntk import output_variable
from cntk.ops.functions import UserFunction
import numpy as np
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms

DEBUG = cfg["CNTK"].DEBUG_LAYERS
debug_fwd = cfg["CNTK"].DEBUG_FWD
debug_bkw = cfg["CNTK"].DEBUG_BKW

class ProposalLayer(UserFunction):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, arg1, arg2, name='ProposalLayer', im_info=None):
        super(ProposalLayer, self).__init__([arg1, arg2], name=name)
        # parse the layer parameter string, which must be valid YAML
        # layer_params = yaml.load(self.param_str_)

        self._feat_stride = 16 # layer_params['feat_stride']
        anchor_scales = (8, 16, 32) # layer_params.get('scales', (8, 16, 32))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        self._im_info = im_info

        if DEBUG:
            print ('feat_stride: {}'.format(self._feat_stride))
            print ('anchors:')
            print (self._anchors)

    def infer_outputs(self):
        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(1, 5)
        # for CNTK the proposal shape is [4 x roisPerImage], and mirrored in Python
        cfg_key = 'TRAIN' # str(self.phase) # either 'TRAIN' or 'TEST'
        proposalShape = (cfg[cfg_key].RPN_POST_NMS_TOP_N, 4)

        # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #    top[1].reshape(1, 1, 1, 1)

        return [output_variable(proposalShape, self.inputs[0].dtype, self.inputs[0].dynamic_axes,
                                name="pl_rois", needs_gradient=False)] # , name="rpn_rois"

    # returns
    # - pred_boxes (n, 4) as [x_low, y_low, x_high, y_high]
    def forward(self, arguments, device=None, outputs_to_retain=None):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)
        if debug_fwd: print("--> Entering forward in {}".format(self.name))

        bottom = arguments
        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        cfg_key = 'TRAIN' # str(self.phase) # either 'TRAIN' or 'TEST'
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        # scores = bottom[0][:, self._num_anchors:, :, :]
        scores = bottom[0][:,1,:, :, :]
        bbox_deltas = bottom[1]
        im_info = self._im_info

        if DEBUG:
            print ('im_size: ({}, {})'.format(im_info[0], im_info[1]))
            print ('scale: {}'.format(im_info[2]))

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        if DEBUG:
            print ('score map size: {}'.format(scores.shape))

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]

        # pad with zeros if too few rois were found
        num_found_proposals = proposals.shape[0]
        if num_found_proposals < post_nms_topN:
            if DEBUG:
                print("Only {} proposals generated in ProposalLayer".format(num_found_proposals))
            proposals_padded = np.zeros(((post_nms_topN,) + proposals.shape[1:]), dtype=np.float32)
            proposals_padded[:num_found_proposals, :] = proposals
            proposals = proposals_padded

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        # for CNTK: add batch axis to output shape
        proposals.shape = (1,) + proposals.shape

        # top[0].reshape(*(blob.shape))
        # top[0].data[...] = blob
        return None, proposals

    def backward(self, state, root_gradients, variables):
        """This layer does not propagate gradients."""
        if debug_bkw: print("<-- Entering backward in {}".format(self.name))
        pass

    def clone(self, cloned_inputs):
        return ProposalLayer(cloned_inputs[0], cloned_inputs[1], im_info=self._im_info)


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

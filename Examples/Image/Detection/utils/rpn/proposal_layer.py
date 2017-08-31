# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk import output_variable, FreeDimension
from cntk.ops.functions import UserFunction
import numpy as np
import yaml
from utils.rpn.generate_anchors import generate_anchors
from utils.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from utils.nms_wrapper import nms

DEBUG = False

class ProposalLayer(UserFunction):
    '''
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    '''

    def __init__(self, arg1, arg2, arg3, layer_config, name='ProposalLayer'):
        super(ProposalLayer, self).__init__([arg1, arg2, arg3], attributes=layer_config, name=name)

        self._layer_config = layer_config
        self._feat_stride = 16 if 'feat_stride' not in layer_config else layer_config['feat_stride']
        anchor_scales = [8, 16, 32] if 'scales' not in layer_config else layer_config['scales']

        # parse the layer parameter string, which must be valid YAML
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]

        if DEBUG:
            print ('feat_stride: {}'.format(self._feat_stride))
            print ('anchors:')
            print (self._anchors)

    def infer_outputs(self):
        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # for CNTK the proposal shape is [4 x roisPerImage], and mirrored in Python
        proposalShape = (FreeDimension, 4)

        return [output_variable(proposalShape, self.inputs[0].dtype, self.inputs[0].dynamic_axes,
                            name="rpn_rois_raw", needs_gradient=False)]

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

        # use potentially different number of proposals for training vs evaluation
        if len(outputs_to_retain) == 0:
            # print("EVAL")
            pre_nms_topN = self._layer_config['test_pre_nms_topN']
            post_nms_topN = self._layer_config['test_post_nms_topN']
            nms_thresh = self._layer_config['test_nms_thresh']
            min_size = self._layer_config['test_min_size']
        else:
            pre_nms_topN = self._layer_config['train_pre_nms_topN']
            post_nms_topN = self._layer_config['train_post_nms_topN']
            nms_thresh = self._layer_config['train_nms_thresh']
            min_size = self._layer_config['train_min_size']

        bottom = arguments
        assert bottom[0].shape[0] == 1, \
            'Only single item batches are supported'

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0][:, self._num_anchors:, :, :]
        bbox_deltas = bottom[1]
        im_info = bottom[2][0]

        if DEBUG:
            # im_info = (pad_width, pad_height, scaled_image_width, scaled_image_height, orig_img_width, orig_img_height)
            # e.g.(1000, 1000, 1000, 600, 500, 300) for an original image of 600x300 that is scaled and padded to 1000x1000
            print ('im_size: ({}, {})'.format(im_info[0], im_info[1]))
            print ('scaled im_size: ({}, {})'.format(im_info[2], im_info[3]))
            print ('original im_size: ({}, {})'.format(im_info[4], im_info[5]))

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
        proposals = clip_boxes(proposals, im_info)

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale. Original size = im_info[4:6], scaled size = im_info[2:4])
        cntk_image_scale = im_info[2] / im_info[4]
        keep = _filter_boxes(proposals, min_size * cntk_image_scale)
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort(kind='mergesort')[::-1]
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

        return None, proposals

    def backward(self, state, root_gradients, variables):
        """This layer does not propagate gradients."""
        pass

    def clone(self, cloned_inputs):
        return ProposalLayer(cloned_inputs[0], cloned_inputs[1], cloned_inputs[2], layer_config=self._layer_config)

    def serialize(self):
        internal_state = {}
        internal_state['layer_config'] = self._layer_config
        return internal_state

    @staticmethod
    def deserialize(inputs, name, state):
        return ProposalLayer(inputs[0], inputs[1], inputs[2], layer_config=state['layer_config'], name=name)

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

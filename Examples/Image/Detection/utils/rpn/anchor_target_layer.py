# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
from cntk import output_variable
from cntk.ops.functions import UserFunction
import yaml
import numpy as np
import numpy.random as npr
from utils.rpn.generate_anchors import generate_anchors
from utils.rpn.bbox_transform import bbox_transform
from utils.cython_modules.cython_bbox import bbox_overlaps

try:
    from config import cfg
except ImportError:
    from utils.default_config import cfg

DEBUG = False

class AnchorTargetLayer(UserFunction):
    '''
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    '''

    def __init__(self, arg1, arg2, arg3, name='AnchorTargetLayer', param_str=None, cfm_shape=None, deterministic=False):
        super(AnchorTargetLayer, self).__init__([arg1, arg2, arg3], name=name)
        self.param_str_ = param_str if param_str is not None else "'feat_stride': 16\n'scales':\n - 8 \n - 16 \n - 32"

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = layer_params['feat_stride']
        self._cfm_shape = cfm_shape
        self._determininistic_mode = deterministic

        if DEBUG:
            print ('anchors:')
            print (self._anchors)
            print ('anchor shapes:')
            print (np.hstack((
                self._anchors[:, 2::4] - self._anchors[:, 0::4],
                self._anchors[:, 3::4] - self._anchors[:, 1::4],
            )))
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = False # layer_params.get('allowed_border', 0)

    def infer_outputs(self):
        # This is a necessary work around since anfter cloning the cloned inputs are just place holders without the proper shape
        if self._cfm_shape is None:
            self._cfm_shape = self.inputs[0].shape
        height, width = self._cfm_shape[-2:]

        if DEBUG:
            print('AnchorTargetLayer: height', height, 'width', width)

        A = self._num_anchors
        # labels
        labelShape = (1, A, height, width)
        # Comment: this layer uses encoded labels, while in CNTK we mostly use one hot labels
        # bbox_targets
        bbox_target_shape = (1, A * 4, height, width)
        # bbox_inside_weights
        bbox_inside_weights_shape = (1, A * 4, height, width)

        return [output_variable(labelShape, self.inputs[0].dtype, self.inputs[0].dynamic_axes,
                                name="objectness_target", needs_gradient=False),
                output_variable(bbox_target_shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes,
                                name="rpn_bbox_target", needs_gradient=False),
                output_variable(bbox_inside_weights_shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes,
                                name="rpn_bbox_inside_w", needs_gradient=False),]

    def forward(self, arguments, outputs, device=None, outputs_to_retain=None):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        bottom = arguments

        # map of shape (..., H, W)
        height, width = bottom[0].shape[-2:]
        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[1][0,:]
        # im_info
        im_info = bottom[2][0]

        # remove zero padded ground truth boxes
        keep = np.where(
            ((gt_boxes[:,2] - gt_boxes[:,0]) > 0) &
            ((gt_boxes[:,3] - gt_boxes[:,1]) > 0)
        )
        gt_boxes = gt_boxes[keep]

        if DEBUG:
            print ('')
            # im_info = (pad_width, pad_height, scaled_image_width, scaled_image_height, orig_img_width, orig_img_height)
            # e.g.(1000, 1000, 1000, 600, 500, 300) for an original image of 600x300 that is scaled and padded to 1000x1000
            print ('im_size: ({}, {})'.format(im_info[0], im_info[1]))
            print ('scaled im_size: ({}, {})'.format(im_info[2], im_info[3]))
            print ('original im_size: ({}, {})'.format(im_info[4], im_info[5]))
            print ('height, width: ({}, {})'.format(height, width))
            print ('rpn: gt_boxes.shape', gt_boxes.shape)
            #print ('rpn: gt_boxes', gt_boxes)

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = (self._anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        padded_wh = im_info[0:2]
        scaled_wh = im_info[2:4]
        xy_offset = (padded_wh - scaled_wh) / 2
        xy_min = xy_offset
        xy_max = xy_offset + scaled_wh

        inds_inside = np.where(
            (all_anchors[:, 0] >= xy_min[0] - self._allowed_border) &
            (all_anchors[:, 1] >= xy_min[1] - self._allowed_border) &
            (all_anchors[:, 2] < xy_max[0] + self._allowed_border) &  # width
            (all_anchors[:, 3] < xy_max[1] + self._allowed_border)    # height
        )[0]

        if DEBUG:
            print ('total_anchors', total_anchors)
            print ('inds_inside', len(inds_inside))

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        if DEBUG:
            print ('anchors.shape', anchors.shape)
            print('gt_boxes.shape', gt_boxes.shape)

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not cfg["TRAIN"].RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg["TRAIN"].RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg["TRAIN"].RPN_POSITIVE_OVERLAP] = 1

        if cfg["TRAIN"].RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg["TRAIN"].RPN_NEGATIVE_OVERLAP] = 0

        # subsample positive labels if we have too many
        num_fg = int(cfg["TRAIN"].RPN_FG_FRACTION * cfg["TRAIN"].RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            if self._determininistic_mode:
                disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
            else:
                disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = cfg["TRAIN"].RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            if self._determininistic_mode:
                disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
            else:
                disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array((1.0, 1.0, 1.0, 1.0))

        if DEBUG:
            self._sums += bbox_targets[labels == 1, :].sum(axis=0)
            self._squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels == 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print ('means:')
            print (means)
            print ('stdevs:')
            print (stds)

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)

        if DEBUG:
            print ('rpn: max max_overlap', np.max(max_overlaps))
            print ('rpn: num_positive', np.sum(labels == 1))
            print ('rpn: num_negative', np.sum(labels == 0))
            self._fg_sum += np.sum(labels == 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            print ('rpn: num_positive avg', self._fg_sum / self._count)
            print ('rpn: num_negative avg', self._bg_sum / self._count)

        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        outputs[self.outputs[0]] = np.ascontiguousarray(labels)

        # bbox_targets
        bbox_targets = bbox_targets.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        outputs[self.outputs[1]] = np.ascontiguousarray(bbox_targets)

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width
        outputs[self.outputs[2]] = np.ascontiguousarray(bbox_inside_weights)

        # No state needs to be passed to backward() so we just pass None
        return None

    def backward(self, state, root_gradients, variables):
        """This layer does not propagate gradients."""
        pass

    def clone(self, cloned_inputs):
        return AnchorTargetLayer(cloned_inputs[0], cloned_inputs[1], cloned_inputs[2], param_str=self.param_str_, cfm_shape=self._cfm_shape)

    def serialize(self):
        internal_state = {}
        internal_state['param_str'] = self.param_str_
        return internal_state

    @staticmethod
    def deserialize(inputs, name, state):
        param_str = state['param_str']
        return AnchorTargetLayer(inputs[0], inputs[1], inputs[2], name=name, param_str=param_str)


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

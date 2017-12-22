# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk import output_variable, FreeDimension
from cntk.ops.functions import UserFunction
import yaml
import numpy as np
import numpy.random as npr
from utils.rpn.bbox_transform import bbox_transform
from utils.cython_modules.cython_bbox import bbox_overlaps

DEBUG = False

class ProposalTargetLayer(UserFunction):
    '''
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    '''
    
    def __init__(self, arg1, arg2,
                 batch_size=128,
                 fg_fraction=0.25,
                 normalize_targets=True,
                 normalize_means=[0.0, 0.0, 0.0, 0.0],
                 normalize_stds=[0.1, 0.1, 0.2, 0.2],
                 fg_thresh=0.5,
                 bg_thresh_hi=0.5,
                 bg_thresh_lo=0.0,
                 param_str=None,
                 name='ProposalTargetLayer', deterministic=False):
        super(ProposalTargetLayer, self).__init__([arg1, arg2], name=name)
        self._batch_size = batch_size
        self._fg_fraction = fg_fraction
        self._normalize_targets = normalize_targets
        self._normalize_means = normalize_means
        self._normalize_stds = normalize_stds
        self._fg_thresh = fg_thresh
        self._bg_thresh_hi = bg_thresh_hi
        self._bg_thresh_lo = bg_thresh_lo
        self._param_str = param_str if param_str is not None else "'num_classes': 2"

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self._param_str)
        self._num_classes = layer_params['num_classes']
        self._determininistic_mode = deterministic

        self._count = 0
        self._fg_num = 0
        self._bg_num = 0

    def infer_outputs(self):
        # sampled rois (0, x1, y1, x2, y2)
        # for CNTK the proposal shape is [4 x roisPerImage], and mirrored in Python
        rois_shape = (FreeDimension, 4)
        labels_shape = (FreeDimension, self._num_classes)
        bbox_targets_shape = (FreeDimension, self._num_classes * 4)
        bbox_inside_weights_shape = (FreeDimension, self._num_classes * 4)

        return [output_variable(rois_shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes,
                                name="rpn_target_rois_raw", needs_gradient=False),
                output_variable(labels_shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes,
                                name="label_targets_raw", needs_gradient=False),
                output_variable(bbox_targets_shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes,
                                name="bbox_targets_raw", needs_gradient=False),
                output_variable(bbox_inside_weights_shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes,
                                name="bbox_inside_w_raw", needs_gradient=False)]

    def forward(self, arguments, outputs, device=None, outputs_to_retain=None):
        bottom = arguments

        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0][0,:]
        # remove zero padded proposals
        keep0 = np.where(
            ((all_rois[:, 2] - all_rois[:, 0]) > 0) &
            ((all_rois[:, 3] - all_rois[:, 1]) > 0)
        )
        all_rois = all_rois[keep0]

        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[1][0,:]
        # remove zero padded ground truth boxes
        keep1 = np.where(
            ((gt_boxes[:,2] - gt_boxes[:,0]) > 0) &
            ((gt_boxes[:,3] - gt_boxes[:,1]) > 0)
        )
        gt_boxes = gt_boxes[keep1]

        assert gt_boxes.shape[0] > 0, \
            "No ground truth boxes provided"

        # Include ground-truth boxes in the set of candidate rois
        # for CNTK: add batch index axis with all zeros to both inputs
        all_rois = np.vstack((all_rois, gt_boxes[:, :-1]))
        zeros = np.zeros((all_rois.shape[0], 1), dtype=all_rois.dtype)
        all_rois = np.hstack((zeros, all_rois))

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        rois_per_image = self._batch_size
        fg_rois_per_image = np.round(self._fg_fraction * rois_per_image).astype(int)

        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rois, bbox_targets, bbox_inside_weights = self._sample_rois(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes,
            deterministic=self._determininistic_mode)

        if DEBUG:
            print ('num rois: {}'.format(rois_per_image))
            print ('num fg: {}'.format((labels > 0).sum()))
            print ('num bg: {}'.format((labels == 0).sum()))
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print ('num fg avg: {}'.format(self._fg_num / self._count))
            print ('num bg avg: {}'.format(self._bg_num / self._count))
            print ('ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num)))

        # pad with zeros if too few rois were found
        num_found_rois = rois.shape[0]
        if num_found_rois < rois_per_image:
            rois_padded = np.zeros((rois_per_image, rois.shape[1]), dtype=np.float32)
            rois_padded[:num_found_rois, :] = rois
            rois = rois_padded

            labels_padded = np.zeros((rois_per_image), dtype=np.float32)
            labels_padded[:num_found_rois] = labels
            labels = labels_padded

            bbox_targets_padded = np.zeros((rois_per_image, bbox_targets.shape[1]), dtype=np.float32)
            bbox_targets_padded[:num_found_rois, :] = bbox_targets
            bbox_targets = bbox_targets_padded

            bbox_inside_weights_padded = np.zeros((rois_per_image, bbox_inside_weights.shape[1]), dtype=np.float32)
            bbox_inside_weights_padded[:num_found_rois, :] = bbox_inside_weights
            bbox_inside_weights = bbox_inside_weights_padded

        # for CNTK: get rid of batch ind zeros and add batch axis
        rois = rois[:,1:]

        # sampled rois
        rois.shape = (1,) + rois.shape
        outputs[self.outputs[0]] = np.ascontiguousarray(rois)

        # classification labels
        labels_as_int = [i.item() for i in labels.astype(int)]
        labels_dense = np.eye(self._num_classes, dtype=np.float32)[labels_as_int]
        labels_dense.shape = (1,) + labels_dense.shape # batch axis
        outputs[self.outputs[1]] = labels_dense

        # bbox_targets
        bbox_targets.shape = (1,) + bbox_targets.shape # batch axis
        outputs[self.outputs[2]] = np.ascontiguousarray(bbox_targets)

        # bbox_inside_weights
        bbox_inside_weights.shape = (1,) + bbox_inside_weights.shape # batch axis
        outputs[self.outputs[3]] = np.ascontiguousarray(bbox_inside_weights)

    def backward(self, state, root_gradients, variables):
        """This layer does not propagate gradients."""
        pass

    def clone(self, cloned_inputs):
        return ProposalTargetLayer(cloned_inputs[0], cloned_inputs[1],
                                   batch_size=self._batch_size,
                                   fg_fraction=self._fg_fraction,
                                   normalize_targets=self._normalize_targets,
                                   normalize_means=self._normalize_means,
                                   normalize_stds=self._normalize_stds,
                                   fg_thresh=self._fg_thresh,
                                   bg_thresh_hi=self._bg_thresh_hi,
                                   bg_thresh_lo=self._bg_thresh_lo,
                                   param_str=self._param_str)

    def serialize(self):
        internal_state = {}
        internal_state['param_str'] = self._param_str
        internal_state['batch_size'] = self._batch_size
        internal_state['fg_fraction'] = self._fg_fraction
        internal_state['normalize_targets'] = self._normalize_targets
        internal_state['normalize_means'] = self._normalize_means
        internal_state['normalize_stds'] = self._normalize_stds
        internal_state['fg_thresh'] = self._fg_thresh
        internal_state['bg_thresh_hi'] = self._bg_thresh_hi
        internal_state['bg_thresh_lo'] = self._bg_thresh_lo
        return internal_state

    @staticmethod
    def deserialize(inputs, name, state):
        return ProposalTargetLayer(inputs[0], inputs[1],
                                   batch_size=state['batch_size'],
                                   fg_fraction=state['fg_fraction'],
                                   normalize_targets=state['normalize_targets'],
                                   normalize_means=state['normalize_means'],
                                   normalize_stds=state['normalize_stds'],
                                   fg_thresh=state['fg_thresh'],
                                   bg_thresh_hi=state['bg_thresh_hi'],
                                   bg_thresh_lo=state['bg_thresh_lo'],
                                   param_str=state['param_str'],
                                   name=name)

    def _get_bbox_regression_labels(self, bbox_target_data, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): N x 4K blob of regression targets
            bbox_inside_weights (ndarray): N x 4K blob of loss weights
        """

        clss = bbox_target_data[:, 0].astype(int)
        bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
        bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
        inds = np.where(clss > 0)[0]
        for ind in inds:
            cls = clss[ind]
            start = 4 * cls
            end = start + 4
            bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
            bbox_inside_weights[ind, start:end] = [1.0, 1.0, 1.0, 1.0]
        return bbox_targets, bbox_inside_weights


    def _compute_targets(self, ex_rois, gt_rois, labels):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.shape[0] == gt_rois.shape[0]
        assert ex_rois.shape[1] == 4
        assert gt_rois.shape[1] == 4

        targets = bbox_transform(ex_rois, gt_rois)
        if self._normalize_targets:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - np.array(self._normalize_means))
                    / np.array(self._normalize_stds))

        return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

    def _sample_rois(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes, deterministic=False):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)
        labels = gt_boxes[gt_assignment, 4]

        # Select foreground RoIs as those with >= FG_THRESH overlap
        fg_inds = np.where(max_overlaps >= self._fg_thresh)[0]
        # Guard against the case when an image has fewer than fg_rois_per_image
        # foreground RoIs
        fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)

        # Sample foreground regions without replacement
        if fg_inds.size > 0:
            if deterministic:
                fg_inds = fg_inds[:fg_rois_per_this_image]
            else:
                fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((max_overlaps < self._bg_thresh_hi) &
                           (max_overlaps >= self._bg_thresh_lo))[0]
        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
        # Sample background regions without replacement
        if bg_inds.size > 0:
            if deterministic:
                bg_inds = bg_inds[:bg_rois_per_this_image]
            else:
                bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

        # The indices that we're selecting (both fg and bg)
        keep_inds = np.append(fg_inds, bg_inds)
        # Select sampled values from various arrays:
        labels = labels[keep_inds]
        # Clamp labels for the background RoIs to 0
        labels[fg_rois_per_this_image:] = 0
        rois = all_rois[keep_inds]

        bbox_target_data = self._compute_targets(
            rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

        bbox_targets, bbox_inside_weights = \
            self._get_bbox_regression_labels(bbox_target_data, num_classes)

        return labels, rois, bbox_targets, bbox_inside_weights

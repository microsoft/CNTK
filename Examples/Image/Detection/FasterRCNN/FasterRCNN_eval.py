# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import numpy as np
from matplotlib.pyplot import imsave
import cv2
import cntk
from cntk import input_variable, Axis
from utils.nms_wrapper import apply_nms_to_single_image_results
from utils.map_helpers import evaluate_detections
from utils.plot_helpers import load_resize_and_pad, resize_and_pad, visualize_detections
from utils.rpn.bbox_transform import regress_rois
from utils.od_mb_source import ObjectDetectionMinibatchSource
from utils.proposal_helpers import ProposalProvider, compute_proposals, compute_image_stats

class FastRCNN_Evaluator:
    def __init__(self, eval_model, cfg):
        # load model once in constructor and push images through the model in 'process_image()'
        self._img_shape = (cfg.NUM_CHANNELS, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)
        image_input = input_variable(shape=self._img_shape,
                                     dynamic_axes=[Axis.default_batch_axis()],
                                     name=cfg["MODEL"].FEATURE_NODE_NAME)
        roi_proposals = input_variable((cfg.NUM_ROI_PROPOSALS, 4), dynamic_axes=[Axis.default_batch_axis()],
                                       name="roi_proposals")
        self._eval_model = eval_model(image_input, roi_proposals)
        self._min_w = cfg['PROPOSALS_MIN_W']
        self._min_h = cfg['PROPOSALS_MIN_H']
        self._num_proposals = cfg['NUM_ROI_PROPOSALS']

    def process_image(self, img_path):
        out_cls_pred, out_rpn_rois, out_bbox_regr, dims = self.process_image_detailed(img_path)
        labels = out_cls_pred.argmax(axis=1)
        regressed_rois = regress_rois(out_rpn_rois, out_bbox_regr, labels, dims)

        return regressed_rois, out_cls_pred

    def process_image_detailed(self, img_path):
        img = cv2.imread(img_path)
        _, cntk_img_input, dims = resize_and_pad(img, self._img_shape[2], self._img_shape[1])

        #import pdb; pdb.set_trace()

        # compute ROI proposals and apply scaling and padding to them
        # [target_w, target_h, img_width, img_height, top, bottom, left, right, scale_factor]
        img_stats = compute_image_stats(len(img[0]), len(img), self._img_shape[2], self._img_shape[1])
        scale_factor = img_stats[-1]
        top = img_stats[4]
        left = img_stats[6]

        proposals = compute_proposals(img, self._num_proposals, self._min_w, self._min_h)
        proposals = proposals * scale_factor
        proposals += (left, top, left, top)

        output = self._eval_model.eval({self._eval_model.arguments[0]: [cntk_img_input],
                                        self._eval_model.arguments[1]: np.array(proposals, dtype=np.float32)})

        out_dict = dict([(k.name, k) for k in output])
        out_cls_pred = output[out_dict['cls_pred']][0]
        out_rpn_rois = proposals
        out_bbox_regr = output[out_dict['bbox_regr']][0]

        return out_cls_pred, out_rpn_rois, out_bbox_regr, dims

class FasterRCNN_Evaluator:
    def __init__(self, eval_model, cfg):
        # load model once in constructor and push images through the model in 'process_image()'
        self._img_shape = (cfg.NUM_CHANNELS, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)
        image_input = input_variable(shape=self._img_shape,
                                     dynamic_axes=[Axis.default_batch_axis()],
                                     name=cfg["MODEL"].FEATURE_NODE_NAME)
        dims_input = input_variable((1,6), dynamic_axes=[Axis.default_batch_axis()], name='dims_input')
        self._eval_model = eval_model(image_input, dims_input)

    def process_image(self, img_path):
        out_cls_pred, out_rpn_rois, out_bbox_regr, dims = self.process_image_detailed(img_path)
        labels = out_cls_pred.argmax(axis=1)
        regressed_rois = regress_rois(out_rpn_rois, out_bbox_regr, labels, dims)

        return regressed_rois, out_cls_pred

    def process_image_detailed(self, img_path):
        _, cntk_img_input, dims = load_resize_and_pad(img_path, self._img_shape[2], self._img_shape[1])

        cntk_dims_input = np.array(dims, dtype=np.float32)
        cntk_dims_input.shape = (1,) + cntk_dims_input.shape
        output = self._eval_model.eval({self._eval_model.arguments[0]: [cntk_img_input],
                                        self._eval_model.arguments[1]: cntk_dims_input})

        out_dict = dict([(k.name, k) for k in output])
        out_cls_pred = output[out_dict['cls_pred']][0]
        out_rpn_rois = output[out_dict['rpn_rois']][0]
        out_bbox_regr = output[out_dict['bbox_regr']][0]

        return out_cls_pred, out_rpn_rois, out_bbox_regr, dims

def compute_test_set_aps(eval_model, cfg):
    num_test_images = cfg["DATA"].NUM_TEST_IMAGES
    classes = cfg["DATA"].CLASSES
    image_input = input_variable(shape=(cfg.NUM_CHANNELS, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
                                 dynamic_axes=[Axis.default_batch_axis()],
                                 name=cfg["MODEL"].FEATURE_NODE_NAME)
    roi_input = input_variable((cfg.INPUT_ROIS_PER_IMAGE, 5), dynamic_axes=[Axis.default_batch_axis()])
    dims_input = input_variable((6), dynamic_axes=[Axis.default_batch_axis()])
    frcn_eval = eval_model(image_input, dims_input)

    # Create the minibatch source
    minibatch_source = ObjectDetectionMinibatchSource(
        cfg["DATA"].TEST_MAP_FILE,
        cfg["DATA"].TEST_ROI_FILE,
        max_annotations_per_image=cfg.INPUT_ROIS_PER_IMAGE,
        pad_width=cfg.IMAGE_WIDTH,
        pad_height=cfg.IMAGE_HEIGHT,
        pad_value=cfg["MODEL"].IMG_PAD_COLOR,
        randomize=False, use_flipping=False,
        max_images=cfg["DATA"].NUM_TEST_IMAGES,
        num_classes=cfg["DATA"].NUM_CLASSES,
        proposal_provider=None)

    # define mapping from reader streams to network inputs
    input_map = {
        minibatch_source.image_si: image_input,
        minibatch_source.roi_si: roi_input,
        minibatch_source.dims_si: dims_input
    }

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_test_images)] for _ in range(cfg["DATA"].NUM_CLASSES)]

    # evaluate test images and write netwrok output to file
    print("Evaluating Faster R-CNN model for %s images." % num_test_images)
    all_gt_infos = {key: [] for key in classes}
    for img_i in range(0, num_test_images):
        mb_data = minibatch_source.next_minibatch(1, input_map=input_map)

        gt_row = mb_data[roi_input].asarray()
        gt_row = gt_row.reshape((cfg.INPUT_ROIS_PER_IMAGE, 5))
        all_gt_boxes = gt_row[np.where(gt_row[:,-1] > 0)]

        for cls_index, cls_name in enumerate(classes):
            if cls_index == 0: continue
            cls_gt_boxes = all_gt_boxes[np.where(all_gt_boxes[:,-1] == cls_index)]
            all_gt_infos[cls_name].append({'bbox': np.array(cls_gt_boxes),
                                           'difficult': [False] * len(cls_gt_boxes),
                                           'det': [False] * len(cls_gt_boxes)})

        output = frcn_eval.eval({image_input: mb_data[image_input], dims_input: mb_data[dims_input]})
        out_dict = dict([(k.name, k) for k in output])
        out_cls_pred = output[out_dict['cls_pred']][0]
        out_rpn_rois = output[out_dict['rpn_rois']][0]
        out_bbox_regr = output[out_dict['bbox_regr']][0]

        labels = out_cls_pred.argmax(axis=1)
        scores = out_cls_pred.max(axis=1)
        regressed_rois = regress_rois(out_rpn_rois, out_bbox_regr, labels, mb_data[dims_input].asarray())

        labels.shape = labels.shape + (1,)
        scores.shape = scores.shape + (1,)
        coords_score_label = np.hstack((regressed_rois, scores, labels))

        #   shape of all_boxes: e.g. 21 classes x 4952 images x 58 rois x 5 coords+score
        for cls_j in range(1, cfg["DATA"].NUM_CLASSES):
            coords_score_label_for_cls = coords_score_label[np.where(coords_score_label[:,-1] == cls_j)]
            all_boxes[cls_j][img_i] = coords_score_label_for_cls[:,:-1].astype(np.float32, copy=False)

        if (img_i+1) % 100 == 0:
            print("Processed {} samples".format(img_i+1))

    # calculate mAP
    aps = evaluate_detections(all_boxes, all_gt_infos, classes,
                              use_gpu_nms = cfg.USE_GPU_NMS,
                              device_id = cfg.GPU_ID,
                              nms_threshold=cfg.RESULTS_NMS_THRESHOLD,
                              conf_threshold = cfg.RESULTS_NMS_CONF_THRESHOLD)

    return aps

def plot_test_set_results(evaluator, num_images_to_plot, results_base_path, cfg):
    # get image paths
    with open(cfg["DATA"].TEST_MAP_FILE) as f:
        content = f.readlines()
    img_base_path = os.path.dirname(os.path.abspath(cfg["DATA"].TEST_MAP_FILE))
    img_file_names = [os.path.join(img_base_path, x.split('\t')[1]) for x in content]
    img_shape = (cfg.NUM_CHANNELS, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)

    print("Plotting results from Faster R-CNN model for %s images." % num_images_to_plot)
    for i in range(0, num_images_to_plot):
        img_path = img_file_names[i]
        out_cls_pred, out_rpn_rois, out_bbox_regr, dims = evaluator.process_image_detailed(img_path)
        labels = out_cls_pred.argmax(axis=1)
        scores = out_cls_pred.max(axis=1)

        if cfg.DRAW_UNREGRESSED_ROIS:
            # plot results without final regression
            imgDebug = visualize_detections(img_path, out_rpn_rois, labels, scores,
                                            img_shape[2], img_shape[1],
                                            classes=cfg["DATA"].CLASSES,
                                            draw_negative_rois=cfg.DRAW_NEGATIVE_ROIS,
                                            decision_threshold=cfg.RESULTS_BGR_PLOT_THRESHOLD)
            imsave("{}/{}_{}".format(results_base_path, i, os.path.basename(img_path)), imgDebug)

        # apply regression and nms to bbox coordinates
        regressed_rois = regress_rois(out_rpn_rois, out_bbox_regr, labels, dims)
        nmsKeepIndices = apply_nms_to_single_image_results(regressed_rois, labels, scores,
                                                           use_gpu_nms=cfg.USE_GPU_NMS,
                                                           device_id=cfg.GPU_ID,
                                                           nms_threshold=cfg.RESULTS_NMS_THRESHOLD,
                                                           conf_threshold=cfg.RESULTS_NMS_CONF_THRESHOLD)

        filtered_bboxes = regressed_rois[nmsKeepIndices]
        filtered_labels = labels[nmsKeepIndices]
        filtered_scores = scores[nmsKeepIndices]

        img = visualize_detections(img_path, filtered_bboxes, filtered_labels, filtered_scores,
                                   img_shape[2], img_shape[1],
                                   classes=cfg["DATA"].CLASSES,
                                   draw_negative_rois=cfg.DRAW_NEGATIVE_ROIS,
                                   decision_threshold=cfg.RESULTS_BGR_PLOT_THRESHOLD)
        imsave("{}/{}_regr_{}".format(results_base_path, i, os.path.basename(img_path)), img)

def compute_test_set_aps_fast_rcnn(eval_model, cfg):
    num_test_images = cfg["DATA"].NUM_TEST_IMAGES
    classes = cfg["DATA"].CLASSES
    image_input = input_variable(shape=(cfg.NUM_CHANNELS, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
                                 dynamic_axes=[Axis.default_batch_axis()],
                                 name=cfg["MODEL"].FEATURE_NODE_NAME)
    roi_input = input_variable((cfg.INPUT_ROIS_PER_IMAGE, 5), dynamic_axes=[Axis.default_batch_axis()])
    roi_proposals = input_variable((cfg.NUM_ROI_PROPOSALS, 4), dynamic_axes=[Axis.default_batch_axis()], name="roi_proposals")
    dims_input = input_variable((6), dynamic_axes=[Axis.default_batch_axis()])
    frcn_eval = eval_model(image_input, roi_proposals)

    # Create the minibatch source
    proposal_provider = ProposalProvider.fromconfig(cfg)
    minibatch_source = ObjectDetectionMinibatchSource(
        cfg["DATA"].TEST_MAP_FILE,
        cfg["DATA"].TEST_ROI_FILE,
        max_annotations_per_image=cfg.INPUT_ROIS_PER_IMAGE,
        pad_width=cfg.IMAGE_WIDTH,
        pad_height=cfg.IMAGE_HEIGHT,
        pad_value=cfg["MODEL"].IMG_PAD_COLOR,
        randomize=False, use_flipping=False,
        max_images=cfg["DATA"].NUM_TEST_IMAGES,
        num_classes=cfg["DATA"].NUM_CLASSES,
        proposal_provider=proposal_provider,
        provide_targets=False)

    # define mapping from reader streams to network inputs
    input_map = {
        minibatch_source.image_si: image_input,
        minibatch_source.roi_si: roi_input,
        minibatch_source.proposals_si: roi_proposals,
        minibatch_source.dims_si: dims_input
    }

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_test_images)] for _ in range(cfg["DATA"].NUM_CLASSES)]

    # evaluate test images and write netwrok output to file
    print("Evaluating Fast R-CNN model for %s images." % num_test_images)
    all_gt_infos = {key: [] for key in classes}
    for img_i in range(0, num_test_images):
        mb_data = minibatch_source.next_minibatch(1, input_map=input_map)

        gt_row = mb_data[roi_input].asarray()
        gt_row = gt_row.reshape((cfg.INPUT_ROIS_PER_IMAGE, 5))
        all_gt_boxes = gt_row[np.where(gt_row[:,-1] > 0)]

        for cls_index, cls_name in enumerate(classes):
            if cls_index == 0: continue
            cls_gt_boxes = all_gt_boxes[np.where(all_gt_boxes[:,-1] == cls_index)]
            all_gt_infos[cls_name].append({'bbox': np.array(cls_gt_boxes),
                                           'difficult': [False] * len(cls_gt_boxes),
                                           'det': [False] * len(cls_gt_boxes)})

        output = frcn_eval.eval({image_input: mb_data[image_input], roi_proposals: mb_data[roi_proposals]})
        out_dict = dict([(k.name, k) for k in output])
        out_cls_pred = output[out_dict['cls_pred']][0]
        out_rpn_rois = mb_data[roi_proposals].data.asarray()
        out_bbox_regr = output[out_dict['bbox_regr']][0]

        labels = out_cls_pred.argmax(axis=1)
        scores = out_cls_pred.max(axis=1)
        regressed_rois = regress_rois(out_rpn_rois, out_bbox_regr, labels, mb_data[dims_input].asarray())

        labels.shape = labels.shape + (1,)
        scores.shape = scores.shape + (1,)
        coords_score_label = np.hstack((regressed_rois, scores, labels))

        #   shape of all_boxes: e.g. 21 classes x 4952 images x 58 rois x 5 coords+score
        for cls_j in range(1, cfg["DATA"].NUM_CLASSES):
            coords_score_label_for_cls = coords_score_label[np.where(coords_score_label[:,-1] == cls_j)]
            all_boxes[cls_j][img_i] = coords_score_label_for_cls[:,:-1].astype(np.float32, copy=False)

        if (img_i+1) % 100 == 0:
            print("Processed {} samples".format(img_i+1))

    # calculate mAP
    aps = evaluate_detections(all_boxes, all_gt_infos, classes,
                              use_gpu_nms = cfg.USE_GPU_NMS,
                              device_id = cfg.GPU_ID,
                              nms_threshold=cfg.RESULTS_NMS_THRESHOLD,
                              conf_threshold = cfg.RESULTS_NMS_CONF_THRESHOLD)

    return aps


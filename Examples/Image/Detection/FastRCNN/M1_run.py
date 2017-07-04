# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from A2_RunWithPyModel import create_mb_source, train_fast_rcnn, base_path
import os
from cntk import *

import hierarchical_classification_tool as HCT

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, ".."))
from utils.map.map_helpers import evaluate_detections


def eval_fast_rcnn_mAP(eval_model, img_map_file=None, roi_map_file=None):
    output_mapper = HCT.tree_map.get_output_mapper()
    classes = output_mapper.get_all_classes()
    num_test_images = 5
    num_classes = len(classes)
    num_original_classes=17
    num_channels=3
    image_height=1000
    image_width=1000
    feature_node_name='data'
    rois_per_image=2000

    image_input = input_variable((num_channels, image_height, image_width), dynamic_axes=[Axis.default_batch_axis()], name=feature_node_name)
    #roi_input = input_variable((rois_per_image, 5), dynamic_axes=[Axis.default_batch_axis()])
    roi_input = input_variable((rois_per_image, 4), dynamic_axes=[Axis.default_batch_axis()])
    #dims_input = input_variable((6), dynamic_axes=[Axis.default_batch_axis()])
    label_input = input_variable((rois_per_image, num_classes))
    frcn_eval = eval_model(image_input, roi_input)

    if False:
        # Create the minibatch source
        minibatch_source = ObjectDetectionMinibatchSource(
            img_map_file, roi_map_file,
            max_annotations_per_image=cfg["CNTK"].INPUT_ROIS_PER_IMAGE,
            pad_width=image_width, pad_height=image_height, pad_value=img_pad_value,
            randomize=False, use_flipping=False,
            max_images=cfg["CNTK"].NUM_TEST_IMAGES)

        # define mapping from reader streams to network inputs
        input_map = {
            minibatch_source.image_si: image_input,
            minibatch_source.roi_si: roi_input,
            minibatch_source.dims_si: dims_input
        }
    else:
        data_path=base_path
        data_set="test"
        minibatch_source = create_mb_source(image_height, image_width, num_channels,num_original_classes, rois_per_image, data_path, data_set)
        input_map = {
            minibatch_source.streams.features: image_input,
            minibatch_source.streams.rois: roi_input,
            minibatch_source.streams.roiLabels: label_input
        }

    #img_key = cfg["CNTK"].FEATURE_NODE_NAME
    #roi_key = "x 5]"
    #dims_key = "[6]"


    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)

    all_boxes = [[[] for _ in range(num_test_images)] for _ in range(num_classes)]

    # evaluate test images and write netwrok output to file
    print("Evaluating Faster R-CNN model for %s images." % num_test_images)
    print(type(classes))
    all_gt_infos = {key: [] for key in classes}
    for img_i in range(0, num_test_images):
        #import ipdb;ipdb.set_trace()
        mb_data = minibatch_source.next_minibatch(1, input_map=input_map)

        rkeys = [k for k in mb_data if roi_key in str(k)]
        gt_row = mb_data[rkeys[0]].asarray()
        gt_row = gt_row.reshape((cfg["CNTK"].INPUT_ROIS_PER_IMAGE, 5))
        all_gt_boxes = gt_row[np.where(gt_row[:,-1] > 0)]

        for cls_index, cls_name in enumerate(classes):
            if cls_index == 0: continue
            #   gtBoxes = [box for box, label in zip(gtBoxes, gtLabels) if
            #              label.decode('utf-8') == self.classes[classIndex]]
            cls_gt_boxes = all_gt_boxes[np.where(all_gt_boxes[:,-1] == cls_index)]
            #   gtInfos.append({'bbox': np.array(gtBoxes),
            #                   'difficult': [False] * len(gtBoxes),
            #                   'det': [False] * len(gtBoxes)})
            all_gt_infos[cls_name].append({'bbox': np.array(cls_gt_boxes),
                                           'difficult': [False] * len(cls_gt_boxes),
                                           'det': [False] * len(cls_gt_boxes)})

        fkeys = [k for k in mb_data if img_key in str(k)]
        dkeys = [k for k in mb_data if dims_key in str(k)]

        output = frcn_eval.eval({fkeys[0]: mb_data[fkeys[0]], dkeys[0]: mb_data[dkeys[0]]})
        out_dict = dict([(k.name, k) for k in output])
        out_cls_pred = output[out_dict['cls_pred']][0]                      # (300, 17)
        out_rpn_rois = output[out_dict['rpn_rois']][0]
        out_bbox_regr = output[out_dict['bbox_regr']][0]

        labels = out_cls_pred.argmax(axis=1)
        scores = out_cls_pred.max(axis=1)
        regressed_rois = regress_rois(out_rpn_rois, out_bbox_regr, labels)  # (300, 4)

        labels.shape = labels.shape + (1,)
        scores.shape = scores.shape + (1,)
        coords_score_label = np.hstack((regressed_rois, scores, labels))

        #   shape of all_boxes: e.g. 21 classes x 4952 images x 58 rois x 5 coords+score
        for cls_j in range(1, num_classes):
            coords_score_label_for_cls = coords_score_label[np.where(coords_score_label[:,-1] == cls_j)]
            all_boxes[cls_j][img_i] = coords_score_label_for_cls[:,:-1].astype(np.float32, copy=False)

        if (img_i+1) % 100 == 0:
            print("Processed {} samples".format(img_i+1))

    # calculate mAP
    aps = evaluate_detections(all_boxes, all_gt_infos, classes, apply_mms=False)
    ap_list = []
    for class_name in aps:
        ap_list += [aps[class_name]]
        print('AP for {:>15} = {:.4f}'.format(class_name, aps[class_name]))
    print('Mean AP = {:.4f}'.format(np.nanmean(ap_list)))

    return aps


if __name__ == '__main__':
    os.chdir(base_path)
    model_path = os.path.join(abs_path, "Output", "frcn_py.model")

    # Train only is no model exists yet
    if os.path.exists(model_path):
        print("Loading existing model from %s" % model_path)
        trained_model = load_model(model_path)
    else:
        trained_model = train_fast_rcnn()
        trained_model.save(model_path)
        print("Stored trained model at %s" % model_path)

    # eval trained_model
    print("\n---Evaluation---")

    output_mapper = HCT.tree_map.get_output_mapper()
    known_classes = output_mapper.get_all_classes()

    aps=eval_fast_rcnn_mAP(trained_model)


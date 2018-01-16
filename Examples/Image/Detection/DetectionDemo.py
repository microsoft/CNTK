# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os, sys
import numpy as np
import utils.od_utils as od
from utils.config_helpers import merge_configs

available_detectors = ['FastRCNN', 'FasterRCNN']

def get_detector_name(args):
    detector_name = None
    default_detector = 'FasterRCNN'
    if len(args) != 2:
        print("Please provide a detector name as the single argument. Usage:")
        print("    python DetectionDemo.py <detector_name>")
        print("Available detectors: {}".format(available_detectors))
    else:
        detector_name = args[1]
        if not any(detector_name == x for x in available_detectors):
            print("Unknown detector: {}.".format(detector_name))
            print("Available detectors: {}".format(available_detectors))
            detector_name = None

    if detector_name is None:
        print("Using default detector: {}".format(default_detector))
        return default_detector
    else:
        return detector_name

def get_configuration(detector_name):
    # load configs for detector, base network and data set
    if detector_name == "FastRCNN":
        from FastRCNN.FastRCNN_config import cfg as detector_cfg
    elif detector_name == "FasterRCNN":
        from FasterRCNN.FasterRCNN_config import cfg as detector_cfg
    else:
        print('Unknown detector: {}'.format(detector_name))

    # for VGG16 base model use:         from utils.configs.VGG16_config import cfg as network_cfg
    # for AlexNet base model use:       from utils.configs.AlexNet_config import cfg as network_cfg
    from utils.configs.AlexNet_config import cfg as network_cfg
    # for Pascal VOC 2007 data set use: from utils.configs.Pascal_config import cfg as dataset_cfg
    # for the Grocery data set use:     from utils.configs.Grocery_config import cfg as dataset_cfg
    from utils.configs.Grocery_config import cfg as dataset_cfg

    return merge_configs([detector_cfg, network_cfg, dataset_cfg, {'DETECTOR': detector_name}])

if __name__ == '__main__':
    # Currently supported detectors: 'FastRCNN', 'FasterRCNN'
    args = sys.argv
    detector_name = get_detector_name(args)
    cfg = get_configuration(detector_name)

    # train and test
    eval_model = od.train_object_detector(cfg)
    eval_results = od.evaluate_test_set(eval_model, cfg)

    # write AP results to output
    for class_name in eval_results: print('AP for {:>15} = {:.4f}'.format(class_name, eval_results[class_name]))
    print('Mean AP = {:.4f}'.format(np.nanmean(list(eval_results.values()))))

    # detect objects in single image
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), r"../DataSets/Grocery/testImages/WIN_20160803_11_28_42_Pro.jpg")
    regressed_rois, cls_probs = od.evaluate_single_image(eval_model, img_path, cfg)
    bboxes, labels, scores = od.filter_results(regressed_rois, cls_probs, cfg)

    # write detection results to output
    fg_boxes = np.where(labels > 0)
    print("#bboxes: before nms: {}, after nms: {}, foreground: {}".format(len(regressed_rois), len(bboxes), len(fg_boxes[0])))
    for i in fg_boxes[0]: print("{:<12} (label: {:<2}), score: {:.3f}, box: {}".format(
                                cfg["DATA"].CLASSES[labels[i]], labels[i], scores[i], [int(v) for v in bboxes[i]]))

    # visualize detections on image
    od.visualize_results(img_path, bboxes, labels, scores, cfg)

    # measure inference time
    od.measure_inference_time(eval_model, img_path, cfg, num_repetitions=100)

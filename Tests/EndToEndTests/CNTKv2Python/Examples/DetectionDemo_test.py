# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import pytest
import sys
from cntk import load_model
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import try_set_default_device, gpu
from cntk.logging.graph import get_node_outputs
from cntk.ops.tests.ops_test_utils import cntk_device
from _cntk_py import force_deterministic_algorithms
force_deterministic_algorithms()

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Detection"))

win35_linux34 = pytest.mark.skipif(not ((sys.platform == 'win32' and sys.version_info[:2] == (3,5)) or
                                        (sys.platform != 'win32' and sys.version_info[:2] == (3,4))),
                                   reason="it runs currently only in windows-py35 and linux-py34 due to precompiled cython modules")

@win35_linux34
def test_detection_demo(device_id):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')  # it runs very slow in CPU
    try_set_default_device(cntk_device(device_id))

    from prepare_test_data import prepare_Grocery_data, prepare_alexnet_v0_model
    grocery_path = prepare_Grocery_data()
    prepare_alexnet_v0_model()

    from FastRCNN.install_data_and_model import create_grocery_mappings
    create_grocery_mappings(grocery_path)

    from DetectionDemo import get_configuration
    import utils.od_utils as od

    cfg = get_configuration('FasterRCNN')
    cfg["CNTK"].FORCE_DETERMINISTIC = True
    cfg["CNTK"].DEBUG_OUTPUT = False
    cfg["CNTK"].MAKE_MODE = False
    cfg["CNTK"].FAST_MODE = False
    cfg.CNTK.E2E_MAX_EPOCHS = 3
    cfg.CNTK.RPN_EPOCHS = 2
    cfg.CNTK.FRCN_EPOCHS = 2
    cfg.IMAGE_WIDTH = 400
    cfg.IMAGE_HEIGHT = 400
    cfg["CNTK"].TRAIN_E2E = True
    cfg.USE_GPU_NMS = False
    cfg.VISUALIZE_RESULTS = False
    cfg["DATA"].MAP_FILE_PATH = grocery_path

    externalData = 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' in os.environ
    if externalData:
        extPath = os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY']
        cfg['BASE_MODEL_PATH'] = os.path.join(extPath, "PreTrainedModels", "AlexNet", "v1", "AlexNet_ImageNet_Caffe.model")
    else:
        cfg['BASE_MODEL_PATH'] = os.path.join(abs_path, *"../../../../PretrainedModels/AlexNet_ImageNet_Caffe.model".split("/"))

    # train and test
    eval_model = od.train_object_detector(cfg)
    eval_results = od.evaluate_test_set(eval_model, cfg)

    meanAP = np.nanmean(list(eval_results.values()))
    print('meanAP={}'.format(meanAP))
    assert meanAP > 0.01

    # detect objects in single image
    img_path = os.path.join(grocery_path, "testImages", "WIN_20160803_11_28_42_Pro.jpg")
    regressed_rois, cls_probs = od.evaluate_single_image(eval_model, img_path, cfg)
    bboxes, labels, scores = od.filter_results(regressed_rois, cls_probs, cfg)
    assert bboxes.shape[0] == labels.shape[0]

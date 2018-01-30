# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os, sys
import pytest
from cntk import load_model, cntk_py
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import try_set_default_device, gpu, cpu
from cntk.logging.graph import get_node_outputs
from cntk.ops.tests.ops_test_utils import cntk_device
from _cntk_py import force_deterministic_algorithms
force_deterministic_algorithms()

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Detection"))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Detection", "FastRCNN"))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Detection", "FasterRCNN"))

from prepare_test_data import prepare_Grocery_data, prepare_alexnet_v0_model
grocery_path = prepare_Grocery_data()
prepare_alexnet_v0_model()

from install_data_and_model import create_grocery_mappings
create_grocery_mappings(grocery_path)

from test_decorators import run_if_win35_linux35_linux36

def run_fasterrcnn_grocery_training(e2e):
    from FasterRCNN_eval import compute_test_set_aps
    from utils.config_helpers import merge_configs
    from FasterRCNN_config import cfg as detector_cfg
    from utils.configs.AlexNet_config import cfg as network_cfg
    from utils.configs.Grocery_config import cfg as dataset_cfg

    cfg = merge_configs([detector_cfg, network_cfg, dataset_cfg])
    cfg["CNTK"].FORCE_DETERMINISTIC = True
    cfg["CNTK"].DEBUG_OUTPUT = False
    cfg["CNTK"].MAKE_MODE = False
    cfg["CNTK"].FAST_MODE = False
    cfg.CNTK.E2E_MAX_EPOCHS = 3
    cfg.CNTK.RPN_EPOCHS = 2
    cfg.CNTK.FRCN_EPOCHS = 2
    cfg.IMAGE_WIDTH = 400
    cfg.IMAGE_HEIGHT = 400
    cfg["CNTK"].TRAIN_E2E = e2e
    cfg.USE_GPU_NMS = False
    cfg.VISUALIZE_RESULTS = False
    cfg["DATA"].MAP_FILE_PATH = grocery_path

    externalData = 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' in os.environ
    if externalData:
        extPath = os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY']
        model_file = os.path.join(extPath, "PreTrainedModels", "AlexNet", "v1", "AlexNet_ImageNet_Caffe.model")
    else:
        model_file = os.path.join(abs_path, *"../../../../PretrainedModels/AlexNet_ImageNet_Caffe.model".split("/"))

    from FasterRCNN_train import prepare, train_faster_rcnn

    np.random.seed(seed=3)
    prepare(cfg, False)
    cfg['BASE_MODEL_PATH'] = model_file
    trained_model = train_faster_rcnn(cfg)
    eval_results = compute_test_set_aps(trained_model, cfg)
    meanAP = np.nanmean(list(eval_results.values()))
    print('meanAP={}'.format(meanAP))
    assert meanAP > 0.01
    return trained_model, meanAP, cfg

@run_if_win35_linux35_linux36
def reenable_once_sorting_is_stable_test_native_fasterrcnn_eval(device_id):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')  # it runs very slow in CPU
    try_set_default_device(cntk_device(device_id))

    from FasterRCNN_eval import compute_test_set_aps
    eval_model, meanAP_python, cfg = run_fasterrcnn_grocery_training(True)

    cntk_py.always_allow_setting_default_device()
    try_set_default_device(cpu())

    sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Extensibility", "ProposalLayer"))
    from native_proposal_layer import clone_with_native_proposal_layer
    model_with_native_pl = clone_with_native_proposal_layer(eval_model)
    eval_results = compute_test_set_aps(model_with_native_pl, cfg)
    meanAP_native = np.nanmean(list(eval_results.values()))

    # 0.2067 (python) vs 0.2251 (native) -- the difference stems
    # from different sorting algorithms: quicksort in python and 
    # heapsort in c++ (both are not stable).
    print("Python: {}, native: {}".format(meanAP_python, meanAP_native))
    assert abs(meanAP_python - meanAP_native) < 0.1

@run_if_win35_linux35_linux36
def test_fasterrcnn_grocery_training_e2e(device_id):
    try_set_default_device(cntk_device(device_id))
    _, _, _ = run_fasterrcnn_grocery_training(e2e = True)

@run_if_win35_linux35_linux36
def test_fasterrcnn_grocery_training_4stage(device_id):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')  # it runs very slow in CPU
    try_set_default_device(cntk_device(device_id))
    _, _, _ = run_fasterrcnn_grocery_training(e2e = False)

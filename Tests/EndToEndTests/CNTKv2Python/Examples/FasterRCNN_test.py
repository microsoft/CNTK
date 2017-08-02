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
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Extensibility", "ProposalLayer"))

from prepare_test_data import prepare_Grocery_data, prepare_alexnet_v0_model
grocery_path = prepare_Grocery_data()
prepare_alexnet_v0_model()

from FastRCNN.install_data_and_model import create_grocery_mappings
create_grocery_mappings(grocery_path)
from utils.config_helpers import merge_configs

win35_linux34 = pytest.mark.skipif(not ((sys.platform == 'win32' and sys.version_info[:2] == (3,5)) or
                                        (sys.platform != 'win32' and sys.version_info[:2] == (3,4))),
                                   reason="it runs currently only in windows-py35 and linux-py34 due to precompiled cython modules")

def run_fasterrcnn_grocery_training(device_id, e2e):
    from FasterRCNN.config import cfg as detector_cfg
    from utils.configs.AlexNet_config import cfg as network_cfg
    from utils.configs.Grocery_config import cfg as dataset_cfg

    cfg = merge_configs([detector_cfg, network_cfg, dataset_cfg])
    cfg["CNTK"].FORCE_DETERMINISTIC = True
    cfg["CNTK"].DEBUG_OUTPUT = False
    cfg["CNTK"].MAKE_MODE = False
    cfg["CNTK"].FAST_MODE = True
    cfg["CNTK"].TRAIN_E2E = e2e
    cfg.USE_GPU_NMS = True
    cfg.VISUALIZE_RESULTS = False
    cfg["DATA"].MAP_FILE_PATH = grocery_path

    externalData = 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' in os.environ
    if externalData:
        extPath = os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY']
        cfg['BASE_MODEL_PATH'] = os.path.join(extPath, "PreTrainedModels", "AlexNet", "v0", "AlexNet.model")
    else:
        model_file = os.path.join(abs_path, *"../../../../PretrainedModels/AlexNet.model".split("/"))

    from FasterRCNN import train_faster_rcnn_e2e, eval_faster_rcnn_mAP

    np.random.seed(seed=3)
    eval_model = train_faster_rcnn_e2e(model_file, debug_output=False)
    meanAP = eval_faster_rcnn_mAP(eval_model)
    assert meanAP > 0.01

@win35_linux34
def test_native_fasterrcnn_eval(tmpdir, device_id):
    from config import cfg
    cfg["CNTK"].FORCE_DETERMINISTIC = True
    cfg["CNTK"].DEBUG_OUTPUT = False
    cfg["CNTK"].VISUALIZE_RESULTS = False
    cfg["CNTK"].FAST_MODE = True
    cfg["CNTK"].MAP_FILE_PATH = grocery_path

    from FasterRCNN import set_global_vars
    set_global_vars(False)

    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')  # it runs very slow in CPU
    try_set_default_device(cntk_device(device_id))

    # since we do not use a reader for evaluation we need unzipped data
    externalData = 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' in os.environ

    if externalData:
        extPath = os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY']
        model_file = os.path.join(extPath, "PreTrainedModels", "AlexNet", "v0", "AlexNet.model")
    else:
        model_file = os.path.join(abs_path, *"../../../../PretrainedModels/AlexNet.model".split("/"))

    from FasterRCNN import train_faster_rcnn_e2e, eval_faster_rcnn_mAP

    np.random.seed(seed=3)

    eval_model = train_faster_rcnn_e2e(model_file, debug_output=False)

    meanAP_python = eval_faster_rcnn_mAP(eval_model)

    cntk_py.always_allow_setting_default_device()
    
    try_set_default_device(cpu())

    from native_proposal_layer import clone_with_native_proposal_layer
    
    model_with_native_pl = clone_with_native_proposal_layer(eval_model)
    meanAP_native = eval_faster_rcnn_mAP(model_with_native_pl)

    # 0.2067 (python) vs 0.2251 (native) -- the difference stems
    # from different sorting algorithms: quicksort in python and 
    # heapsort in c++ (both are not stable).
    assert abs(meanAP_python - meanAP_native) < 0.1

@win35_linux34
def test_fasterrcnn_grocery_training_4stage(device_id):
    from config import cfg
    cfg["CNTK"].FORCE_DETERMINISTIC = True
    cfg["CNTK"].DEBUG_OUTPUT = False
    cfg["CNTK"].VISUALIZE_RESULTS = False
    cfg["CNTK"].FAST_MODE = True
    cfg["CNTK"].MAP_FILE_PATH = grocery_path

    from FasterRCNN.FasterRCNN_train import prepare, train_faster_rcnn
    from FasterRCNN.FasterRCNN_eval import compute_test_set_aps
    prepare(cfg, False)

    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')  # it runs very slow in CPU
    try_set_default_device(cntk_device(device_id))

    np.random.seed(seed=3)
    trained_model = train_faster_rcnn(cfg)
    eval_results = compute_test_set_aps(trained_model, cfg)
    meanAP = np.nanmean(list(eval_results.values()))
    print('meanAP={}'.format(meanAP))
    assert meanAP > 0.01

@win35_linux34
def test_fasterrcnn_grocery_training_4stage(device_id):
    run_fasterrcnn_grocery_training(device_id, e2e = False)

@win35_linux34
def test_fasterrcnn_grocery_training_e2e(device_id, e2e=True):
    run_fasterrcnn_grocery_training(device_id, e2e = True)

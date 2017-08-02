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

from prepare_test_data import prepare_Grocery_data, prepare_alexnet_v0_model
grocery_path = prepare_Grocery_data()
prepare_alexnet_v0_model()

from FastRCNN.install_data_and_model import create_grocery_mappings
create_grocery_mappings(grocery_path)
from utils.config_helpers import merge_configs

win35_linux34 = pytest.mark.skipif(not ((sys.platform == 'win32' and sys.version_info[:2] == (3,5)) or
                                        (sys.platform != 'win32' and sys.version_info[:2] == (3,4))),
                                   reason="it runs currently only in windows-py35 and linux-py34 due to precompiled cython modules")

@win35_linux34
def test_fastrcnnpy_grocery_training(device_id):
    from FastRCNN.config import cfg as detector_cfg
    from utils.configs.AlexNet_config import cfg as network_cfg
    from utils.configs.Grocery_config import cfg as dataset_cfg

    cfg = merge_configs([detector_cfg, network_cfg, dataset_cfg])
    cfg["CNTK"].FORCE_DETERMINISTIC = True
    cfg["CNTK"].DEBUG_OUTPUT = False
    cfg["CNTK"].MAKE_MODE = False
    cfg["CNTK"].FAST_MODE = False
    cfg["CNTK"].MAX_EPOCHS = 2
    cfg.NUM_ROI_PROPOSALS = 100
    cfg.USE_GPU_NMS = True
    cfg.VISUALIZE_RESULTS = False
    cfg["DATA"].MAP_FILE_PATH = grocery_path

    externalData = 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' in os.environ
    if externalData:
        extPath = os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY']
        cfg['BASE_MODEL_PATH'] = os.path.join(extPath, "PreTrainedModels", "AlexNet", "v0", "AlexNet.model")
    else:
        cfg['BASE_MODEL_PATH'] = os.path.join(abs_path, *"../../../../PretrainedModels/AlexNet.model".split("/"))

    from FastRCNN.FastRCNN_train import prepare, train_fast_rcnn
    from FastRCNN.FastRCNN_eval import compute_test_set_aps
    prepare(cfg, False)

    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')  # it runs very slow in CPU
    try_set_default_device(cntk_device(device_id))

    np.random.seed(seed=3)
    trained_model = train_fast_rcnn(cfg)
    eval_results = compute_test_set_aps(trained_model, cfg)
    meanAP = np.nanmean(list(eval_results.values()))
    print('meanAP={}'.format(meanAP))
    assert meanAP > 0.01

# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
from cntk.ops.tests.ops_test_utils import cntk_device
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import set_default_device
import pytest

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "FeatureExtraction"))
from FeatureExtraction import create_mb_source, eval_and_write

TOLERANCE_ABSOLUTE = 2E-2

def disabled_fix_data_set_zip___test_feature_extraction(device_id):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU') # due to batch normalization in ResNet_18
    set_default_device(cntk_device(device_id))

    base_path = os.path.dirname(os.path.abspath(__file__))
    externalData = 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' in os.environ
    if externalData:
        extPath = os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY']
        print("Reading data and model from %s" % extPath)
        model_file = os.path.join(extPath, *"PreTrainedModels/ResNet/v1/ResNet_18.model".split("/"))
        map_file = os.path.join(extPath, *"Image/CIFAR/v0/cifar-10-batches-py/test_map.txt".split("/"))
    else:
        model_file = os.path.join(base_path, *"../../../../Examples/Image/PretrainedModels/ResNet_18.model".split("/"))
        map_file = os.path.join(base_path, *"../../../../Examples/Image/DataSets/CIFAR-10/test_map.txt".split("/"))

    os.chdir(os.path.join(os.path.dirname(map_file), '..'))

    minibatch_source = create_mb_source(224, 224, 3, map_file)
    node_name = "z.x"
    output_file = os.path.join(base_path, "layerOutput.txt")
    eval_and_write(model_file, node_name, output_file, minibatch_source, num_objects=2)

    expected_output_file = os.path.join(base_path, "feature_extraction_expected_output.txt")
    output = np.fromfile(output_file)
    expected_output = np.fromfile(expected_output_file)

    print(output.shape)
    print(expected_output.shape)
    assert np.allclose(output, expected_output, atol=TOLERANCE_ABSOLUTE)

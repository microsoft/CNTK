# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
from cntk.ops.tests.ops_test_utils import cntk_device
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import set_default_device, gpu
import pytest

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "TransferLearning"))
from TransferLearning import train_model, eval_test_images

TOLERANCE_ABSOLUTE = 2E-2

def test_transfer_learning(device_id):
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

    feature_node_name = "features"
    last_hidden_node_name = "z.x"
    image_width = 224
    image_height = 224
    num_channels = 3
    num_classes = 10

    num_train_images = 10
    num_test_images = 2

    output_file = os.path.join(base_path, "tl_output.txt")
    trained_model = train_model(model_file, feature_node_name, last_hidden_node_name,
                                image_width, image_height, num_channels, num_classes, map_file,
                                max_images=num_train_images)
    eval_test_images(trained_model, output_file, map_file, image_width, image_height,
                     max_images=num_test_images)

    expected_output_file = os.path.join(base_path, "tl_expected_output.txt")
    output = np.fromfile(output_file)
    expected_output = np.fromfile(expected_output_file)
    assert np.allclose(output, expected_output, atol=TOLERANCE_ABSOLUTE)


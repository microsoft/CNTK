# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import pytest
import sys
import json
import zipfile
from cntk import load_model
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import try_set_default_device, gpu
from cntk.logging.graph import get_node_outputs
from cntk.ops.tests.ops_test_utils import cntk_device

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "TransferLearning"))
from _cntk_py import set_fixed_random_seed, force_deterministic_algorithms
from TransferLearning_Extended import train_and_eval
from prepare_test_data import prepare_animals_data

TOLERANCE_ABSOLUTE = 1E-1

def test_transfer_learning(device_id):
    set_fixed_random_seed(1)
    force_deterministic_algorithms()

    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU') # due to batch normalization in ResNet_18
    try_set_default_device(cntk_device(device_id))

    base_path = os.path.dirname(os.path.abspath(__file__))
    animals_path = os.path.join(base_path, *"../../../../Examples/Image/DataSets/Animals".split("/"))
    externalData = 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' in os.environ
    if externalData:
        extPath = os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY']
        model_file = os.path.join(extPath, *"PreTrainedModels/ResNet/v1/ResNet_18.model".split("/"))

        if not os.path.isfile(os.path.join(animals_path, 'Test', 'Weaver_bird.jpg')):
            # copy data from external test data location and unzip
            os.chdir(os.path.join(base_path, '..', '..', '..'))
            prepare_animals_data()
            os.chdir(base_path)
            zip_path = os.path.join(animals_path, 'Animals.zip')
            with zipfile.ZipFile(zip_path) as myzip:
                myzip.extractall(os.path.join(animals_path, '..'))

    else:
        model_file = os.path.join(base_path, *"../../../../PretrainedModels/ResNet_18.model".split("/"))

    train_image_folder = os.path.join(animals_path, "Train")
    test_image_folder = os.path.join(animals_path, "Test")
    output_file = os.path.join(base_path, "tl_extended_output.txt")

    train_and_eval(model_file, train_image_folder, test_image_folder, output_file, None, testing=True)

    expected_output_file = os.path.join(base_path, "tl_extended_expected_output.txt")

    with open(output_file) as output_json:
        output_lines = output_json.readlines()
    with open(expected_output_file) as expected_output_json:
        expected_output_lines = expected_output_json.readlines()

    # handling different ordering of files
    out_dict = {}
    exp_dict = {}
    for i in range(len(output_lines)):
        output = json.loads(output_lines[i])[0]
        expected_output = json.loads(expected_output_lines[i])[0]

        out_dict[output["image"]] = output
        exp_dict[expected_output["image"]] = expected_output

    # debug output
    for k in out_dict:
        output = out_dict[k]
        expected_output = exp_dict[k]

        print("output: {}".format(output))
        print("expect: {}".format(expected_output))

    for k in out_dict:
        output = out_dict[k]
        expected_output = exp_dict[k]

        assert np.allclose(output["predictions"]["Sheep"], expected_output["predictions"]["Sheep"], atol=TOLERANCE_ABSOLUTE)
        assert np.allclose(output["predictions"]["Wolf"], expected_output["predictions"]["Wolf"], atol=TOLERANCE_ABSOLUTE)

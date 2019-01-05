# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
from cntk.ops.tests.ops_test_utils import cntk_device
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import try_set_default_device
import pytest

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Extensibility", "BinaryConvolution"))
from prepare_test_data import prepare_CIFAR10_data
from binary_convnet import *
from cntk.contrib.netopt import native_convolve_function_registered;


TOLERANCE_ABSOLUTE = 4e-1

def test_binary_convnet_error(device_id):

    if not native_convolve_function_registered:
      pytest.skip("Could not find {0} library. "
        "Please check if HALIDE_PATH is configured properly "
        "and try building {1} again"
        .format('Cntk.BinaryConvolution-' + C.__version__.rstrip('+'),
        'Extnsibiliy\\BinaryConvolution'))
     
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')
    try_set_default_device(cntk_device(device_id))

    base_path = prepare_CIFAR10_data()
    # change dir to locate data.zip correctly
    os.chdir(base_path)

    from _cntk_py import set_fixed_random_seed, force_deterministic_algorithms
    set_fixed_random_seed(1)
    force_deterministic_algorithms()

    reader_train = create_reader(os.path.join(base_path, 'train_map.txt'), os.path.join(base_path, 'CIFAR-10_mean.xml'), False)
    model = create_binary_convolution_model()
    z, criterion = get_z_and_criterion(model)
    train_loss, metric = train_model(reader_train, z, criterion, epoch_size=8192, max_epochs=5)

    expected_loss_metric = (2.677057718858123, 0.6043701171875)
    assert np.allclose((train_loss, metric), expected_loss_metric, atol=TOLERANCE_ABSOLUTE)

    # save and load (as an illustration)
    model_path = "model.cmf"
    model.save(model_path)
    eval_device = C.cpu()
    model = Function.load(model_path, device=eval_device)

    # test
    model_with_native_binary_convolutions = clone_with_native_binary_convolutions(model)
    _, criterion = get_z_and_criterion(model_with_native_binary_convolutions)

    reader_test = create_reader(os.path.join(base_path, 'test_map.txt'), os.path.join(base_path, 'CIFAR-10_mean.xml'), False)
    test_loss, metric = evaluate(reader_test, criterion, device=eval_device, minibatch_size=1, max_samples=200)

    expected_loss_metric = (0.0, 0.695)
    assert np.allclose((test_loss, metric), expected_loss_metric, atol=TOLERANCE_ABSOLUTE)

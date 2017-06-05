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
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ConvNet", "Python"))
from prepare_test_data import prepare_CIFAR10_data
from ConvNet_CIFAR10_DataAug import *

TOLERANCE_ABSOLUTE = 1e-1

def test_cifar_convnet_error(device_id):
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
    model = create_convnet_cifar10_model(num_classes=10)
    model.update_signature((num_channels, image_height, image_width))
    criterion = create_criterion_function(model, normalize=lambda x: x / 256)
    train_loss, metric = train_model(reader_train, model, criterion, epoch_size=128, max_epochs=5)

    expected_loss_metric = (2.2963, 0.9062)
    assert np.allclose((train_loss, metric), expected_loss_metric, atol=TOLERANCE_ABSOLUTE)

if __name__=='__main__':
    test_cifar_convnet_error(0)

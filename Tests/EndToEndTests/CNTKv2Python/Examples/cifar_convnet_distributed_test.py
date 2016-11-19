# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import sys
from cntk.utils import cntk_device
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import set_default_device
from cntk.io import ReaderConfig, ImageDeserializer
from cntk import distributed
import pytest
import platform

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Classification", "ConvNet", "Python"))
from ConvNet_CIFAR10_DataAug_Distributed import convnet_cifar10_dataaug, create_reader

TOLERANCE_ABSOLUTE = 2E-1

def test_cifar_convnet_error(device_id):
    if platform.system() == 'Windows':
        pytest.skip('test skipped on Windows')

    set_default_device(cntk_device(device_id))

    try:
        base_path = os.path.join(os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'],
                                *"Image/CIFAR/v0/cifar-10-batches-py".split("/"))
        # N.B. CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY has {train,test}_map.txt
        #      and CIFAR-10_mean.xml in the base_path.
    except KeyError:
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                *"../../../../Examples/Image/DataSets/CIFAR-10".split("/"))

    base_path = os.path.normpath(base_path)
    os.chdir(os.path.join(base_path, '..'))

    from _cntk_py import set_computation_network_trace_level, set_fixed_random_seed, force_deterministic_algorithms
    set_computation_network_trace_level(1) 
    set_fixed_random_seed(1)  # BUGBUG: has no effect at present  # TODO: remove debugging facilities once this all works
    #force_deterministic_algorithms()
    # TODO: do the above; they lead to slightly different results, so not doing it for now

    reader_train = create_reader(os.path.join(base_path, 'train_map.txt'), os.path.join(base_path, 'CIFAR-10_mean.xml'), True, 0)
    reader_test  = create_reader(os.path.join(base_path, 'test_map.txt'), os.path.join(base_path, 'CIFAR-10_mean.xml'), False)

    distributed_after_samples = 0
    num_quantization_bits = 32
    distributed_trainer = distributed.data_parallel_distributed_trainer(
        num_quantization_bits=num_quantization_bits,
        distributed_after=distributed_after_samples)

    test_error = convnet_cifar10_dataaug(reader_train, reader_test, distributed_trainer, max_epochs=1)
    expected_test_error = 0.617

    assert np.allclose(test_error, expected_test_error,
                       atol=TOLERANCE_ABSOLUTE)

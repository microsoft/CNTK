# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from cntk import DeviceDescriptor
from cntk.cntk_py import DeviceKind_GPU
from cntk.persist import load_model, save_model
import os

TOLERANCE_ABSOLUTE = 1E-1

from cntk.blocks import *
from cntk.layers import *
from cntk.models import *
from cntk.utils import *
from cntk.ops import splice
from examples.CifarConvNet.CifarConvNet import data_path, create_reader, create_basic_model_layer, train_and_evaluate, evaluate


TOLERANCE_ABSOLUTE = 2E-1

def test_cifar_convnet_error(device_id):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')
    DeviceDescriptor.set_default_device(cntk_device(device_id))

    from _cntk_py import set_computation_network_trace_level, set_fixed_random_seed, force_deterministic_algorithms
    #set_computation_network_trace_level(1)  # TODO: remove debugging facilities once this all works
    set_fixed_random_seed(1)
    force_deterministic_algorithms()
    # TODO: do the above; they lead to slightly different results, so not doing it for now

    # create model
    model = create_basic_model_layer()   # TODO: clean this up more

    # train
    os.chdir(data_path) # BUGBUG: This is only needed because ImageReader uses relative paths in the map file. Ugh.
    reader_train = create_reader(data_path, 'train_map.txt', 'CIFAR-10_mean.xml', is_training=True)
    reader_test  = create_reader(data_path, 'test_map.txt',  'CIFAR-10_mean.xml', is_training=False)
    loss_avg, evaluation_avg = train_and_evaluate(reader_train, reader_test, model, max_epochs=1)
    print("-->", evaluation_avg, loss_avg)
    expected_avg = [5.47968, 1.5783466666030883]
    assert np.allclose([evaluation_avg, loss_avg], expected_avg, atol=TOLERANCE_ABSOLUTE)

    # save and load
    path = data_path + "/model.cmf"
    save_model(model, path)
    model = load_model(path)

    # test
    reader_test  = create_reader(data_path, 'test_map.txt', 'CIFAR-10_mean.xml', is_training=False)
    evaluate(reader_test, model)
    # BUGBUG: fails eval with "RuntimeError: __v2libuid__BatchNormalization226__v2libname__BatchNormalization19: inference mode is used, but nothing has been trained."

if __name__=='__main__':
    test_cifar_convnet_error(0)

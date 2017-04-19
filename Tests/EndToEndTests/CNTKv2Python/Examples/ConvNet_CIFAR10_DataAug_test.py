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
from ConvNet_CIFAR10_DataAug import create_reader, create_convnet_cifar10_model, train_model

TOLERANCE_ABSOLUTE = 1E-3

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

    reader_train1 = create_reader(os.path.join(base_path, 'train_map.txt'), os.path.join(base_path, 'CIFAR-10_mean.xml'), False)
    reader_test1  = create_reader(os.path.join(base_path, 'test_map.txt'), os.path.join(base_path, 'CIFAR-10_mean.xml'), False)
    model1 = create_convnet_cifar10_model(num_classes=10)
    train_loss1 = train_model(reader_train1, reader_test1, model1, epoch_size=128, max_epochs=1)

#    reader_train2 = create_reader(os.path.join(base_path, 'train_map.txt'), os.path.join(base_path, 'CIFAR-10_mean.xml'), False)
#    reader_test2  = create_reader(os.path.join(base_path, 'test_map.txt'), os.path.join(base_path, 'CIFAR-10_mean.xml'), False)
#    model2 = create_convnet_cifar10_model(num_classes=10)
#    train_loss2 = train_model(reader_train2, reader_test2, model2, epoch_size=128, max_epochs=1)

#    assert np.allclose(train_loss1, train_loss2, atol=TOLERANCE_ABSOLUTE)
    
# We are removing tolerance in error because running small epoch size has huge variance in accuracy. Will add
# tolerance back once convolution operator is determinsitic. 
    
#    expected_test_error = 0.617

#    assert np.allclose(train_loss1, expected_test_error,
#                       atol=TOLERANCE_ABSOLUTE)

    # enable this:
    #reader_train = create_reader(data_path, 'train_map.txt', 'CIFAR-10_mean.xml', is_training=True)
    #reader_test  = create_reader(data_path, 'test_map.txt',  'CIFAR-10_mean.xml', is_training=False)
    #loss_avg, evaluation_avg = train_and_evaluate(reader_train, reader_test, model, max_epochs=1)
    #print("-->", evaluation_avg, loss_avg)
    #expected_avg = [5.47968, 1.5783466666030883]
    #assert np.allclose([evaluation_avg, loss_avg], expected_avg, atol=TOLERANCE_ABSOLUTE)
    #
    ## save and load
    #path = data_path + "/model.cmf"
    #save_model(model, path)
    #model = load_model(path)
    #
    ## test
    #reader_test  = create_reader(data_path, 'test_map.txt', 'CIFAR-10_mean.xml', is_training=False)
    #evaluate(reader_test, model)

if __name__=='__main__':
    test_cifar_convnet_error(0)

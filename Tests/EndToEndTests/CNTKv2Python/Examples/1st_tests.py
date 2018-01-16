# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os, sys
import numpy as np
from cntk import DeviceDescriptor

from cntk import placeholder
from cntk.layers import *
from cntk.internal.utils import *
from cntk.logging import *
from cntk.ops import splice
from cntk.cntk_py import reset_random_seed
from cntk.device import try_set_default_device

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "1stSteps"))

def test_1st_steps_functional(device_id):
    from cntk.ops.tests.ops_test_utils import cntk_device
    try_set_default_device(cntk_device(device_id))
    reset_random_seed(0)
    from LogisticRegression_FunctionalAPI import final_loss, final_metric, final_samples, test_metric
    # these are the final values from the log output
    assert np.allclose(final_loss,   0.344399, atol=1e-5)
    assert np.allclose(final_metric, 0.1258,   atol=1e-4)
    assert np.allclose(test_metric,  0.0811,   atol=1e-4)
    assert final_samples == 20000

def test_1st_steps_graph(device_id):
    from cntk.ops.tests.ops_test_utils import cntk_device
    try_set_default_device(cntk_device(device_id))
    reset_random_seed(0)
    from LogisticRegression_GraphAPI import trainer, evaluator, X_test, Y_test, data, label_one_hot
    #print(trainer.previous_minibatch_loss_average)
    assert np.allclose(trainer.previous_minibatch_loss_average, 0.1233455091714859, atol=1e-5)
    assert trainer.previous_minibatch_sample_count == 32
    # evaluator does not have a way to correctly get the aggregate, so we run one more MB on it
    i = 0
    x = X_test[i:i+32] # get one minibatch worth of data
    y = Y_test[i:i+32]
    metric = evaluator.test_minibatch({data: x, label_one_hot: y})
    #print(metric)
    assert np.allclose(metric, 0.0625, atol=1e-5)

def test_1st_steps_mnist(device_id):
    from cntk.ops.tests.ops_test_utils import cntk_device
    cntk_py.force_deterministic_algorithms()
    cntk_py.set_fixed_random_seed(1)
    try_set_default_device(cntk_device(device_id))
    reset_random_seed(0)
    from MNIST_Complex_Training import final_loss, final_metric, final_samples, test_metric
    print(final_loss, final_metric, final_samples, test_metric)
    # these are the final values from the log output
    # Since this has convolution, there is some variance.
    # Finished Epoch[36]: loss = 0.009060 * 54000, metric = 0.27% * 54000 1.971s (27397.3 samples/s);
    # Finished Evaluation [12]: Minibatch[1-24]: metric = 0.65% * 6000;
    # Learning rate 7.8125e-06 too small. Training complete.
    # Finished Evaluation [13]: Minibatch[1-313]: metric = 0.63% * 10000;
    assert np.allclose(final_loss,   0.009060, atol=1e-5)
    assert np.allclose(final_metric, 0.0027,   atol=1e-3)
    assert np.allclose(test_metric,  0.0063,   atol=1e-3)
    assert final_samples == 54000

if __name__=='__main__':
    # run them directly so that this can be run without pytest
    test_1st_steps_mnist(0)
    test_1st_steps_functional(0)
    test_1st_steps_graph(0)

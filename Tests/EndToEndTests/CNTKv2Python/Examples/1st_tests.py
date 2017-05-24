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

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "1stSteps"))

def test_1st_steps_functional():
    reset_random_seed(0)
    from LogisticRegression_FunctionalAPI import losses, metrics, metric, num_samples
    # these are the final values from the log output
    assert np.allclose(losses[-1],  0.243953, atol=1e-5)
    assert np.allclose(metrics[-1], 0.0831,   atol=1e-4)
    assert np.allclose(metric,      0.0811,   atol=1e-4)
    assert num_samples == 1024

def test_1st_steps_graph():
    reset_random_seed(0)
    from LogisticRegression_GraphAPI import trainer, evaluator, X_test, Y_test, data, label_one_hot
    print(trainer.previous_minibatch_loss_average)
    assert np.allclose(trainer.previous_minibatch_loss_average, 0.1233455091714859, atol=1e-5)
    assert trainer.previous_minibatch_sample_count == 32
    # evaluator does not have a way to correctly get the aggregate, so we run one more MB on it
    i = 0
    x = X_test[i:i+32] # get one minibatch worth of data
    y = Y_test[i:i+32]
    metric = evaluator.test_minibatch({data: x, label_one_hot: y})
    print(metric)
    assert np.allclose(metric, 0.0625, atol=1e-5)

if __name__=='__main__':
    # run them directly so that this can be run without pytest
    test_1st_steps_functional()
    test_1st_steps_graph()

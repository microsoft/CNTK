# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os, sys
import numpy as np
import shutil
from cntk import DeviceDescriptor

from cntk import placeholder
from cntk.layers import *
from cntk.internal.utils import *
from cntk.logging import *
from cntk.ops import splice

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "1stSteps"))
sys.path.append("../1stSteps")

def test_1st_steps_functional():
    from LogisticRegression_FunctionalAPI import losses, metrics, metric, num_samples
    assert np.allclose(losses[-1],  0.239423828125, atol=1e-5)
    assert np.allclose(metrics[-1], 0.0832,         atol=1e-5)
    assert np.allclose(metric,      0.087,          atol=1e-5)
    assert num_samples == 1000

def test_1st_steps_graph():
    from LogisticRegression_GraphAPI import trainer, evaluator, X_test, Y_test, data, label_one_hot
    assert np.allclose(trainer.previous_minibatch_loss_average, 0.11803203582763672, atol=1e-5)
    assert trainer.previous_minibatch_sample_count == 25
    # evaluator does not have a way to correctly get the aggregate, so we run one more MB on it
    i = 0
    x = X_test[i:i+25] # get one minibatch worth of data
    y = Y_test[i:i+25]
    metric = evaluator.test_minibatch({data: x, label_one_hot: y})
    print(metric)
    assert np.allclose(metric, 0.08, atol=1e-5)

if __name__=='__main__':
    # run them directly so that this can be run without pyteset
    test_1st_steps_graph()
    test_1st_steps_functional()

# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Example of logictic regression implementation using training and testing data
from a NumPy array. 
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import numpy as np

import cntk as C

def train_eval_logistic_regression_with_numpy(criterion_name=None, eval_name=None):

    # for repro and tests :-)
    np.random.seed(1)

    N = 500
    d = 250

    # create synthetic data using numpy
    X = np.random.randn(N, d)
    Y = np.random.randint(size=(N, 1), low=0, high=2)
    Y = np.hstack((Y, 1-Y))

    # set up the training data for CNTK
    x = C.input_numpy(X, has_dynamic_axis=False)
    y = C.input_numpy(Y, has_dynamic_axis=False)

    # define our network -- one weight tensor and a bias
    W = C.ops.parameter((2, d))
    b = C.ops.parameter((2, 1))
    out = C.ops.times(W, x) + b

    ce = C.ops.cross_entropy_with_softmax(y, out)
    ce.tag = 'criterion'
    ce.name = criterion_name    
    
    eval = C.ops.cntk1.SquareError(y, out)
    eval.tag = 'eval'
    eval.name = eval_name

    my_sgd = C.SGDParams(epoch_size=0, minibatch_size=25, learning_rates_per_mb=0.1, max_epochs=3)
    with C.LocalExecutionContext('logreg', clean_up=False) as ctx:
        ctx.train(
                root_nodes=[ce,eval], 
                optimizer=my_sgd)

        result = ctx.test(root_nodes=[ce,eval])
        return result


def test_logistic_regression_with_numpy():
    result = train_eval_logistic_regression_with_numpy('crit_node', 'eval_node')

    TOLERANCE_ABSOLUTE = 1E-06
    assert result['SamplesSeen'] == 500
    assert np.allclose(result['Perplexity'], 1.5575403, atol=TOLERANCE_ABSOLUTE)
    assert np.allclose(result['crit_node'], 0.44310782, atol=TOLERANCE_ABSOLUTE)
    assert np.allclose(result['eval_node'], 1.4050217, atol=TOLERANCE_ABSOLUTE)

if __name__ == "__main__":
    print(train_eval_logistic_regression_with_numpy())

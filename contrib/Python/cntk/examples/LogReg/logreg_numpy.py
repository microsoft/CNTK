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

def train_eval_logistic_regression_with_numpy(criterion_name=None,
        eval_name=None, device_id=-1):

    # for repro and tests :-)
    np.random.seed(1)

    N = 500
    d = 250

    # create synthetic data using numpy
    X = np.random.randn(N, d)
    Y = np.random.randint(size=(N, 1), low=0, high=2)
    Y = np.hstack((Y, 1-Y))

    # set up the training data for CNTK
    x = C.input_numpy(X)
    y = C.input_numpy(Y)

    # define our network -- one weight tensor and a bias
    W = C.parameter(value=np.zeros(shape=(2, d)))
    b = C.parameter(value=np.zeros(shape=(2, 1)))
    out = C.times(W, x) + b

    ce = C.cross_entropy_with_softmax(y, out)
    ce.tag = 'criterion'
    ce.name = criterion_name    
    
    eval = C.ops.cntk1.SquareError(y, out)
    eval.tag = 'eval'
    eval.name = eval_name

    my_sgd = C.SGDParams(epoch_size=0, minibatch_size=25, learning_rates_per_mb=0.1, max_epochs=3)
    with C.LocalExecutionContext('logreg') as ctx:
        ctx.device_id = device_id

        ctx.train(
                root_nodes=[ce,eval], 
                training_params=my_sgd)

        result = ctx.test(root_nodes=[ce,eval])
        return result


def test_logistic_regression_with_numpy(device_id):
    result = train_eval_logistic_regression_with_numpy('crit_node',
            'eval_node', device_id)

    TOLERANCE_ABSOLUTE = 1E-02
    assert np.allclose(result['perplexity'], 1.55057073, atol=TOLERANCE_ABSOLUTE)
    assert np.allclose(result['crit_node'], 0.43862308, atol=TOLERANCE_ABSOLUTE)
    assert np.allclose(result['eval_node'], 1.16664551, atol=TOLERANCE_ABSOLUTE)

if __name__ == "__main__":
    print(train_eval_logistic_regression_with_numpy())

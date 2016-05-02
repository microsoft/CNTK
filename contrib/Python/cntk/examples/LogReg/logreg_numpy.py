# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Example of logictic regression implementation using training and testing data
from a file. 
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import numpy as np

import cntk as C


def train_eval_logistic_regression_with_numpy(criterion_name=None, eval_name=None):

    N = 500
    d = 250

    X = np.random.randn(N, d)
    Y = np.random.randint(size=(N, 1), low=0, high=2)

    x = C.input_reader(X, has_dynamic_axis=False)
    y = C.input_reader(Y, has_dynamic_axis=False)

    W = C.ops.parameter((2, d))
    b = C.ops.parameter((2, 1))
    out = C.ops.times(W, x) + b
    out.tag = 'output'

    ce = C.ops.cross_entropy_with_softmax(y, out)
    ce.tag = 'criterion'

    eval = C.ops.cntk1.SquareError(y, out)
    eval.tag = 'eval'
    eval.name = eval_name

    my_sgd = C.SGDParams(epoch_size=0, minibatch_size=25, learning_rates_per_mb=0.1, max_epochs=3)
    with C.LocalExecutionContext('logreg') as ctx:
        ctx.train(
                root_nodes=[ce], 
                optimizer=my_sgd)

        result = ctx.test(
                root_nodes=[ce], 
                input_map=test_reader.map(X, alias='I', dim=2).map(y, alias='L', dim=3))

        return result


def test_logistic_regression_with_numpy():
    result = train_eval_logistic_regression_with_numpy('crit_node', 'eval_node')

    if False:
        TOLERANCE_ABSOLUTE = 1E-06
        assert result['SamplesSeen'] == 500
        assert np.allclose(result['Perplexity'], 1.5584637, atol=TOLERANCE_ABSOLUTE)
        assert np.allclose(result['crit_node'], 0.4437005, atol=TOLERANCE_ABSOLUTE)
        assert np.allclose(result['eval_node'], 2.7779043, atol=TOLERANCE_ABSOLUTE)

if __name__ == "__main__":
    print(train_eval_logistic_regression_with_numpy())

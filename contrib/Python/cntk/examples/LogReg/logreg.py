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


def train_eval_logistic_regression_from_file(criterion_name=None, eval_name=None):
    cur_dir = os.path.dirname(__file__)

    # Using data from https://github.com/Microsoft/CNTK/wiki/Tutorial
    train_file = os.path.join(cur_dir, "Train-3Classes.txt")
    test_file = os.path.join(cur_dir, "Test-3Classes.txt")

    X = C.input(2)
    y = C.input(3)
    
    W = C.parameter((3, 2))
    b = C.parameter((3, 1))

    out = C.times(W, X) + b
    out.tag = 'output'
    ce = C.cross_entropy_with_softmax(y, out)
    ce.name = criterion_name
    ce.tag = 'criterion'
    eval = C.ops.cntk1.SquareError(y, out)
    eval.tag = 'eval'
    eval.name = eval_name

    # training data readers
    train_reader = C.CNTKTextFormatReader(train_file)

    # testing data readers
    test_reader = C.CNTKTextFormatReader(test_file)

    my_sgd = C.SGDParams(
        epoch_size=0, minibatch_size=25, learning_rates_per_mb=0.1, max_epochs=3)

    with C.LocalExecutionContext('demo') as ctx:

        ctx.train(
            root_nodes=[ce, eval], 
            training_params=my_sgd,
            input_map=train_reader.map(X, alias='I', dim=2).map(y, alias='L', dim=3))

        result = ctx.test(
                root_nodes=[ce, eval], 
                input_map=test_reader.map(X, alias='I', dim=2).map(y, alias='L', dim=3))

        return result


def test_logistic_regression_from_file():
    result = train_eval_logistic_regression_from_file('crit_node', 'eval_node')

    TOLERANCE_ABSOLUTE = 1E-06
    assert result['SamplesSeen'] == 500
    assert np.allclose(result['Perplexity'], 1.5584637, atol=TOLERANCE_ABSOLUTE)
    assert np.allclose(result['crit_node'], 0.4437005, atol=TOLERANCE_ABSOLUTE)
    assert np.allclose(result['eval_node'], 2.7779043, atol=TOLERANCE_ABSOLUTE)

if __name__ == "__main__":
    print(train_eval_logistic_regression_from_file())

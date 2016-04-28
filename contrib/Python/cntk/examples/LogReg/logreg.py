# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# TODO: re-write the example using the new facade

"""
Example of logictic regression implementation 
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from cntk import *
from cntk.ops import *
from cntk.ops import cntk1

cur_dir = os.path.dirname(__file__)

# Using data from https://github.com/Microsoft/CNTK/wiki/Tutorial
train_file = os.path.join(cur_dir, "Train-3Classes.txt")
test_file = os.path.join(cur_dir, "Test-3Classes.txt")


def train_eval_logistic_regression(criterion_name=None, eval_name=None):
    X = input(2)
    y = input(3)
    
    W = parameter((3, 2))
    b = parameter((3, 1))

    out = times(W, X) + b
    out.tag = 'output'
    ce = cntk1.CrossEntropyWithSoftmax(y, out)
    ce.var_name = criterion_name
    ce.tag = 'criterion'
    eval = cntk1.SquareError(y, out)
    eval.tag = 'eval'
    eval.var_name = eval_name

    # training data readers
    train_reader = CNTKTextFormatReader(train_file)

    # testing data readers
    test_reader = CNTKTextFormatReader(test_file)

    my_sgd = SGDParams(
        epoch_size=0, minibatch_size=25, learning_ratesPerMB=0.1, max_epochs=3)

    with Context('demo') as ctx:

        ctx.train(
            root_nodes=[ce, eval], 
            optimizer=my_sgd,
            input_map=train_reader.map(X, alias='I', dim=2).map(y, alias='L', dim=3))

        result = ctx.test(input_map=test_reader.map(X, alias='I', dim=2).map(y, alias='L', dim=3))

        return result


def test_logistic_regression():
    result = train_eval_logistic_regression('crit_node', 'eval_node')

    assert result['SamplesSeen'] == 500
    assert np.allclose(result['Perplexity'], 1.5584637)
    assert np.allclose(result['crit_node'], 0.4437005)
    assert np.allclose(result['eval_node'], 2.7779043)

if __name__ == "__main__":
    print(train_eval_logistic_regression())

# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from cntk.metrics import classification_error
import cntk as C


def test_eval():
    input_dim = 2
    proj_dim = 2

    x = C.input_variable(shape=(input_dim,))
    W = C.parameter(shape=(input_dim, proj_dim), init=[[1, 0], [0, 1]])
    B = C.parameter(shape=(proj_dim,), init=[[0, 1]])
    t = C.times(x, W)
    z = t + B

    labels = C.input_variable(shape=(proj_dim,))
    pe = classification_error(z, labels)

    tester = C.eval.Evaluator(pe)

    x_value = [[0, 1], [2, 2]]
    label_value = [[0, 1], [1, 0]]
    arguments = {x: x_value, labels: label_value}
    eval_error = tester.test_minibatch(arguments)

    assert np.allclose(eval_error, .5)

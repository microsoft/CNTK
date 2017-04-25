# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import math
import numpy as np
from cntk import *
from cntk.ops.tests.ops_test_utils import cntk_device
from ..evaluator import *
from cntk.metrics import classification_error
from cntk import parameter, input, times, plus, reduce_sum, Axis, cntk_py
import pytest

def test_eval():
    input_dim = 2
    proj_dim = 2
    
    x = input(shape=(input_dim,))
    W = parameter(shape=(input_dim, proj_dim), init=[[1, 0], [0, 1]])
    B = parameter(shape=(proj_dim,), init=[[0, 1]])
    t = times(x, W)
    z = t + B

    labels = input(shape=(proj_dim,))
    pe = classification_error(z, labels)

    tester = Evaluator(pe)

    x_value = [[0, 1], [2, 2]]
    label_value = [[0, 1], [1, 0]]
    arguments = {x: x_value, labels: label_value}
    eval_error = tester.test_minibatch(arguments)

    assert np.allclose(eval_error, .5)

# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import pytest

from cntk.ops import *
from cntk.utils import load_model, save_model


def test_load_save_constant():
    c = constant(value=[1,3])
    root_node = c * 5

    result = root_node.eval()
    expected = [[[[5,15]]]]
    assert np.allclose(result, expected)

    filename = 'c_plus_c.mod'
    save_model(root_node, filename)

    loaded_node = load_model('float', filename)
    loaded_result = loaded_node.eval()
    assert np.allclose(loaded_result, expected)

def test_load_save_input():
    i1 = input_variable((1,2), name='i1')
    root_node = abs(i1)
    input1 = [[[-1,2]]]

    result = root_node.eval({i1: input1})
    expected = [[[[1,2]]]]
    assert np.allclose(result, expected)

    filename = 'i_plus_c_0.mod'
    save_model(root_node, filename)

    loaded_node = load_model('float', filename)

    # Test spefying the input node names by order
    loaded_result = loaded_node.eval([input1])
    assert np.allclose(loaded_result, expected)
    
def test_load_save_inputs():
    i1 = input_variable((1,2), name='i1')
    i2 = input_variable((2,1), name='i2')
    root_node = plus(i1, i2)
    input1 = [[[1,2]]]
    input2 = [[[[1],[2]]]]

    result = root_node.eval({i1: input1, i2: input2})
    expected = [[[[2,3],[3,4]]]]
    assert np.allclose(result, expected)

    filename = 'i_plus_i_0.mod'
    save_model(root_node, filename)

    loaded_node = load_model('float', filename)

    # Test specifying the input nodes by name
    loaded_result = loaded_node.eval({'i1': input1, 'i2': input2})
    assert np.allclose(loaded_result, expected)

    # Test spefying the input node names by order
    loaded_result = loaded_node.eval([input1, input2])
    assert np.allclose(loaded_result, expected)
    

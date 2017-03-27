# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np

from cntk.ops import *
from cntk.debugging import save_as_legacy_model
from cntk.ops.functions import load_model

# TODO: a test for restore_model?

def test_load_save_constant(tmpdir):
    c = constant(value=[1,3])
    root_node = c * 5

    result = root_node.eval()
    expected = [[[[5,15]]]]
    assert np.allclose(result, expected)

    filename = str(tmpdir / 'c_plus_c.mod')
    root_node.save(filename)

    loaded_node = Function.load(filename)
    loaded_result = loaded_node.eval()
    assert np.allclose(loaded_result, expected)

    filename = filename + '.legacy'
    save_as_legacy_model(root_node, filename)
    loaded_node = Function.load(filename)
    loaded_result = loaded_node.eval()
    assert np.allclose(loaded_result, expected)

def test_load_save_input_legacy_names(tmpdir):
    i1 = input((1,2), name='i1')
    root_node = abs(i1)
    input1 = [[[-1,2]]]

    result = root_node.eval({i1: input1})
    expected = [[[[1,2]]]]
    assert np.allclose(result, expected)

    filename = str(tmpdir / 'i_plus_c_0.mod')
    root_node.save_model(filename)

    loaded_node = load_model(filename)

    # Test specifying the input node names by order
    loaded_result = loaded_node.eval([input1])
    assert np.allclose(loaded_result, expected)

    filename = filename + '.legacy'
    save_as_legacy_model(root_node, filename)
    loaded_node = load_model(filename)
    loaded_result = loaded_node.eval([input1])
    assert np.allclose(loaded_result, expected)
    
def test_load_save_inputs(tmpdir):
    i1 = input((1,2), name='i1')
    i2 = input((2,1), name='i2')
    root_node = plus(i1, i2)
    input1 = [[[1,2]]]
    input2 = [[[[1],[2]]]]

    result = root_node.eval({i1: input1, i2: input2})
    expected = [[[[2,3],[3,4]]]]
    assert np.allclose(result, expected)

    filename = str(tmpdir / 'i_plus_i_0.mod')
    root_node.save(filename)

    loaded_node = Function.load(filename)

    # Test specifying the input nodes by name
    loaded_result = loaded_node.eval({'i1': input1, 'i2': input2})
    assert np.allclose(loaded_result, expected)

    filename = filename + '.legacy'
    save_as_legacy_model(root_node, filename)
    loaded_node = Function.load(filename)
    loaded_result = loaded_node.eval({'i1': input1, 'i2': input2})
    assert np.allclose(loaded_result, expected)

def test_load_save_unique_input(tmpdir):
    i1 = input((1,2), name='i1')
    root_node = softmax(i1)

    input1 = [[[1,2]]]
    result = root_node.eval(input1)
    expected = [[[[ 0.268941,  0.731059]]]]
    assert np.allclose(result, expected)

    filename = str(tmpdir / 'i_plus_0.mod')
    root_node.save(filename)

    loaded_node = Function.load(filename)

    # Test specifying the only value for an unique input
    loaded_result = loaded_node.eval(input1)
    assert np.allclose(loaded_result, expected)

    filename = filename + '.legacy'
    save_as_legacy_model(root_node, filename)
    loaded_node = Function.load(filename)
    loaded_result = loaded_node.eval(input1)
    assert np.allclose(loaded_result, expected)

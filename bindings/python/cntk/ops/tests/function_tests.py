# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for the function class.
"""

import numpy as np
import pytest
from ..functions import *
from ...trainer import *
from .. import constant, parameter, input_variable, placeholder_variable, times, plus
from .ops_test_utils import compare_lists_of_np_arrays

def test_variable_forwarding():
    op = constant(value=2, shape=(3,4)) + 1
    assert op.shape == (3,4)


def test_replace_placeholders():
    p = placeholder_variable(shape=(1,))
    i = input_variable(shape=(1,),
                       needs_gradient=True,
                       name='i')
    res = p + 3
    res.replace_placeholders({p: i})

    assert res.eval({i: [[3]]}) == [6]

    if False:
        res2 = p + 2
        from .. import plus
        func = plus(res2, 10)
        res2.replace_placeholders({p: func.output})

        assert res2.eval({i: [3]}) == [15]

def test_cloning():
    p = placeholder_variable(shape=(1,), name='p')
    i = input_variable(shape=(1,),
                       needs_gradient=True,
                       name='i')
    res = p + i

    with pytest.raises(ValueError):
        res.clone(2)

    from ..functions import CloneMethod

    # Test freeze
    cloned = res.clone(CloneMethod.freeze)
    assert cloned.inputs[0].name == 'p'
    assert cloned.inputs[0].uid != p.uid
    assert cloned.inputs[1].name == 'i'
    assert cloned.inputs[1].uid != i.uid

    cloned = res.clone('freeze')
    assert cloned.inputs[0].name == 'p'
    assert cloned.inputs[0].uid != p.uid
    assert cloned.inputs[1].name == 'i'
    assert cloned.inputs[1].uid != i.uid


def test_replace_placeholder_s():
    left_val = [[10,2]]
    right_val = [[2],[3]]

    p = placeholder_variable(shape=(1,2))
    c = constant(left_val)

    op = times(p, right_val)
    op.replace_placeholders({p:c})
    assert op.eval() == 26

    op = times(p, right_val)
    op.replace_placeholder(c)
    assert op.eval() == 26

def test_exception_for_unnamed_arguments():
    i1 = input_variable((1,2), name='i1')
    i2 = input_variable((2,1), name='i2')
    root_node = plus(i1, i2)
    input1 = [[[1,2]]]
    input2 = [[[[1],[2]]]]

    with pytest.raises(Exception):
        # not allowed, since plus has more than 1 input
        result = root_node.eval([input1, input2])

def test_output_in_intermediate_node():
    x = input_variable((2,))
    y = input_variable((2,))
    x0 = np.asarray([[2., 1.]])
    y0 = np.asarray([[4., 6.]])

    sum_node = x + 2
    times_node = sum_node * y

    sum_output = times_node.forward({x: x0, y: y0}, sum_node.outputs)

    assert len(sum_output[1]) == 1
    assert np.allclose(list(sum_output[1].values())[0], np.asarray(x0 + 2))

    two_nodes_output = times_node.forward({x: x0, y: y0}, times_node.outputs + sum_node.outputs)

    assert len(two_nodes_output[1]) == 2

    sum_forward = np.asarray(x0 + 2)
    expected_results = [sum_forward, sum_forward * y0]

    assert compare_lists_of_np_arrays(list(two_nodes_output[1].values()), expected_results)

def test_getting_output_from_non_existent_node():
    x = input_variable((2,))
    y = input_variable((2,))
    x0 = np.asarray([[2., 1.]])
    y0 = np.asarray([[4., 6.]])

    sum_node = x + 2

    # times_node not having sum_node
    times_node = x * y

    with pytest.raises(ValueError):
        sum_output = times_node.forward({x: x0, y: y0}, sum_node.outputs)

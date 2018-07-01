# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for operations that are not differentiable.
"""

from __future__ import division
import numpy as np
import cntk as C
import pytest
from .ops_test_utils import unittest_helper, _test_unary_op, _test_binary_op, AA, precision, PRECISION_TO_TYPE

TENSORS = [
    ([12.3, -12.3]),
    ([10.2, -10.2]),
    ([0.5, -0.5]),
    ([0.01, -0.01]),
    ([0.499, -0.499]),
    ([5.0, -5.0]),
    ([0.0]),
    ([[2.1, 9.9], [4.7, 5.3]])
]


@pytest.mark.parametrize("operand", TENSORS)
def test_op_floor(operand, device_id, precision):
    operand = AA(operand)
    expected = np.floor(operand)

    expected_forward = [expected]
    expected_backward = {
        'arg': [np.zeros_like(expected)],
    }

    from .. import floor
    _test_unary_op(precision, device_id, floor, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("operand", TENSORS)
def test_op_ceil(operand, device_id, precision):
    operand = AA(operand)
    expected = np.ceil(operand)

    expected_forward = [expected]
    expected_backward = {
        'arg': [np.zeros_like(expected)],
    }

    from .. import ceil
    _test_unary_op(precision, device_id, ceil, operand,
                   expected_forward, expected_backward)

# Manually setting the expectation since CNTK's round behaves differently than
# NumPy's round (see operator's docstring).
ROUND_TENSORS = [
    ([0.2, 1.3, 4.0, 5.5, 0.0],
     [0.0, 1.0, 4.0, 6.0, 0.0]),

    ([[0.6, 3.3], [1.9, 5.6]],
     [[1.0, 3.0], [2.0, 6.0]]),

    ([-5.5, -4.2, -3., -0.7, 0],
     [-5.0, -4.0, -3., -1.0, 0]),

    ([[-0.6, -4.3], [1.9, -3.2]],
     [[-1.0, -4.0], [2.0, -3.0]]),

    # CNTK is always rounding up values starting at x.5, while numpy rounds
    # to the nearest even value for half-integers
    # Refer here: https://en.wikipedia.org/wiki/Rounding#Tie-breaking
    # This test shows such values are not equal comparing numpy and CNTK
    ([0.5, 1.5, 2.5, 3.5],
     # NumPy would round to
     #        [0.0, 2.0, 2.0, 4.0]))
     # while CNTK rounds to
     [1.0, 2.0, 3.0, 4.0])
]


@pytest.mark.parametrize("operand,expected", ROUND_TENSORS)
def test_op_round(operand, expected, device_id, precision):
    operand, expected = AA(operand), AA(expected)

    expected_forward = [expected]
    expected_backward = {
        'arg': [np.zeros_like(expected)],
    }

    from .. import round
    _test_unary_op(precision, device_id, round, operand,
                   expected_forward, expected_backward)

def test_input_variable():
    from .. import sequence
    i = C.input_variable(shape=(2,3), name='i')
    assert i.shape == (2,3)
    assert i.name == 'i'
    assert len(i.dynamic_axes)==1

    sequence_i = sequence.input_variable(shape=(3,2), name='sequence_i')
    assert sequence_i.shape == (3,2)
    assert sequence_i.name == 'sequence_i'
    assert len(sequence_i.dynamic_axes)==2


@pytest.mark.parametrize("operand", TENSORS)
def test_zeros_like(operand, device_id, precision):
    operand = AA(operand)
    expected = np.zeros_like(operand)

    expected_forward = [expected]
    expected_backward = {
        'arg': [np.zeros_like(expected)],
    }

    from .. import zeros_like
    _test_unary_op(precision, device_id, zeros_like, operand,
                   expected_forward, expected_backward)

def test_zeros_like_empty_shape(device_id, precision):
    operand = np.array(3)
    expected = np.zeros_like(operand)

    expected_forward = [expected]
    expected_backward = {
        'arg': [np.zeros_like(expected)],
    }

    from .. import zeros_like
    _test_unary_op(precision, device_id, zeros_like, operand,
                   expected_forward, expected_backward)


@pytest.mark.parametrize("operand", TENSORS)
def test_ones_like(operand, device_id, precision):
    operand = AA(operand)
    expected = np.ones_like(operand)

    expected_forward = [expected]
    expected_backward = {
        'arg': [np.zeros_like(expected)],
    }

    from .. import ones_like
    _test_unary_op(precision, device_id, ones_like, operand,
                   expected_forward, expected_backward)


Matrices = [
    ([[2.1, 4.7], [2.1, 2.1]], True),
    ([[2.1, 2., 2.], [4.7, 3, 5], [5.1, 2, 5]], True),
    ([[2.1], [4.7], [5.1], [5.8]], True),
    ([[2.1, 4.7], [2.1, 2.1]], False),
    ([[2.1, 2., 2.], [4.7, 3, 5], [5.1, 2, 5]], False),
    ([[2.1], [4.7], [5.1], [5.8]], False),
]


@pytest.mark.parametrize("operand, sparse_output", Matrices)
def test_eye_like(operand, sparse_output, device_id, precision):
    np_eye_like = lambda matrix: np.eye(matrix.shape[0], matrix.shape[1], dtype=np.float32)
    operand = AA(operand).astype(np.float32)
    expected = np_eye_like(operand)
    expected_grad = np.zeros_like(operand).reshape(expected.shape)

    my_eval = (lambda f, arg: f.eval(arg).todense()) if sparse_output else (lambda f, arg: f.eval(arg))

    from .. import eye_like
    import cntk as C

    #testing with direct numpy input
    y =  C.eye_like(operand, sparse_output=sparse_output)
    actual = y.eval().todense() if sparse_output else y.eval()
    np.testing.assert_almost_equal(actual, expected)

    #testing through input_variable
    #test load and save:
    import tempfile
    import os
    x = C.input_variable(operand.shape[1:], dtype=np.float32, needs_gradient=True)
    cntk_eye_like = C.eye_like(x, sparse_output=sparse_output)
    actual = my_eval(cntk_eye_like, {x: operand})
    grad = cntk_eye_like.grad({x: operand})
    np.testing.assert_almost_equal(actual, expected)
    np.testing.assert_almost_equal(grad, expected_grad)
    tempdir = os.path.join(tempfile.gettempdir(), 'eye_like_test')
    cntk_eye_like.save(tempdir)
    cntk_eye_like2 = C.load_model(tempdir)
    np.testing.assert_almost_equal(my_eval(cntk_eye_like2, {cntk_eye_like2.arguments[0]: operand}), expected)
    os.remove(tempdir)

    cntk_eye_like = C.eye_like(C.unpack_batch(x), sparse_output=sparse_output)
    actual = my_eval(cntk_eye_like, {x: operand})
    grad = cntk_eye_like.grad({x: operand})
    np.testing.assert_almost_equal(actual, expected)
    np.testing.assert_almost_equal(grad, expected_grad)
    tempdir = os.path.join(tempfile.gettempdir(), 'eye_like_test2')
    cntk_eye_like.save(tempdir)
    cntk_eye_like2 = C.load_model(tempdir)
    np.testing.assert_almost_equal(my_eval(cntk_eye_like2, {cntk_eye_like2.arguments[0]: operand}), expected)
    os.remove(tempdir)

    cntk_eye_like = C.eye_like(C.transpose(C.unpack_batch(x), (1,0)), sparse_output=sparse_output)
    actual = my_eval(cntk_eye_like, {x: operand})
    grad = cntk_eye_like.grad({x: operand})
    np.testing.assert_almost_equal(actual, expected.transpose())
    np.testing.assert_almost_equal(grad, expected_grad)
    tempdir = os.path.join(tempfile.gettempdir(), 'eye_like_test3')
    cntk_eye_like.save(tempdir)
    cntk_eye_like2 = C.load_model(tempdir)
    np.testing.assert_almost_equal(my_eval(cntk_eye_like2, {cntk_eye_like2.arguments[0]: operand}), expected.transpose())
    os.remove(tempdir)

    #test expecting exception with sequence axis
    with pytest.raises(Exception) as info:
        #no sequence axis is allowed
        x = C.sequence.input_variable(operand.shape[1:], dtype=np.float32, needs_gradient=True)
        cntk_eye_like = C.eye_like(x, sparse_output=sparse_output)

    with pytest.raises(Exception) as info:
        #no more than 2 axes is allowed (including any dynamic axes)
        x = C.input_variable((3, 3), dtype=np.float32, needs_gradient=True)
        cntk_eye_like = C.eye_like(x, sparse_output=sparse_output)

    with pytest.raises(Exception) as info:
        #no less than 2 axes is allowed (including any dynamic axes)
        x = C.input_variable((), dtype=np.float32, needs_gradient=True)
        cntk_eye_like = C.eye_like(x, sparse_output=sparse_output)

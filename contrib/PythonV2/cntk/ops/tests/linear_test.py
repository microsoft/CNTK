# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for linear algebra operations, each operation is tested for 
the forward and the backward pass
"""

from __future__ import division
import numpy as np
import pytest
from .ops_test_utils import unittest_helper, _test_unary_op, _test_binary_op, AA, I, precision, PRECISION_TO_TYPE, batch_dense_to_sparse, left_matrix_type, right_matrix_type
from ...utils import sanitize_dtype_cntk
from ...context import get_context

# TODO: Test plus(), times(), etc, not only the overloaded opeartors (+,
# *, etc.)

# TODO: perhaps include some rand() testing; and
TENSOR_PAIRS = [
    ([30.], [10.]),
    ([[30.]], [[10.]]),
    ([[1.5, 2.1]], [[10., 20.]]),
    ([[100., 200.], [300., 400.], [10., 20.]],
      [[10., 20.], [30., 40.], [1., 2.]]),

    #([[5],[6],[7]], [[10, 20], [30,40], [1,2]]),     
    
    # Adding two 3x2 inputs of sequence length 1
    ([[30.,40.], [1.,2.], [0.1, 0.2]], [[10,20], [3,4], [-0.5, -0.4]]),
]

# -- plus operation tests --

@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_plus(left_operand, right_operand, device_id, precision):
    ctx = get_context()
    ctx.precision = precision
    ctx.device = device_id

    expected_forward = [AA([left_operand]) + AA([right_operand])]

    expected_backward = {
            'left_arg':  [[[np.ones_like(x, dtype=ctx.precision_numpy) for x in left_operand]]],
            'right_arg': [[[np.ones_like(x, dtype=ctx.precision_numpy) for x in right_operand]]]
            }
    from .. import plus
    _test_binary_op(ctx, plus,
            left_operand, right_operand, 
            expected_forward, expected_backward)

    _test_binary_op(ctx, '+',
            left_operand, right_operand, 
            expected_forward, expected_backward)

SEQ_TENSOR_PAIRS = [
    # two inputs each having sequences of length 1 and 2
    ([[[30.]], [[40], [50]]],  # first batch with two sequences
     [[[ 3.]], [[ 4], [ 5]]]), # second batch with two sequences

    ([[[30.,   0]], [[40,   1], [50,   2]]],  # first batch with two sequences
     [[[ 3., -10]], [[ 4, -20], [ 5, -30]]]), # second batch with two sequences
]
@pytest.mark.parametrize("left_batch, right_batch", SEQ_TENSOR_PAIRS)
def _test_op_plus_var_sequences_input_input(left_batch, right_batch, device_id, precision):
    ctx = get_context()
    ctx.precision = precision
    ctx.device = device_id

    assert len(left_batch) == len(right_batch)
    expected_forward = [AA([left_batch[i]]) + AA([right_batch[i]]) \
            for i in range(len(left_batch))]

    def ones_like(batch):
        expected_backward = []
        for sample in batch:
            seq = [np.ones_like(elem, dtype=ctx.precision_numpy) for elem in sample]
            expected_backward.append(seq)
        return expected_backward

    expected_backward = { 'left': ones_like(left_batch), 'right': ones_like(right_batch) }

    left_value = [AA(sample, dtype=ctx.precision_numpy) for sample in left_batch]
    left_shape = left_value[0][0].shape
    right_value = [AA(sample, dtype=ctx.precision_numpy) for sample in right_batch]
    right_shape = right_value[0][0].shape

    a = I(shape=left_shape,
            data_type=sanitize_dtype_cntk(ctx.precision_numpy),
            needs_gradient=True,
            name='a')

    b = I(shape=right_shape,
            data_type=sanitize_dtype_cntk(ctx.precision_numpy),
            needs_gradient=True,
            name='b')

    input_op_input = a + b
    forward_input = {a:left_value, b:right_value}
    output_shape = input_op_input.Output().Shape().Dimensions()
    backward_input = { a: expected_backward['left'], b: expected_backward['right'], }
    expected_backward = { a: expected_backward['left'], b: expected_backward['right'], }
    unittest_helper(input_op_input, 
        forward_input, expected_forward, 
        backward_input, expected_backward,
        device_id=ctx.device, precision=ctx.precision, clean_up=True)

# -- minus operation tests --
#TODO: enable once the function is exposed
@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def _test_op_minus(left_operand, right_operand, device_id, precision):
    ctx = get_context()
    ctx.precision = precision
    ctx.device = device_id

    expected_forward = [AA([left_operand]) + AA([right_operand])]

    expected_backward = {
            'left_arg':  [[[np.ones_like(x, dtype=ctx.precision_numpy) for x in left_operand]]],
            'right_arg': [[[np.ones_like(x, dtype=ctx.precision_numpy) for x in right_operand]]]
            }

    from .. import minus
    _test_binary_op(ctx, minus,
            left_operand, right_operand, 
            expected_forward, expected_backward)
    _test_binary_op(ctx, '-',
            left_operand, right_operand, 
            expected_forward, expected_backward)

# -- element times tests --

@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def test_op_element_times(left_operand, right_operand, device_id, precision):
    ctx = get_context()
    ctx.precision = precision
    ctx.device = device_id

    expected_forward = [AA([left_operand]) * AA([right_operand])]

    expected_backward = {
            'left_arg':  [[right_operand]],
            'right_arg': [[left_operand]]
            }
    
    from .. import element_times
    _test_binary_op(ctx, element_times,
            left_operand, right_operand, 
            expected_forward, expected_backward)
    _test_binary_op(ctx, '*',
            left_operand, right_operand, 
            expected_forward, expected_backward)


# -- element divide tests --
#TODO: enable once the function is exposed
@pytest.mark.parametrize("left_operand, right_operand", TENSOR_PAIRS)
def _test_op_element_divide(left_operand, right_operand, device_id, precision):
    ctx = get_context()
    ctx.precision = precision
    ctx.device = device_id

    expected_forward = [AA([left_operand]) / AA([right_operand])]

    expected_backward = {
            'left_arg':  [[[np.ones_like(x) / x for x in right_operand]]],
            'right_arg': [[-AA(left_operand, dtype=PRECISION_TO_TYPE[precision]) / AA(right_operand, dtype=PRECISION_TO_TYPE[precision])**2]]
            }

    from .. import element_divide
    _test_binary_op(ctx, element_divide,
            left_operand, right_operand, 
            expected_forward, expected_backward)
    _test_binary_op(ctx, '/',
            left_operand, right_operand, 
            expected_forward, expected_backward)

# -- times function tests --

TIMES_PAIRS = [
    ([[30.]], [[10.]]),
    ([[1.5, 2.1]], [[10.], [20.]]),
    ([[100., 200.]], [[10.], [20.]]),
]

# TODO: port to v2
#TODO:enable this test once sparse is sorted out
@pytest.mark.parametrize("left_operand, right_operand", TIMES_PAIRS)
def _test_op_times(left_operand, right_operand, device_id, precision,
        left_matrix_type, right_matrix_type):
    if right_matrix_type == 'sparse':
        pytest.skip('second operator of times() has to be dense')

    dt = PRECISION_TO_TYPE[precision]
    # Forward pass test
    #==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample
    expected = [[np.dot(AA(left_operand, dtype=dt), AA(right_operand, dtype=dt))]]
    
    if left_matrix_type == 'sparse':
        a = SI(*batch_dense_to_sparse([left_operand]))
    else:
        a = I([left_operand])

    b = I([right_operand])

    from cntk.ops import times, constant
    left_as_input = times(a, constant(right_operand))
    right_as_input = times(constant(left_operand), b)

    unittest_helper(left_as_input, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)

    unittest_helper(right_as_input, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)

    unittest_helper(times(a, b), None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=False)


    # Backward pass test
    #==================

    def op_grad(A, B):
        '''
        Compute derivative of A with respect to B. For simplicity, assume A
        and B to be matrices.
        Let A be 2x2 and B be 2x1, then we have
        [a11 a12] [b11]  = [ a11 b11 + a12 b21 ]
        [a21 a22] [b21]    [ a21 b11 + a22 b21 ]

        The derivative for A with respect to B is
        [b11 b21]
        [b11 b21]

        The derivative for B with respect to A:
        [a11 + a12]
        [a21 + a22]
        '''
        assert len(A.shape) == len(B.shape) == 2
        D = np.zeros_like(A)
        D[:,:] = B.sum(axis=1)
        
        return D

    if 'sparse' not in [left_matrix_type, right_matrix_type]:
        # FIXME: disabling until the Pass node supports sparse 
        expected_left = [[op_grad(AA(left_operand, dtype=dt), AA(right_operand, dtype=dt))]]
        expected_right = [[op_grad(AA(right_operand, dtype=dt).T, AA(left_operand, dtype=dt).T).T]]

        unittest_helper(left_as_input, None, expected_left, device_id=device_id,
                        precision=precision, clean_up=True, backward_pass=True, input_node=a)
        # BUG: Fails because of Pass node?
        unittest_helper(right_as_input, None, expected_right, device_id=device_id,
                        precision=precision, clean_up=True, backward_pass=True, input_node=b)

# -- identity function tests --
# TODO enable this once the function is exposed
IDENTITY_TENSORS = [
    ([30.]),
    ([[30.]]),
    ([[1.5, 2.1]]),
    ([[100., 200.], [300., 400.], [10., 20.]]),
    ([[30,40], [1,2], [0.1, 0.2]])
]

@pytest.mark.parametrize("tensor", IDENTITY_TENSORS)
def _test_op_identity(tensor, device_id, precision):
    ctx = get_context()
    ctx.precision = precision
    ctx.device = device_id

    expected_forward = [AA([tensor])]

    expected_backward = {
            'arg': np.ones_like(expected_forward),            
            }

    _test_unary_op(ctx, identity,
            tensor, expected_forward, expected_backward)

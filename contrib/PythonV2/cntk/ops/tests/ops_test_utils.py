# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Utils for operations unit tests
"""

import numpy as np
import pytest

from cntk.tests.test_utils import *

from ...context import get_new_context
from ...reader import *
from ...utils import sanitize_dtype_cntk
from .. import constant, input

I = input

@pytest.fixture(params=["dense", "sparse"])
def left_matrix_type(request):
    return request.param

@pytest.fixture(params=["dense", "sparse"])
def right_matrix_type(request):
    return request.param

def _test_unary_op(ctx, op_func,
        operand, expected_forward, expected_backward_all):
    
    value = AA(operand, dtype=ctx.precision_numpy) 
    
    a = I(shape=value.shape,
            data_type=sanitize_dtype_cntk(ctx.precision_numpy),
            needs_gradient=True,
            name='a')

    # create batch
    value.shape = (1,1) + value.shape    

    input_op = op_func(a)
    forward_input = {a:value}
    backward_input = {a:np.ones(value.shape)}
    expected_backward = { a: expected_backward_all['arg'], }
    unittest_helper(input_op, 
            forward_input, expected_forward, 
            backward_input, expected_backward,
            device_id=ctx.device, precision=ctx.precision, clean_up=True)
   
def _test_binary_op(ctx, op_func,
        left_operand, right_operand, 
        expected_forward, expected_backward_all):
    
    left_value = AA(left_operand, dtype=ctx.precision_numpy) 
    right_value = AA(right_operand, dtype=ctx.precision_numpy)

    a = I(shape=left_value.shape,
            data_type=sanitize_dtype_cntk(ctx.precision_numpy),
            needs_gradient=True,
            name='a')

    b = I(shape=right_value.shape,
            data_type=sanitize_dtype_cntk(ctx.precision_numpy),
            needs_gradient=True,
            name='b')

    # create batch
    left_value.shape = (1,1) + left_value.shape
    right_value.shape = (1,1) + right_value.shape
    
    if (type(op_func) == str):
        input_op_constant = eval('a %s right_operand'%op_func)
        constant_op_input = eval('left_operand %s b'%op_func)
        input_op_input = eval('a %s b'%op_func)
    else:
        input_op_constant = op_func(a, right_operand)
        constant_op_input = op_func(left_operand, b)
        input_op_input = op_func(a, b)

    forward_input = {a:left_value}
    backward_input = {a:np.ones(left_value.shape)}
    expected_backward = { a: expected_backward_all['left_arg'], }
    unittest_helper(input_op_constant, 
            forward_input, expected_forward, 
            backward_input, expected_backward,
            device_id=ctx.device, precision=ctx.precision, clean_up=True)
        
    forward_input = {b:right_value}
    backward_input = {b:np.ones(right_value.shape)}
    expected_backward = { b: expected_backward_all['right_arg'], }
    unittest_helper(constant_op_input, 
            forward_input, expected_forward, 
            backward_input, expected_backward,
            device_id=ctx.device, precision=ctx.precision, clean_up=True) 
    
    forward_input = {a:left_value, b:right_value}
    backward_input = {a:np.ones(left_value.shape), b:np.ones(right_value.shape)}
    expected_backward = { a: expected_backward_all['left_arg'], b: expected_backward_all['right_arg'], }
    unittest_helper(input_op_input, 
        forward_input, expected_forward, 
        backward_input, expected_backward,
        device_id=ctx.device, precision=ctx.precision, clean_up=True)
   
def unittest_helper(root_node, 
        forward_input, expected_forward, 
        backward_input, expected_backward,
        device_id=-1, precision="float", clean_up=True):

    from cntk.context import get_new_context
    with get_new_context() as ctx:
        ctx.clean_up = clean_up
        ctx.device_id = device_id
        ctx.precision = precision
        assert not ctx.input_nodes
        forward, backward = ctx.eval(root_node, forward_input, backward_input)

        # for forward we always expect only one result
        assert len(forward)==1
        forward = list(forward.values())[0]
        
        for res, exp in zip(forward, expected_forward):
            assert np.allclose(res, exp, atol=TOLERANCE_ABSOLUTE)
            assert res.shape == AA(exp).shape

        if backward_input:                                
            for key in expected_backward:
                res, exp = backward[key], expected_backward[key]
                assert np.allclose(res, exp, atol=TOLERANCE_ABSOLUTE)
                assert res.shape == AA(exp).shape

def batch_dense_to_sparse(batch, dynamic_axis=''):
    '''
    Helper test function that converts a batch of dense tensors into sparse
    representation that can be consumed by :func:`cntk.ops.sparse_input_numpy`.

    Args:
        batch (list): list of samples. If `dynamic_axis` is given, samples are sequences of tensors. Otherwise, they are simple tensors.
        dynamic_axis (str or :func:`cntk.ops.dynamic_axis` instance): the dynamic axis

    Returns:
        (indices, values, shape)
    '''

    batch_indices = []
    batch_values = []
    tensor_shape = []

    shapes_in_tensor = set()

    for tensor in batch:
        if isinstance(tensor, list):
            tensor = np.asarray(tensor)

        if dynamic_axis:
            # collecting the shapes ignoring the dynamic axis
            shapes_in_tensor.add(tensor.shape[1:])
        else:
            shapes_in_tensor.add(tensor.shape)

        if len(shapes_in_tensor) != 1:
            raise ValueError('except for the sequence dimensions all shapes ' +
                             'should be the same - instead we %s' %
                             (", ".join(str(s) for s in shapes_in_tensor)))

        t_indices = range(tensor.size)
        t_values = tensor.ravel(order='F')
        mask = t_values!=0

        batch_indices.append(list(np.asarray(t_indices)[mask]))
        batch_values.append(list(np.asarray(t_values)[mask]))

    return batch_indices, batch_values, shapes_in_tensor.pop()

def test_batch_dense_to_sparse_full():
    i, v, s = batch_dense_to_sparse(
            [
                [[1,2,3], [4,5,6]],
                [[10,20,30], [40,50,60]],
            ])
    assert i == [
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 5],
            ]
    assert v == [
            [1,4,2,5,3,6],
            [10,40,20,50,30,60]
            ]
    assert s == (2,3)
    
    i, v, s = batch_dense_to_sparse([[1]])
    assert i == [[0]]
    assert v == [[1]]
    assert s == (1,)

def test_batch_dense_to_sparse_zeros():
    i, v, s = batch_dense_to_sparse(
            [
                [[1,2,3], [4,0,6]],
                [[0,0,0], [40,50,60]],
            ])
    assert i == [
            [0, 1, 2, 4, 5],
            [1, 3, 5],
            ]
    assert v == [
            [1,4,2,3,6],
            [40,50,60]
            ]
    assert s == (2,3)
    



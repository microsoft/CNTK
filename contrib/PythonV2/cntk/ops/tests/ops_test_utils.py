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
from .. import constant, input_numpy, sparse_input_numpy


# Keeping things short
I = input_numpy
SI = sparse_input_numpy


# CNTK is column major and thus for broadcasting the axes are aligned to the left,
# however, in Numpy they are aligned to the write, therefore, in order to perform 
# CNTK's broadcast in Numpy we reverse the axes, performs the operation and 
# reverse back.
def _broadcast_col_major(tensor_a, tensor_b, func):
    rev_shape_a = tuple(reversed(tensor_a.shape))
    reshaped_a = np.reshape(tensor_a, rev_shape_a)    
    rev_shape_b = tuple(reversed(tensor_b.shape))
    reshaped_b = np.reshape(tensor_b, rev_shape_b)    
    res = func(reshaped_a, reshaped_b)
    rev_shape_res = tuple(reversed(res.shape))
    return np.reshape(res, rev_shape_res)    


# check whether a valid broadcast is possible between two tensors
# it also returns the axes ids along which we perform reducion when
# we call the backward pass of the node in question
def _check_broadcasting_and_get_reduce_axes(tensor_a, tensor_b):    
    axes_a=[]
    axes_b=[]
    # no broadcasting
    if (tensor_a.shape == tensor_b.shape):
        return (0,axes_a,axes_b) 
    else:
        for i in range(min(len(tensor_a.shape),len(tensor_b.shape))):            
            if (tensor_a.shape[i] != tensor_b.shape[i]):  
                #invalid broadcasting
                if (tensor_a.shape[i]!=1 and tensor_b.shape[i]!=1):
                    return (-1, None,None)
                    
                if (tensor_a.shape[i]==1):
                    axes_a.append(i)
                if (tensor_b.shape[i]==1):
                    axes_b.append(i)
        
        # process the remaining axes        
        if (len(tensor_a.shape)) > (len(tensor_b.shape)):
            for i in range(len(tensor_b.shape), len(tensor_a.shape)):
                axes_b.append(i)
        else:
            for i in range(len(tensor_a.shape), len(tensor_b.shape)):
                axes_a.append(i)                            
            
    return (1,axes_a,axes_b)

def _reduce_sum_on_multiple_axes(tensor, axes):        
    res = tensor
    axis_shift = 0
    for a in axes:        
        res = np.add.reduce(res, a+axis_shift)        
        axis_shift-=1
        
    return res
    

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
    

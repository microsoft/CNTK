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
from .. import constant, input_numpy


# Keeping things short
I = input_numpy


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
    
a = np.asarray([[[1,2,3]],[[4,5,6]]])
print (a.shape)
b = np.asarray([[7,8]])
print (b.shape)
c = _broadcast_col_major(a,b,np.add)
print (c)
print (c.shape)

flag, axes_a, axes_b = _check_broadcasting_and_get_reduce_axes(a,b)
print (axes_a)
print (axes_b)
if flag == 1:
    print (_reduce_sum_on_multiple_axes(np.ones_like(c),axes_a))
    print (_reduce_sum_on_multiple_axes(np.ones_like(c),axes_b))
    
# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for non-linear operations. Each operation is tested for 
the forward and the backward pass
"""

import numpy as np
import pytest
from .ops_test_utils import unittest_helper, AA, I, precision, PRECISION_TO_TYPE
from ...graph import *
from ...reader import *
from .. import constant

EPS_IN_LOG = 1e-37        # 1e-37 is the highest guaranteed precision
BACKWARD_RESULST_FOR_LOG_EPS = 9.08782e+36 # the backward result returned by CNTK log() for epsilon
LOG_OF_EPS_IN_LOG =  -85.1 # log(EPS_IN_LOG)

CLIP_TUPLES = [
    ([1.0], [2.0], [1.5]), # value shouldn't be clipped; gradient is [1.0]
    ([1.0], [2.0], [0.5]), # value should be clipped to 1.0; gradient is [0.0]
    ([1.0], [2.0], [2.5]), # value should be clipped to 2.0; gradient is [0.0]
    
    # should clip to [1.5, 2.0, 1.0]; gradient is [[1.0, 0.0, 0.0]]
    ([1.0], [2.0], [[1.5, 2.1, 0.9]]),

    # should clip to [[1.0, 2.0], [1.0, 2.0], [1.5, 2.0]];
    # gradient is [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]]
    ([1.0], [2.0], [[0.0, 3.0], [1.0, 2.0], [1.5, 2.5]]),
     
    # test what happens if a user puts a higher "min" value than their "max" value
    # should clip to [[5.0, 5.0, 5.0, 5.0, 5.0]] because min is evaluated first
    # gradient should be all zeros: [[0.0, 0.0, 0.0, 0.0, 0.0]]
    ([5.0], [0.5], [[1.5, 2.1, 0.9, -1.0, -2.0]]),
     
    # test a more complicated broadcasting scenario
    ([[1.5, 2.0], [2.5, 3.0]], [[-2.0, 2.5], [2.5, 3.5]], [[-1.0, 2.0], [3.0, 4.0]]),
    ]

# -- clip operation tests --
@pytest.mark.parametrize("min_value, max_value, x", CLIP_TUPLES)
def test_op_clip(min_value, max_value, x, device_id, precision):    
    from .. import clip
    
    # Forward pass test
    #==================
    # we compute the expected output for the forward pass
    # Compare to numpy's implementation of np.clip(x, min, max)
    expected = [[np.clip(AA(x, dtype=PRECISION_TO_TYPE[precision]), AA(min_value, dtype=PRECISION_TO_TYPE[precision]), AA(max_value, dtype=PRECISION_TO_TYPE[precision]))]]
    
    op_node = I([x])
    a = constant(min_value)    
    b = constant(max_value)
    
    result = clip(op_node, a, b)
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=False)
    
    #Backward pass test
    #==================
    # The gradient of the clip() function is equal to 1 when the element 
    # has not been clipped, and 0 if it has been clipped
    expected = [[np.array(np.logical_not(np.logical_or(np.greater(x, max_value), np.less(x, min_value))), dtype=PRECISION_TO_TYPE[precision])]]

    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=True,
                    input_node=op_node)

TENSORS = [
    ([[0, -0.1]]),
    ([[-100, -10], [-1, -0.1], [-0.01, -0.001],
      [0.001, 0.01], [0.1, 1], [10, 100]]),
]

@pytest.mark.parametrize("tensor", TENSORS)
def test_op_sigmoid(tensor, device_id, precision):

    from .. import sigmoid

    def numpy_op(x):
        return 1.0 / (1.0 + np.exp(-AA(x, dtype=PRECISION_TO_TYPE[precision])))

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample

    expected = [[numpy_op(tensor)]]

    input_node = I([tensor])
    op_node = sigmoid(input_node)

    unittest_helper(op_node, None, expected,
                    device_id=device_id,
                    precision=precision,
                    clean_up=True, backward_pass=False)

    # Backward pass test
    # ==================
    # The expected results for the backward pass is sigmoid(x)*(1-sigmoid(x))
    s = numpy_op(tensor)
    expected = [[s * (1 - s)]]

    unittest_helper(op_node, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True,
                    input_node=input_node)


@pytest.mark.parametrize("batch",
                         [
                             [  # 2 samples having 4 classes
                                 [1, 1, 2, 3],
                                 [0, 0, 0, 0]
                             ],
                         ])
def test_op_softmax(batch, device_id, precision):
    from .. import softmax
    
    def numpy_op(x):
        x = AA(x, dtype=PRECISION_TO_TYPE[precision])
        # Expecting classes of one sample
        assert len(x.shape) == 1

        ox = x - x.max()  # subtract max to avoid overflow

        expX = np.exp(ox)
        return expX / np.sum(expX)

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample

    input_node = I(batch)
    op_node = softmax(input_node)

    expected = [[numpy_op(sample)] for sample in batch]
    unittest_helper(op_node, None, expected,
                    device_id=device_id,
                    precision=precision,
                    clean_up=True, backward_pass=False)

    # Backward pass test
    # ==================
    # The expected results for the backward pass is fi(1-fi) for i and -fi*fj
    # for element j!=i.
    def numpy_grad(x):
        grads = np.zeros((len(x), len(x)), dtype=PRECISION_TO_TYPE[precision])

        for i in range(len(x)):
            # deriving wrt i-th element
            for j in range(len(x)):
                if i == j:
                    grads[i, j] = x[i] * (1 - x[i])
                else:
                    grads[i, j] = x[i] * (-x[j])

        return grads.sum(axis=0)

    expected = [[numpy_grad(numpy_op(sample))] for sample in batch]

    unittest_helper(op_node, None, expected,
                    device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True,
                    input_node=input_node)


@pytest.mark.parametrize("tensor", TENSORS)
def test_op_exp(tensor, device_id, precision):
    from .. import exp

    def numpy_op(x):
        return np.exp(AA(x, dtype=PRECISION_TO_TYPE[precision]))

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample

    expected = [[numpy_op(tensor)]]

    input_node = I([tensor])
    op_node = exp(input_node)

    unittest_helper(op_node, None, expected,
                    device_id=device_id,
                    precision=precision,
                    clean_up=True, backward_pass=False)

    # Backward pass test
    # ==================
    # The expected results for the backward pass is exp()
    expected = [[numpy_op(tensor)]]

    unittest_helper(op_node, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True,
                    input_node=input_node)

@pytest.mark.parametrize("tensor", TENSORS)
def test_op_log(tensor, device_id, precision):

    from .. import log

    def numpy_op(x):
        a = np.log(AA(x, dtype=PRECISION_TO_TYPE[precision]))
        # CNTK returns -85.1 for log(x) if x is negative or zero.
        # CNTK uses 1e-37f as the smallest float number for log
        # becuase this is the only guaranteed precision accross platforms
        # something to change in CNTK and perhapas return some standard symbols
        # like (nan and -inf) for numpy
        a[np.isnan(a)] = LOG_OF_EPS_IN_LOG
        a[np.isneginf(a)] = LOG_OF_EPS_IN_LOG       
        return a

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample

    expected = [[numpy_op(tensor)]]

    input_node = I([tensor])
    op_node = log(input_node)

    unittest_helper(op_node, None, expected,
                    device_id=device_id,
                    precision=precision,
                    clean_up=True, backward_pass=False)


    def numpy_op_grad(x):
        a = np.divide(1, AA(x, dtype=PRECISION_TO_TYPE[precision]))
        a[np.isinf(a)] = BACKWARD_RESULST_FOR_LOG_EPS
        a[a<=0] = BACKWARD_RESULST_FOR_LOG_EPS
        return a

    # Backward pass test
    # ==================
    # The expected results for the backward pass is 1/x
    expected = [[numpy_op_grad(tensor)]]

    unittest_helper(op_node, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True,
                    input_node=input_node)

@pytest.mark.parametrize("tensor", TENSORS)
def test_op_sqrt(tensor, device_id, precision):
    from .. import sqrt

    def numpy_op(x):
        a = np.sqrt(AA(x, dtype=PRECISION_TO_TYPE[precision]))
        # CNTK returns zero for sqrt of negative nubmers, perhaps it should be
        # changed to return something linke nan for numpy
        a[np.isnan(a)] = 0
        return a

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample

    expected = [[numpy_op(tensor)]]

    input_node = I([tensor])
    op_node = sqrt(input_node)

    unittest_helper(op_node, None, expected,
                    device_id=device_id,
                    precision=precision,
                    clean_up=True, backward_pass=False)


    def numpy_op_grad(x):
        return np.divide(0.5, numpy_op(AA(x, dtype=PRECISION_TO_TYPE[precision])))

    # Backward pass test
    # ==================
    # The expected results for the backward pass is 0.5/sqrt(x)
    expected = [[numpy_op_grad(tensor)]]

    unittest_helper(op_node, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True,
                    input_node=input_node)

@pytest.mark.parametrize("tensor", TENSORS)
def test_op_square(tensor, device_id, precision):
    from .. import square

    def numpy_op(x):
        return np.square(AA(x, dtype=PRECISION_TO_TYPE[precision]))

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample

    expected = [[numpy_op(tensor)]]

    input_node = I([tensor])
    op_node = square(input_node)

    unittest_helper(op_node, None, expected,
                    device_id=device_id,
                    precision=precision,
                    clean_up=True, backward_pass=False)


    def numpy_op_grad(x):
        return np.multiply(2, AA(x, dtype=PRECISION_TO_TYPE[precision]))

    # Backward pass test
    # ==================
    # The expected results for the backward pass is 2x
    expected = [[numpy_op_grad(tensor)]]

    unittest_helper(op_node, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True,
                    input_node=input_node)

@pytest.mark.parametrize("tensor", TENSORS)
def test_op_tanh(tensor, device_id, precision):

    from .. import tanh

    def numpy_op(x):
        return np.tanh(AA(x, dtype=PRECISION_TO_TYPE[precision]))

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample

    expected = [[numpy_op(tensor)]]

    input_node = I([tensor])
    op_node = tanh(input_node)

    unittest_helper(op_node, None, expected,
                    device_id=device_id,
                    precision=precision,
                    clean_up=True, backward_pass=False)

    # Backward pass test
    # ==================
    # The expected results for the backward pass is 1-tanh(x)^2
    expected = [[1 - numpy_op(tensor)**2]]

    unittest_helper(op_node, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True,
                    input_node=input_node)


@pytest.mark.parametrize("tensor", TENSORS)
def test_op_relu(tensor, device_id, precision):

    from .. import relu

    def numpy_op(x):
        npx = AA(x, dtype=PRECISION_TO_TYPE[precision])
        return np.maximum(np.zeros_like(npx), npx)

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample

    expected = [[numpy_op(tensor)]]

    input_node = I([tensor])
    op_node = relu(input_node)

    unittest_helper(op_node, None, expected,
                    device_id=device_id,
                    precision=precision,
                    clean_up=True, backward_pass=False)

    # Backward pass test
    # ==================
    # The expected results for the backward pass is 1 whenever the value is
    # positive
    def numpy_op_grad(x):
        npx = AA(x, dtype=PRECISION_TO_TYPE[precision])
        return np.asarray(npx > 0, dtype=int)

    expected = [[numpy_op_grad(tensor)]]

    unittest_helper(op_node, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True,
                    input_node=input_node)


@pytest.mark.parametrize("tensor", TENSORS)
def test_op_abs(tensor, device_id, precision):
    from .. import abs
    np_tensor = AA(tensor, dtype=PRECISION_TO_TYPE[precision])

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have dynamic_axis='')
    # the second for batch of one sample

    expected = [[np.abs(tensor)]]

    input_node = I([tensor])
    op_node = abs(input_node)

    unittest_helper(op_node, None, expected,
                    device_id=device_id,
                    precision=precision,
                    clean_up=True, backward_pass=False)

    # Backward pass test
    # ==================
    # The expected results for the backward pass is x/|x|

    expected = np_tensor / np.abs(np_tensor)
    # For 0 NumPy gives a gradient non, while CNTK gives 0
    expected[np.isnan(expected)] = 0
    expected = [[expected]]

    unittest_helper(op_node, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True,
                    input_node=input_node)


COND_TUPLES = [ 
                ([-1], [2], [3]), 
                ([0], [20], [30]),
                ([10],[0],[-100]),
              ]
  
# -- cond operation tests --
@pytest.mark.parametrize("flag, value_a, value_b", COND_TUPLES)
def test_op_cond(flag, value_a, value_b, device_id, precision):    

    from .. import cond

    #Forward pass test
    #==================
    # Comparing to numpy's implementation of where(...)

    expected = [[[np.where(AA(flag, dtype=PRECISION_TO_TYPE[precision]), AA(value_a, dtype=PRECISION_TO_TYPE[precision]), AA(value_b, dtype=PRECISION_TO_TYPE[precision]))]]]

    cond_as_const    = constant([flag])
    value_a_as_const = constant([value_a])    
    value_b_as_const = constant([value_b])   

    cond_as_input    = I([flag])
    value_a_as_input = I([value_a])
    value_b_as_input = I([value_b])

    result = cond(cond_as_input, value_a_as_const, value_b_as_const)
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=False)

    #Backward pass test
    #==================
    # The derivative of the cond() function is zero for the first argument.
    # The derivative for second and thrird argument depends on the first:
    # * Derivative of second argument = derivative of input if cond else 0
    # * Derivative of third argument  = derivative of input if not cond else 0

    # Derivative for first parameter should always be zero
    expected  = [[[np.zeros_like(x) for x in flag]]]
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=True, input_node=cond_as_input)

    # Derivative of second parameter depends on cond
    expected = [[np.array(np.where(flag, 1, 0), dtype=PRECISION_TO_TYPE[precision])]]
    result = cond(cond_as_const, value_a_as_input, value_b_as_const)
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=True, input_node=value_a_as_input)

    # Derivative of third parameter depends on cond
    expected = [[np.array(np.where(flag, 0, 1), dtype=PRECISION_TO_TYPE[precision])]]
    result = cond(cond_as_const, value_a_as_const, value_b_as_input)
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=True, input_node=value_b_as_input)

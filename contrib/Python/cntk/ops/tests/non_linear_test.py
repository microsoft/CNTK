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
from .ops_test_utils import unittest_helper, C, AA, I, precision, PRECISION_TO_TYPE
from ...graph import *
from ...reader import *
from ..non_linear import clip, exp, rectified_linear, sigmoid, softmax, tanh

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

    #Forward pass test
    #==================
    # we compute the expected output for the forward pass
    # Compare to numpy's implementation of np.clip(x, min, max)
    expected = [[np.clip(AA(x, dtype=PRECISION_TO_TYPE[precision]), AA(min_value, dtype=PRECISION_TO_TYPE[precision]), AA(max_value, dtype=PRECISION_TO_TYPE[precision]))]]
    
    a = C(min_value)    
    b = C(max_value)
    c = I([x], has_sequence_dimension=False)
    
    result = clip(a, b, c)
    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=False)
    
    #Backward pass test
    #==================
    # The gradient of the clip() function is equal to 1 when the element 
    # has not been clipped, and 0 if it has been clipped
    expected = [[np.array(np.logical_not(np.logical_or(np.greater(x, max_value), np.less(x, min_value))), dtype=PRECISION_TO_TYPE[precision])]]

    unittest_helper(result, None, expected, device_id=device_id, 
                    precision=precision, clean_up=True, backward_pass=True, input_node=c)

TENSORS = [
    ([[0, -0.1]]),
    ([[-100, -10], [-1, -0.1], [-0.01, -0.001],
      [0.001, 0.01], [0.1, 1], [10, 100]]),
]

@pytest.mark.parametrize("tensor", TENSORS)
def test_op_sigmoid(tensor, device_id, precision):

    def numpy_op(x):
        return 1.0 / (1.0 + np.exp(-AA(x, dtype=PRECISION_TO_TYPE[precision])))

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have has_sequence_dimension=False)
    # the second for batch of one sample

    expected = [[numpy_op(tensor)]]

    input_node = I([tensor], has_sequence_dimension=False)
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
         [ # 2 samples having 4 classes
          [1,1,2,3],
          [0,0,0,0]
         ],
            ])
def test_op_softmax(batch, device_id, precision):

    def numpy_op(x):
        x = AA(x, dtype=PRECISION_TO_TYPE[precision])
        # Expecting classes of one sample 
        assert len(x.shape) == 1

        ox = x-x.max() # subtract max to avoid overflow

        expX = np.exp(ox)
        return expX / np.sum(expX)

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have has_sequence_dimension=False)
    # the second for batch of one sample


    input_node = I(batch, has_sequence_dimension=False)
    op_node = softmax(input_node)
    #from cntk.ops.cntk1 import CrossEntropyWithSoftmax
    #op_node = CrossEntropyWithSoftmax(I([[0,1],[0,1]], has_sequence_dimension=False), input_node)

    expected = [[numpy_op(sample)] for sample in batch]
    unittest_helper(op_node, None, expected,
                    device_id=device_id,
                    precision=precision,
                    clean_up=True, backward_pass=False)

    # Backward pass test
    # ==================
    # The expected results for the backward pass is 
    expected = [['tbd']]

    unittest_helper(op_node, None, expected, 
            device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True,
                    input_node=input_node)


@pytest.mark.parametrize("tensor", TENSORS)
def test_op_exp(tensor, device_id, precision):

    def numpy_op(x):
        return np.exp(AA(x, dtype=PRECISION_TO_TYPE[precision]))

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have has_sequence_dimension=False)
    # the second for batch of one sample

    expected = [[numpy_op(tensor)]]

    input_node = I([tensor], has_sequence_dimension=False)
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
def test_op_tanh(tensor, device_id, precision):

    def numpy_op(x):
        return np.tanh(AA(x, dtype=PRECISION_TO_TYPE[precision]))

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have has_sequence_dimension=False)
    # the second for batch of one sample

    expected = [[numpy_op(tensor)]]

    input_node = I([tensor], has_sequence_dimension=False)
    op_node = tanh(input_node)

    unittest_helper(op_node, None, expected,
                    device_id=device_id,
                    precision=precision,
                    clean_up=True, backward_pass=False)

    # Backward pass test
    # ==================
    # The expected results for the backward pass is 1-tanh(x)^2
    expected = [[1-numpy_op(tensor)**2]]

    unittest_helper(op_node, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True,
                    input_node=input_node)

@pytest.mark.parametrize("tensor", TENSORS)
def test_op_rectified_linear(tensor, device_id, precision):

    def numpy_op(x):
        npx = AA(x, dtype=PRECISION_TO_TYPE[precision])
        return np.maximum(np.zeros_like(npx), npx)

    # Forward pass test
    # ==================
    # we compute the expected output for the forward pass
    # we need two surrounding brackets
    # the first for sequences (length=1, since we have has_sequence_dimension=False)
    # the second for batch of one sample

    expected = [[numpy_op(tensor)]]

    input_node = I([tensor], has_sequence_dimension=False)
    op_node = rectified_linear(input_node)

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
        return np.asarray(npx>0, dtype=int)

    expected = [[numpy_op_grad(tensor)]]

    unittest_helper(op_node, None, expected, device_id=device_id,
                    precision=precision, clean_up=True, backward_pass=True,
                    input_node=input_node)

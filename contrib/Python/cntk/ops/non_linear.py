# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
<<<<<<< HEAD
Non-linear operations. For every operation we explain how the forward and backward
passes are computed. For the backward pass we just explain the scalar case which is the building 
block for computing tensor gradients using the chain rule. For tensors, the backward pass of a node 
is computed as follows : for each element in the output tensor, its gradient with respect to the
given input tensor is computed, then, the resulting tensors are added up.
"""

from cntk.ops.cntk1 import Clip, Exp, RectifiedLinear, Sigmoid, Softmax, Tanh

def clip(min_value, max_value, x, name=None):
    """
    Clips tensor values to fall between `min_value` and `max_value`.
    For the input tensor `x`, this node outputs a tensor of the same shape with 
    all of its values clipped to fall between `min_value` and `max_value`.
    The backward pass propagates the received gradient if no clipping occurred,
    and 0 if the value was clipped.
    
    Example:
        >>> clip(2., 4., [1., 2.1, 3.0, 4.1])
        #[2.0, 2.1, 3.0, 4.0]
        
        >>> clip([-5., -4., 0., 3., 5.], [5., 4., 1., 4., 9.], [-10., -5., 0., 5., 10.])
        #[-5, -4., 0., 4., 9.]
    
    Args:        
        min_value: the minimum value to clip element values to
        max_value: the maximum value to clip element values to
        x: tensor to be clipped
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    return Clip(min_value, max_value, x, var_name = name)

def rectified_linear(x, name=None):
    """
    computes the element-wise rectified linear of `x`: ``max(x, 0)``

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    return RectifiedLinear(x, var_name=name)


def sigmoid(x, name=None):
    """
    computes the element-wise sigmoid of `x`: 

    :math:`sigmoid(x) = {1 \over {1+\exp(-x)}}`

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    return Sigmoid(x, var_name=name)

def tanh(x, name=None):
    """
    computes the element-wise tanh of `x`: 

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    return Tanh(x, var_name=name)

def softmax(X, name=None):
    """
    computes the element-wise sigmoid of `X`: 

    :math:`softmax(x) = {\exp(x) - \max_{x \in X}(\exp(x)) \over {\sum_{x \in
    X} \exp(x)- \max_{x \in X}(\exp(x)) }}`

    The term :math:`\max_{x \in X}(\exp(x))` is subtracted for numerical
    stability.

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    return Softmax(X)#Exp(LogSoftmax(X), var_name=name)

def exp(x, name=None):
    """
    computes the element-wise exponential of `x`: 

    :math:`exp(x) = {e^x}`

    Args:
        x: any :class:`cntk.graph.ComputationNode` that outputs a tensor

    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    return Exp(x, var_name=name)

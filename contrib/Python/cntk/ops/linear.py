# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Linear algebra operations. For every operation we explain how the forward and backward
passes are computed. For the backward pass we just explain the scalar case which is the building 
block for computing tensor gradients using the chain rule. For tensors, the backward pass of a node 
is computed as follows : for each element in the output tensor, its gradient with respect to the
given input tensor is computed, then, the resulting tensors are added up.
"""

from cntk.ops.cntk1 import Times, Plus, Minus, ElementDivide, ElementTimes


def plus(left, right, name=None):
    """
    Tensor addition operation. The output of this operation is the sum of the 
    two input tensors. It supports broadcasting. In case of scalars its backward
    pass propagates the received gradient. 

    Args:
        left: left side tensor
        right: right side tensor
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """

    return Plus(left, right, var_name=name)


def minus(left, right, name=None):
    """
    Tensor subtraction operation. The output of this operation is left minus
    right tensor. It supports broadcasting. In case of scalars its backward
    pass propagates the received gradient. 

    Args:
        left: left side tensor
        right: right side tensor
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """

    return Minus(left, right, var_name=name)


def element_times(left, right, name=None):
    """
    Element-wise multiplication operation. The output of this operation is the
    element-wise product of the two input tensors. It supports broadcasting. In
    case of scalars its backward pass to left propagates right 
    times the received gradient and vice versa. 
    Args:
        left: left side tensor
        right: right side tensor
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """

    return ElementTimes(left, right, var_name=name)


def element_divide(left, right, name=None):
    """
    Element-wise division operation. The output of this operation is the
    element-wise division of the two input tensors. It supports broadcasting. In
    case of scalars its backward pass to left propagates 1/right 
    times the received gradient, and the backward pass to right propagates 
    (-left/right^2) times the received gradient. 

    Args:
        left: left side tensor
        right: right side tensor
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """

    return ElementDivide(left, right, var_name=name)


def times(left, right, name=None):
    """
    Tensor times operation. The output of this operation is the
    tensor product of the two input tensors. It supports broadcasting. In
    case of scalars its backward pass to left propagates right
    times the received gradient and vice versa.

    Args:
        left: left side tensor
        right: right side tensor
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """

    return Times(left, right, var_name=name)

# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root 
# for full license information.
# ==============================================================================

"""
Linear algebra operations
"""

from cntk.ops.cntk1 import Times, Plus, ElementDivide, ElementTimes

def plus(left_operand, right_operand, name=None):
    """
    Tensor addition operation. The output of this operation is the sum of the 
    two input tensors. It supports broadcasting. In case of scalars its backward
    pass returns 1. In case of tensors, the backward pass is computed as follows: 
    for each element in the output tensor, it computes the its gradient with 
    respect to the given input tensor, then it sums all the resulting tensors.
    
    Args:
        left_operand: left side tensor
        right_operand: right side tensor
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    return Plus(left_operand, right_operand, var_name = name)  

def element_times(left_operand, right_operand, name=None):
    """
    Element-wise multiplication operation. The output of this operation is the
    element-wise product of the two input tensors. It supports broadcasting. In
    case of scalars its backward pass to left_operand returns right_operand and
    vice versa. In case of tensors, the backward pass is computed as follows: 
    for each element in the output tensor, it computes the its gradient with
    respect to the given input tensor, then it sums all the resulting tensors.

    Args:
        left_operand: left side tensor
        right_operand: right side tensor
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    return ElementTimes(left_operand, right_operand, var_name = name)

def element_divide(left_operand, right_operand, name=None):
    """
    Element-wise division operation. The output of this operation is the
    element-wise division of the two input tensors. It supports broadcasting. In
    case of scalars its backward pass to left_operand returns 1/right_operand and
    the backward pass to right_operand returns (-left_operand/right_operand^2). 
    In case of tensors, the backward pass is computed as follows: 
    for each element in the output tensor, it computes the its gradient with
    respect to the given input tensor, then it sums all the resulting tensors.

    Args:
        left_operand: left side tensor
        right_operand: right side tensor
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    return ElementDivide(left_operand, right_operand, var_name = name)


def times(left_operand, right_operand, name=None):
    """
    Tensor times operation. The output of this operation is the
    tensor product of the two input tensors. It supports broadcasting. In
    case of scalars its backward pass to left_operand returns right_operand and
    vice versa. In case of tensors, the backward pass is computed as follows: 
    for each element in the output tensor, it computes the its gradient with
    respect to the given input tensor, then it sums all the resulting tensors.

    Args:
        left_operand: left side tensor
        right_operand: right side tensor
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    
    return Times(left_operand, right_operand, var_name = name)    
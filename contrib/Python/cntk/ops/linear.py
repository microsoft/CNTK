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
    tensor addition operation

    Args:
        left_operand: Left side tensor
        right_operand: Right side tensor
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    return Plus(left_operand, right_operand, var_name = name)  

def element_times(left_operand, right_operand, name=None):
    """
    element-wise multiplication operation

    Args:
        left_operand: Left side tensor
        right_operand: Right side tensor
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    return ElementDivide(left_operand, right_operand, var_name = name)

def element_divide(left_operand, right_operand, name=None):
    """
    element-wise division operation

    Args:
        left_operand: Left side tensor
        right_operand: Right side tensor
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    return ElementDivide(left_operand, right_operand, var_name = name)


def times(left_operand, right_operand, name=None):
    """
    tensor times operation

    Args:
        left_operand: Left side tensor
        right_operand: Right side tensor
        name: the name of the node in the network            
    Returns:
        :class:`cntk.graph.ComputationNode`
    """
    
    return Times(left_operand, right_operand, var_name = name)    
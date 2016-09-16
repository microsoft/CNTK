# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from ...utils import sanitize_input, sanitize_shape, get_data_type

################################################################################
# sequence ops
################################################################################
def is_first(operand, name = ''):
    '''
    TBA
        
    Example:
        TBA
    Args:        
        operand: the symbolic tensor operand denoting a sequence
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.Function`
    '''    
    from cntk import is_first
    operand = sanitize_input(operand, get_data_type(operand))
    return is_first(operand, name)

def is_last(operand, name = ''):
    '''
    TBA
        
    Example:
        TBA
    Args:        
        operand: the symbolic tensor operand denoting a sequence
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.Function`
    '''    
    from cntk import is_last
    operand = sanitize_input(operand, get_data_type(operand))
    return is_last(operand, name)

def first(operand, name = ''):
    '''
    TBA
        
    Example:
        TBA
    Args:        
        operand: the symbolic tensor operand denoting a sequence
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.Function`
    '''    
    from cntk import first
    operand = sanitize_input(operand, get_data_type(operand))
    return first(operand, name)

def last(operand, name = ''):
    '''
    TBA
        
    Example:
        TBA
    Args:        
        operand: the symbolic tensor operand denoting a sequence
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.Function`
    '''    
    from cntk import last
    operand = sanitize_input(operand, get_data_type(operand))
    return last(operand, name)

def where(condition, name = ''):
    '''
    TBA
        
    Example:
        TBA
    Args:        
        condition: the symbolic tensor operand denoting a boolean condition flag for each step of a sequence
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.Function`
    '''    
    from cntk import where
    condition = sanitize_input(condition, get_data_type(condition))
    return where(condition, name)

def gather(operand, condition, name = ''):
    '''
    TBA
        
    Example:
        TBA
    Args:        
        operand: the symbolic tensor operand denoting a sequence
        condition: the symbolic tensor operand denoting a boolean condition flag for each step of a sequence
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.Function`
    '''    
    from cntk import gather
    operand = sanitize_input(operand, get_data_type(operand))
    condition = sanitize_input(condition, get_data_type(condition))
    return gather(operand, condition, name)

def scatter(operand, condition, name = ''):
    '''
    TBA
        
    Example:
        TBA
    Args:        
        operand: the symbolic tensor operand denoting a sequence
        condition: the symbolic tensor operand denoting a boolean condition flag for each step of a sequence
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.Function`
    '''    
    from cntk import scatter
    operand = sanitize_input(operand, get_data_type(operand))
    condition = sanitize_input(condition, get_data_type(condition))
    return scatter(operand, condition, name)

def broadcast_as(operand, broadcast_as_operand, name = ''):
    '''
    TBA
        
    Example:
        TBA
    Args:        
        operand: the symbolic tensor operand denoting a tensor
        broadcast_as_operand: the symbolic tensor operand denoting a sequence per whose layout the main operand id to be broadcast
        name (str): the name of the node in the network
    Returns:
        :class:`cntk.Function`
    '''    
    from cntk import broadcast_as
    operand = sanitize_input(operand, get_data_type(operand))
    broadcast_as_operand = sanitize_input(broadcast_as_operand, get_data_type(broadcast_as_operand))
    return broadcast_as(operand, broadcast_as_operand, name)

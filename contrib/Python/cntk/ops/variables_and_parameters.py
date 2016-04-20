# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root 
# for full license information.
# ==============================================================================


"""
This modules contains the interfaces to Inputs and Parameters. 
"""

from cntk.ops.cntk1 import Input
from ..graph import input_reader 
from ..graph import parameter as param

def input_array(value, has_sequence_dimension=True, name=None):
    """
    It creates an input node from a numpy array. 
    
    Args:
        value: the numpy array, it can hold an arbitrary tensor
        has_sequence_dimension: if the array carries a sequence of elements
        name: the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    return input_reader(value, has_sequence_dimension, var_name=name, alias=name)  
    
def input(dims, name=None):
    """
    It creates an input node, the data will be read by a CNTK reader
    
    Args:
        dims: the shape of the input tensor
        name: the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    return Input(dims, var_name = name)  
    
def parameter(dims=None, name=None, learning_rate_multiplier=1.0, init='uniform', 
              init_value_scale=1, value=0, init_from_file_path='', init_from_literal=None,
              random_seed=-1):
    """
    It creates a parameter tensor. 
    
    Args:
        dims: the shape of the input tensor. If init='fromLiteral', dims is not 
        needed as it will be inferred from the litteral.
        name: the name of the node in the network
        TODO: document the rest of the arguments
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    

    return param(dims, name, learning_rate_multiplier, init, init_value_scale,
              value, init_from_file_path, init_from_literal, random_seed)
    
def constant(value, name=None):
    """
    It creates constant tensor initialized from a numpy array
    
    Args:
        value: the tensor constant passed as numpy array
        name: the name of the node in the network
    Returns:
        :class:`cntk.graph.ComputationNode`
    """    
    
    return parameter(name=name, init='fromLiteral', init_from_literal=value,
                     learning_rate_multiplier=0.0)    